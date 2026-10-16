from flask import Flask, request, render_template, jsonify, send_file
from utils.model_utils import load_dataframe, train_model, predict_model, generate_shap_plot, UserError
from utils.chatbot_utils import handle_chat_request, server_groq_key
from utils.report import build_report
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename
import pandas as pd
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import threading
import atexit
from cleanup import cleanup_old_files
from utils.training_progress import TrainingProgress
from utils.model_signing import sign_model_dir, verify_and_extract, UntrustedModel
from apscheduler.schedulers.background import BackgroundScheduler
import os

import time
import traceback
import uuid

# Trainings that may run at the same time (each one uses every CPU core and several GB of RAM)
MAX_CONCURRENT_TRAININGS = max(1, int(os.getenv('MAX_CONCURRENT_TRAININGS', '').strip() or 1))
# Finished results are kept this long so the page can fetch them again (e.g. after a reload)
RESULT_TTL_SECONDS = 6 * 3600


# Dictionaries to keep track of training threads and stop events
training_threads = {}
stop_events = {}

# Dictionary to store training results
training_results = {}

# Live training progress reported by AutoGluon callbacks: training_id → TrainingProgress
training_progress = {}

# Server-side cache of model paths: training_id → absolute model_path
# Used by the SHAP route so clients never need to transmit filesystem paths.
_trained_model_paths = {}

# SHAP async tasks: shap_id → {progress, done, result, error}
_shap_tasks = {}

# Load environment variables from .env file
load_dotenv()

# Set up logging format
log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Set up rotating file handlers for logs and errors
log_file = "logs/server.log"
file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

error_file = "logs/errors.log"
error_handler = RotatingFileHandler(error_file, maxBytes=5 * 1024 * 1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.WARNING)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

# Remove default handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
is_debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")

logging.basicConfig(
    level=logging.DEBUG if is_debug else logging.INFO,
    handlers=[file_handler, error_handler, stream_handler]
)

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Set up background scheduler for periodic cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_files, 'interval', hours=1)
scheduler.start()
atexit.register(scheduler.shutdown)

# Reduce verbosity of apscheduler logs
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# Absolute base directory (location of app.py) — required for send_file and os.path.exists
# to work correctly regardless of the process's working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
DECOMPRESSION_FOLDER = os.path.join(BASE_DIR, 'decompression')
PREDICTION_FOLDER = os.path.join(BASE_DIR, 'prediction')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Flask app configuration
app.config['ENV'] = os.getenv("FLASK_ENV", "production")
app.config['DEBUG'] = is_debug
# Trained model ZIPs (bagged ensembles) are often tens of MB, so uploads for prediction need room
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_MB', '').strip() or 500) * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'xlsm', 'arff'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Return a markdown preview of the dataset
def preview_dataset(df, max_rows=5):
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception as e:
        logger.warning(f"Error during dataset preview: {e}")
        return f"Error during dataset preview: {e}"

# Create a unique folder for each training session
def create_training_folder(dataset_filename, base_dir=MODEL_FOLDER):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    # The random part makes names unguessable and unique even for simultaneous uploads
    token = uuid.uuid4().hex[:10]
    folder_name = f"{dataset_name}_{timestamp}_{token}"
    zip_filename = f"model_{dataset_name}_{timestamp}_{token}.zip"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path, folder_name, zip_filename

def _load_split(file, train_df, target_column):
    """Read an optional validation/test file and align it to the training columns."""
    if not file or not file.filename:
        return None
    if not allowed_file(file.filename):
        raise UserError("error_unsupported_file_format")
    split = load_dataframe(file)
    missing = [c for c in train_df.columns if c not in split.columns]
    if missing:
        raise UserError("error_split_columns")
    split = split[list(train_df.columns)].reset_index(drop=True)
    if split[target_column].isnull().any():
        raise UserError("error_split_target_missing")
    if len(split) < 2:
        raise UserError("error_split_too_small")
    return split


def _model_path_for(training_id):
    """Model folder of a training run, resolved on the server only (never a client-supplied path)."""
    if training_id in _trained_model_paths:
        return _trained_model_paths[training_id]
    name = secure_filename(training_id or '')
    if not name:
        return None
    root = os.path.realpath(MODEL_FOLDER)
    path = os.path.realpath(os.path.join(root, name))
    if path.startswith(root + os.sep) and os.path.isdir(path):
        return path
    return None


def _purge_old_state():
    """Forget finished runs and SHAP tasks after a while so memory does not grow."""
    now = time.time()
    for tid, result in list(training_results.items()):
        if now - result.get('finished_at', now) > RESULT_TTL_SECONDS:
            training_results.pop(tid, None)
            training_progress.pop(tid, None)
            _trained_model_paths.pop(tid, None)
    for sid, task in list(_shap_tasks.items()):
        if task.get('done') and now - task.get('finished_at', now) > 3600:
            _shap_tasks.pop(sid, None)


scheduler.add_job(_purge_old_state, 'interval', minutes=30)


@app.route('/')
def index():
    logger.info("Displaying home page.")
    return render_template('interface.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        logger.info("Starting model training.")
        file = request.files.get('file')
        if not file or file.filename == '':
            raise UserError("error_no_file")
        if not allowed_file(file.filename):
            raise UserError("error_unsupported_file_format")

        target_column = request.form['target_column'].strip()
        time_limit = int(request.form.get('time_limit', 3600))
        if time_limit < 60 or time_limit > 10800:  # 1 min – 3 h in seconds
            raise UserError("error_time_limit_range")
        time_column = request.form.get('time_column', None)
        prediction_length_percent = request.form.get('prediction_length_percent', None)
        eval_metric = request.form.get('eval_metric', None)
        excluded_str = request.form.get('excluded_model_types', '')
        excluded_model_types = [m.strip() for m in excluded_str.split(',') if m.strip()] or None
        use_cv = request.form.get('use_cv', '1') == '1'
        lang = request.form.get('lang', 'en')
        num_bag_folds = 8 if use_cv else 0
        val_size = None
        if not use_cv:
            try:
                val_size = float(request.form.get('val_size', '0.20'))
                val_size = max(0.10, min(0.50, val_size))
            except (ValueError, TypeError):
                val_size = 0.20

        running = [t for t in training_threads.values() if t.is_alive()]
        if len(running) >= MAX_CONCURRENT_TRAININGS:
            raise UserError("error_training_busy", 429)

        filename = secure_filename(file.filename) or "dataset.csv"
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:10]}_{filename}")
        file.save(path)
        logger.debug(f"File received: {filename}")

        df = load_dataframe(path)

        # Validate target column
        if target_column not in df.columns:
            raise UserError("error_target_not_in_columns")
        if df[target_column].isnull().all():
            raise UserError("error_target_all_missing")

        # Optional user-defined validation and test sets (same columns as the training file)
        val_df = _load_split(request.files.get('val_file'), df, target_column)
        test_df = _load_split(request.files.get('test_file'), df, target_column)
        if (val_df is not None or test_df is not None) and time_column:
            raise UserError("error_split_timeseries")

        markdown_preview = preview_dataset(df)
        training_folder, training_id, zip_filename = create_training_folder(filename)
        logger.info(f"Training folder created: {training_id}")

        stop_event = threading.Event()
        stop_events[training_id] = stop_event
        progress = TrainingProgress(time_limit=time_limit)
        training_progress[training_id] = progress

        # Training task to run in a separate thread
        def training_task():
            try:
                metrics = train_model(df, target_column, time_limit, training_folder, stop_event,
                                     time_column=time_column, prediction_length_percent=prediction_length_percent,
                                     eval_metric=eval_metric, excluded_model_types=excluded_model_types,
                                     num_bag_folds=num_bag_folds, val_size=val_size, progress=progress,
                                     lang=lang, val_df=val_df, test_df=test_df)
                zip_path = os.path.join(MODEL_FOLDER, zip_filename)
                sign_model_dir(training_folder)
                shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=training_folder)
                logger.info(f"Model compressed and saved: {zip_filename}")
                metrics['download_url'] = f"/download_model/{zip_filename}"
                metrics['markdown_preview'] = markdown_preview
                metrics['training_id'] = training_id

                # Cache model path server-side so SHAP route never needs a client-supplied path
                if metrics.get('model_path'):
                    _trained_model_paths[training_id] = metrics['model_path']

                progress.set_phase('done')
                metrics['finished_at'] = time.time()
                training_results[training_id] = metrics
                logger.info(f"Training successfully completed for {training_id}.")

                training_threads.pop(training_id, None)
                stop_events.pop(training_id, None)
            except UserError as ue:
                logger.warning(f"User error in training thread: {ue.message_key}")
                training_results[training_id] = {"error": ue.message_key, "finished_at": time.time()}
                training_threads.pop(training_id, None)
                stop_events.pop(training_id, None)
            except Exception as e:
                logger.exception(f"Error in training thread")
                # Report the failure so the client stops polling
                training_results[training_id] = {"error": "error_training_failed", "finished_at": time.time()}
                training_threads.pop(training_id, None)
                stop_events.pop(training_id, None)

        # Start training in a background thread
        training_thread = threading.Thread(target=training_task, daemon=True)
        training_threads[training_id] = training_thread
        training_thread.start()

        return jsonify({"training_id": training_id})

    except UserError as ue:
        logger.warning(f"User error during training: {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Critical error during training: {traceback.format_exc()}")
        return jsonify({"error": "error_training_failed"}), 500
    
@app.route('/training_result/<training_id>', methods=['GET'])
def get_training_result(training_id):
    # Return training result if available, else 202 Accepted
    # Results are not removed when read: a lost response or a page reload can ask again
    if training_id in training_results:
        return jsonify(training_results[training_id])
    if training_id in training_threads or training_id in training_progress:
        return '', 202
    # Unknown id: the server restarted (or the result expired) while the page was waiting
    return jsonify({"error": "error_training_lost"}), 404

    
@app.route('/training_progress/<training_id>', methods=['GET'])
def get_training_progress(training_id):
    progress = training_progress.get(training_id)
    if progress is None:
        return jsonify({"error": "training_id_not_found"}), 404
    return jsonify(progress.to_dict())


@app.route('/stop_training/<training_id>', methods=['POST'])
def stop_training(training_id):
    # Allow user to stop a running training session
    stop_event = stop_events.get(training_id)
    if stop_event:
        stop_event.set()
        if training_id in training_progress:
            training_progress[training_id].stopping = True
        return jsonify({"message": "training_stopped"})
    else:
        return jsonify({"error": "training_id_not_found"}), 404


@app.route('/download_model/<zip_filename>')
def download_model(zip_filename):
    # Allow user to download the trained model as a zip file
    zip_path = os.path.join(MODEL_FOLDER, secure_filename(zip_filename))
    if not os.path.exists(zip_path):
        logger.warning(f"Download failed: file not found {zip_filename}")
        return "File not found", 404
    logger.info(f"Model download: {zip_filename}")
    return send_file(zip_path, as_attachment=True)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json(silent=True) or {}
        buffer = build_report(data, lang=data.get('lang', 'en'))
        return send_file(buffer, as_attachment=True, download_name='training_report.pdf',
                         mimetype='application/pdf')
    except Exception:
        logger.error(f"Error generating PDF: {traceback.format_exc()}")
        return jsonify({'error': 'PDF generation failed.'}), 500


@app.route('/generate_shap_plot', methods=['POST'])
def shap_plot():
    try:
        logger.info("Generating SHAP plot.")
        file = request.files.get('dataset')
        if not file or not allowed_file(file.filename):
            raise UserError("error_no_dataset")

        df = load_dataframe(file)
        target_column = request.form.get('target_column')

        model_path = _model_path_for(request.form.get('training_id', ''))

        if not model_path or not os.path.exists(model_path):
            raise UserError("error_invalid_model_path")
        if not target_column:
            raise UserError("error_missing_target_column")

        lang = request.form.get('lang', 'en')
        shap_id = uuid.uuid4().hex[:12]
        _shap_tasks[shap_id] = {'progress': 0, 'done': False, 'result': None, 'error': None}

        def _worker():
            try:
                result = generate_shap_plot(model_path, df, target_column, lang=lang)
                _shap_tasks[shap_id].update({'progress': 100, 'done': True, 'result': result, 'finished_at': time.time()})
                logger.info("SHAP plot generated successfully.")
            except Exception as exc:
                _shap_tasks[shap_id].update({'done': True, 'error': str(exc), 'finished_at': time.time()})
                logger.error(f"SHAP worker error: {traceback.format_exc()}")

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({'shap_id': shap_id})

    except UserError as ue:
        logger.warning(f"User error SHAP: {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Critical error SHAP: {traceback.format_exc()}")
        return jsonify({"error": "error_shap_failed"}), 500


@app.route('/shap_progress/<shap_id>', methods=['GET'])
def shap_progress(shap_id):
    task = _shap_tasks.get(shap_id)
    if not task:
        return jsonify({"error": "not_found"}), 404
    return jsonify({
        'progress': task['progress'],
        'done': task['done'],
        'error': task['error'],
        'result': task['result'] if task['done'] and not task['error'] else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Starting prediction.")
        dataset_file = request.files.get('dataset')
        model_zip = request.files.get('zip_model')

        if not dataset_file or not model_zip:
            raise UserError("error_missing_dataset_or_model")
        if not allowed_file(dataset_file.filename):
            raise UserError("error_unsupported_file_format")

        # Create unique identifier for prediction session
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = os.path.splitext(secure_filename(dataset_file.filename))[0]
        identifier = f"{timestamp}_{dataset_name}_{uuid.uuid4().hex[:10]}"

        file_ext = os.path.splitext(dataset_file.filename)[1].lower()
        original_filename = f"{identifier}{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        dataset_file.save(file_path)

        # Decompress model zip file
        decompression_folder = os.path.join(DECOMPRESSION_FOLDER, f"predict_{identifier}")
        os.makedirs(decompression_folder, exist_ok=True)
        zip_path = os.path.join(decompression_folder, "model.zip")
        model_zip.save(zip_path)
        model_dir = os.path.join(decompression_folder, "model")
        os.makedirs(model_dir, exist_ok=True)

        # Models are pickles: only load ZIPs signed by this server (see utils/model_signing.py)
        try:
            verify_and_extract(zip_path, model_dir)
        except UntrustedModel as exc:
            logger.warning(f"Rejected model ZIP for prediction: {exc}")
            shutil.rmtree(decompression_folder, ignore_errors=True)
            raise UserError("error_model_not_trusted")

        # Run prediction
        result = predict_model(file_path, model_dir, lang=request.form.get('lang', 'en'))
        predictions = result['predictions']
        plots = result['plots']

        # Save predictions to CSV
        pred_filename = f"{identifier}_predictions.csv"
        pred_path = os.path.join(PREDICTION_FOLDER, pred_filename)
        pd.DataFrame(predictions).to_csv(pred_path, index=False)

        logger.info(f"Predictions generated successfully: {pred_filename}")
        return jsonify({
            "preview": predictions[:5],
            "download_url": f"/download_prediction/{pred_filename}",
            "plots": plots,
            "summary": result.get('summary'),
        })

    except UserError as ue:
        logger.warning(f"User error prediction: {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Critical error prediction: {traceback.format_exc()}")
        return jsonify({"error": "error_prediction_failed"}), 500

@app.route('/download_prediction/<filename>')
def download_prediction(filename):
    # Allow user to download prediction results
    filepath = os.path.join(PREDICTION_FOLDER, secure_filename(filename))
    if not os.path.exists(filepath):
        logger.warning(f"Prediction file not found: {filename}")
        return "File not found", 404
    logger.info(f"Prediction download: {filename}")
    return send_file(filepath, as_attachment=True)

@app.route('/assistant_status')
def assistant_status():
    # Only reports whether a key exists, never the key itself
    return jsonify({"server_key": bool(server_groq_key())})

@app.route('/chat', methods=['POST'])   
def chat():
    try:
        data = request.get_json()
        result = handle_chat_request(data)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in chat endpoint: {traceback.format_exc()}")
        # Pass the provider's reason (e.g. retired model, invalid key) on to the user
        detail = getattr(e, 'message', None) or str(e)
        return jsonify({"error": "error_chat", "detail": detail[:300]}), 500
    


if __name__ == '__main__':
    logger.info("Starting Flask server.")

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=False)

