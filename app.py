from flask import Flask, request, render_template, jsonify, send_file
from utils.model_utils import train_model, predict_model, generate_shap_plot, UserError
from utils.chatbot_utils import handle_chat_request, build_preprocessing_summary, generate_pdf
from datetime import datetime
import shutil
import zipfile
from werkzeug.utils import secure_filename
import pandas as pd
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import threading
from cleanup import cleanup_old_files
from apscheduler.schedulers.background import BackgroundScheduler
import os

import traceback


# Dictionaries to keep track of training threads and stop events
training_threads = {}
stop_events = {}

# Dictionary to store training results
training_results = {}

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

# Reduce verbosity of apscheduler logs
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# Create necessary folders for uploads, models, decompression, and predictions
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DECOMPRESSION_FOLDER = 'decompression'
PREDICTION_FOLDER = 'prediction'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Flask app configuration
app.config['ENV'] = os.getenv("FLASK_ENV", "production")
app.config['DEBUG'] = os.getenv("DEBUG", "False").lower() == "true"
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload

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
    folder_name = f"{dataset_name}_{timestamp}"
    zip_filename = f"model_{dataset_name}_{timestamp}.zip"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path, folder_name, zip_filename

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
        time_limit = int(request.form.get('time_limit', 60))
        if time_limit < 10 or time_limit > 600:
            raise UserError("error_time_limit_range")
        time_column = request.form.get('time_column', None)
        prediction_length_percent = request.form.get('prediction_length_percent', None)

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        logger.debug(f"File received: {filename}")

        # Load dataset based on file extension
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext in ['.xls', '.xlsx', '.xlsm']:
            df = pd.read_excel(path)
        elif ext == '.arff':
            from scipy.io import arff
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        else:
            raise UserError("error_unsupported_file_format")

        # Validate target column
        if target_column not in df.columns:
            raise UserError("error_target_not_in_columns")
        if df[target_column].isnull().all():
            raise UserError("error_target_all_missing")

        markdown_preview = preview_dataset(df)
        training_folder, training_id, zip_filename = create_training_folder(filename)
        logger.info(f"Training folder created: {training_id}")

        stop_event = threading.Event()
        stop_events[training_id] = stop_event

        # Training task to run in a separate thread
        def training_task():
            try:
                metrics = train_model(df, target_column, time_limit, training_folder, stop_event, time_column=time_column, prediction_length_percent=prediction_length_percent)
                zip_path = os.path.join(MODEL_FOLDER, zip_filename)
                shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=training_folder)
                logger.info(f"Model compressed and saved: {zip_filename}")
                metrics['download_url'] = f"/download_model/{zip_filename}"
                metrics['markdown_preview'] = markdown_preview
                metrics['training_id'] = training_id

                training_results[training_id] = metrics
                logger.info(f"Training successfully completed for {training_id}.")

                training_threads.pop(training_id, None)
                stop_events.pop(training_id, None)
            except Exception as e:
                logger.exception(f"Error in training thread")
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
    if training_id in training_results:
        result = training_results.pop(training_id)  
        return jsonify(result)
    else:
        return '', 202  

    
@app.route('/stop_training/<training_id>', methods=['POST'])
def stop_training(training_id):
    # Allow user to stop a running training session
    stop_event = stop_events.get(training_id)
    if stop_event:
        stop_event.set()
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
        data = request.get_json()

        # Récupération des différentes parties
        summary_results = data.get('summary', {})
        dataset_preview = data.get('preview', [])
        dataset_stats = data.get('stats', [])
        preprocessing = data.get('data_preprocessing', {})
        target_column = data.get('target_column', '')
        dataset_name = data.get('dataset_name', '')
        shap_plot = data.get('shap_summary_plot', None)
        
        # Construction du résumé des étapes de prétraitement
        preprocessing_summary = build_preprocessing_summary(preprocessing, groq_api_key=data.get('groq_api_key'))

        # Génération du PDF
        buffer = generate_pdf(
            summary_results,
            dataset_preview,
            dataset_stats,
            preprocessing_summary,
            target_column,
            dataset_name,
            shap_plot,
            groq_api_key=data.get('groq_api_key')
        )

        return send_file(
            buffer,
            as_attachment=True,
            download_name='training_summary.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {traceback.format_exc()}")
        return jsonify({'error': 'PDF generation failed.'}), 500

    

@app.route('/generate_shap_plot', methods=['POST'])
def shap_plot():
    try:
        logger.info("Generating SHAP plot.")
        file = request.files.get('dataset')
        if not file or not allowed_file(file.filename):
            raise UserError("error_no_dataset")

        df = pd.read_csv(file)
        model_path = request.form.get('model_path')
        target_column = request.form.get('target_column')

        if not model_path or not os.path.exists(model_path):
            raise UserError("error_invalid_model_path")
        if not target_column:
            raise UserError("error_missing_target_column")

        result = generate_shap_plot(model_path, df, target_column)
        logger.info("SHAP plot generated successfully.")
        return jsonify(result)

    except UserError as ue:
        logger.warning(f"User error SHAP: {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Critical error SHAP: {traceback.format_exc()}")
        return jsonify({"error": "error_shap_failed"}), 500

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
        identifier = f"{timestamp}_{dataset_name}"

        file_ext = os.path.splitext(dataset_file.filename)[1].lower()
        original_filename = f"{identifier}{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        dataset_file.save(file_path)

        # Decompress model zip file
        decompression_folder = os.path.join(DECOMPRESSION_FOLDER, f"predict_{identifier}")
        os.makedirs(decompression_folder, exist_ok=True)
        zip_path = os.path.join(decompression_folder, "model.zip")
        model_zip.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(decompression_folder)
        except zipfile.BadZipFile:
            raise UserError("error_invalid_zip")

        # Run prediction
        result = predict_model(file_path, decompression_folder)
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
            "plots": plots
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

@app.route('/chat', methods=['POST'])   
def chat():
    try:
        data = request.get_json()
        response = handle_chat_request(data)
        return jsonify({"response": response})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in chat endpoint: {traceback.format_exc()}")
        return jsonify({"error": "error_chat"}), 500
    


if __name__ == '__main__':
    logger.info("Starting Flask server.")

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=False)

