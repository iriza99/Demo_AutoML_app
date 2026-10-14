from groq import Groq
from utils.assistant_tools import (NO_TOOLS_INSTRUCTIONS, TOOL_INSTRUCTIONS, build_tools,
                                   extract_text_actions, parse_tool_calls)
from utils.prompts import get_system_prompt_chat, get_system_prompt_preprocessing, get_system_prompt_explanation_plot
import os
import logging
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, SimpleDocTemplate, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_LEFT
from datetime import datetime
import io
import base64
import re
import json

def server_groq_key():
    """Groq key from the environment (.env is loaded by app.py after this import)."""
    return os.getenv('GROQ_API_KEY')


# Groq retired meta-llama/llama-4-scout-17b-16e-instruct on 2026-07-17; the model
# is configurable so the next retirement only needs a .env change.
DEFAULT_GROQ_MODEL = 'openai/gpt-oss-120b'


def groq_model():
    return os.getenv('GROQ_MODEL', DEFAULT_GROQ_MODEL)


def groq_supports_images():
    """Only send plot images when GROQ_VISION=1 names a vision-capable model."""
    return os.getenv('GROQ_VISION', '0').strip().lower() in ('1', 'true', 'yes')


def _text_only(messages):
    """Drop image parts for text-only models, keeping any accompanying text."""
    out = []
    for m in messages:
        content = m.get('content')
        if isinstance(content, list):
            continue  # image messages only carry a caption about the image itself
        out.append(m)
    return out


def _groq_request(api_key, messages, max_tokens, temperature, tools=None):
    model = groq_model()
    if not groq_supports_images():
        messages = _text_only(messages)
    kwargs = {}
    if model.startswith('openai/gpt-oss'):
        # Reasoning tokens count towards the completion budget; keep them short.
        # Sent through extra_body because older groq SDKs do not know the argument.
        kwargs['extra_body'] = {'reasoning_effort': 'low'}
        max_tokens = max_tokens * 2
    if tools:
        kwargs['tools'] = tools
        kwargs['tool_choice'] = 'auto'
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
        **kwargs,
    )
    return completion.choices[0].message


def groq_complete(api_key, messages, max_tokens=1024, temperature=0.7):
    """Single entry point for plain Groq chat completions."""
    return _groq_request(api_key, messages, max_tokens, temperature).content or ''


def groq_chat_with_tools(api_key, messages, tools, max_tokens=1024, temperature=0.3, attempts=2):
    """Completion that may return tool calls; falls back to plain text if tool use keeps failing."""
    for attempt in range(attempts):
        try:
            message = _groq_request(api_key, messages, max_tokens, temperature, tools=tools)
            return message.content or '', getattr(message, 'tool_calls', None) or []
        except Exception as exc:
            # Groq answers 400 "tool_use_failed" when the model writes a malformed call
            if 'tool' not in str(exc).lower():
                raise
            logging.warning(f"Tool use failed (attempt {attempt + 1}/{attempts}): {exc}")
    # Answer in words: swap the tool instructions for "explain which button to press"
    plain = [m for m in messages if m.get('content') != TOOL_INSTRUCTIONS]
    insert_at = max(i for i, m in enumerate(plain) if m['role'] == 'system') + 1
    plain.insert(insert_at, {"role": "system", "content": NO_TOOLS_INSTRUCTIONS})
    return groq_complete(api_key, plain, max_tokens, temperature), []


def clean_markdown(text):
    # Remove markdown bold and italic markers from text
    return re.sub(r'(\*\*|\*)', '', text)

MAX_HISTORY = 12


def _history(data):
    """Previous turns sent by the browser: plain user/assistant text only."""
    out = []
    for item in (data.get('history') or [])[-MAX_HISTORY:]:
        if isinstance(item, dict) and item.get('role') in ('user', 'assistant') and isinstance(item.get('content'), str):
            out.append({'role': item['role'], 'content': item['content'][:2000]})
    return out


def _state_summary(state):
    """Compact JSON of what the user currently sees in the app."""
    try:
        return json.dumps(state, ensure_ascii=False, default=str)[:6000]
    except (TypeError, ValueError):
        return ''


def handle_chat_request(data, extra_system='', allow_actions=True):
    """
    Answer a chat message with the app context, and optionally propose app actions.

    Returns {"response": text, "actions": [{"name", "args"}, ...]}. Actions are
    proposals only; the browser applies them when the user confirms.
    """
    user_input = data.get('message', '')
    if not user_input:
        raise ValueError("error_empty_message")

    # Allow overriding the API key via payload, fallback to env
    groq_key = data.get('groq_api_key') or server_groq_key()
    if not groq_key:
        raise ValueError("groqMissing")

    lang = data.get('lang', 'en')
    summary = data.get('summary')
    model_metadata = data.get('model_metadata') or {}
    dataset = data.get('markdown_preview')
    stats = data.get('stats')
    shap_plot = data.get('shap_summary_plot')
    plot_predictions = data.get('plots_prediction_results')
    training_pred_plot = data.get('png_result_training_for_prediction')
    # Start with a system prompt for the chat
    messages = [{"role": "system", "content": get_system_prompt_chat(lang)}]
    if extra_system:
        messages.append({"role": "system", "content": extra_system})

    app_state = data.get('app_state') if isinstance(data.get('app_state'), dict) else None
    use_tools = bool(allow_actions and app_state and app_state.get('actions_enabled'))
    if app_state:
        messages.append({"role": "system", "content":
                         "Current state of the app the user is looking at (JSON). If no dataset is loaded, "
                         "tell the user to upload one first.\n" + _state_summary(app_state)})
    if use_tools:
        messages.append({"role": "system", "content": TOOL_INSTRUCTIONS})

    # Append dataset preview, stats, training summary if available
    append_if_exists(messages, "Aperçu du dataset fourni :", dataset)
    append_if_exists(messages, "Statistiques du dataset :", stats)
    if summary:
        append_if_exists(messages, "Résumé des résultats d'entraînement :", summary.get('text'))
        # Add metrics values as text
        metrics_plot = summary.get("metrics_plot", {})
        if metrics_plot:
            metric_lines = []
            for k, v in metrics_plot.items():
                try:
                    val = v.get("value")
                    if val is not None:
                        metric_lines.append(f"{k.upper()}: {val:.4f}")
                except Exception:
                    continue
            if metric_lines:
                append_if_exists(messages, "Métriques:", "\n".join(metric_lines))
        # Add model metadata text
        meta_lines = []
        if model_metadata.get("task_type"):
            meta_lines.append(f"Tâche: {model_metadata['task_type']}")
        if model_metadata.get("best_model"):
            meta_lines.append(f"Meilleur modèle: {model_metadata['best_model']}")
        if model_metadata.get("target_column"):
            meta_lines.append(f"Colonne cible: {model_metadata['target_column']}")
        if model_metadata.get("prediction_length"):
            meta_lines.append(f"Longueur de prédiction: {model_metadata['prediction_length']}")
        if meta_lines:
            append_if_exists(messages, "Métadonnées du modèle:", "\n".join(meta_lines))

    importance = data.get('feature_importance') or {}
    if importance:
        lines = [f"{name}: {value:.4f}" for name, value in
                 sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:15]]
        append_if_exists(messages, "Variable importance (permutation, higher = more important):", "\n".join(lines))

    # Add metric plots from summary if present
    if summary:
        add_metric_plots_from_summary(messages, summary)

    # Add SHAP plot if present
    if shap_plot:
        add_image_message(messages, "Voici le graphe SHAP de l'entraînement.", shap_plot)

    # Add prediction result plots if present
    if plot_predictions:
        for name, plot in plot_predictions.items():
            add_image_message(messages, f"Voici un des graphes {name} des résultats de prédiction.", plot)

    # Add training prediction plots if present
    if training_pred_plot:
        for name, plot in training_pred_plot.items():
            add_image_message(messages, f"Voici un des graphes {name} qui montre les résultats de l'entrainement du modèle qui sert à faire les prédictions.", plot)

    # Earlier turns, then the user's new message
    messages.extend(_history(data))
    messages.append({"role": "user", "content": user_input})

    if not use_tools:
        return {"response": groq_complete(groq_key, messages, max_tokens=1024, temperature=0.7), "actions": []}
    text, tool_calls = groq_chat_with_tools(groq_key, messages, build_tools(app_state))
    actions = parse_tool_calls(tool_calls, app_state)
    if not actions:
        # Some answers describe the action as JSON instead of calling the tool
        text, actions = extract_text_actions(text, app_state)
    return {"response": text, "actions": actions}

def append_if_exists(messages, title, content):
    # Append a message to the conversation if content is not empty
    if content:
        messages.append({"role": "user", "content": f"{title}\n{content}"})

def add_image_message(messages, caption, base64_str):
    # Add an image (base64-encoded) with a caption to the conversation
    try:
        if not base64_str:
            return
        # Validate base64 to avoid Groq 400 invalid image data
        base64.b64decode(base64_str, validate=True)
    except Exception:
        logging.exception("Invalid base64 image, skipping.")
        return

    messages.append({"role": "user", "content": [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
    ]})

def add_metric_plots_from_summary(messages, summary):
    # Add feature importance plot and metric plots from the summary to the conversation
    if "feature_importance_plot" in summary:
        add_image_message(messages, "Voici le graphe de l'importance des variables.", summary["feature_importance_plot"])
    if "forecast_plot" in summary and summary["forecast_plot"]:
        add_image_message(messages, "Prévision vs valeurs réelles", summary["forecast_plot"])
    for metric_name, metric_obj in summary.get("metrics_plot", {}).items():
        for plot_type in ['plot', 'plot_hist']:
            if plot_type in metric_obj:
                base64_plot = metric_obj[plot_type]
                if base64_plot:
                    add_image_message(messages, f"Voici le graphe {metric_name} ({plot_type}).", base64_plot)

def build_preprocessing_summary(preprocessing_summary: dict, lang='en', groq_api_key=None) -> str:
    """
    Generates a natural language summary of the preprocessing steps using the Groq API.
    If no API key is available, returns a basic text summary of the preprocessing dict.
    """
    key = groq_api_key or server_groq_key()
    if not key:
        return f"Preprocessing steps: {preprocessing_summary}"

    system_prompt = get_system_prompt_preprocessing()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(preprocessing_summary)}  
    ]

    try:
        return groq_complete(key, messages, max_tokens=512, temperature=0.5)
    except Exception:
        logging.exception("Groq call failed for preprocessing summary")
        return f"Preprocessing steps: {preprocessing_summary}"

def generate_explanation_plot(plot = None, name = None, value = None, groq_api_key=None):
    """
    Generates an explanation for a given plot and metric value using the Groq API.
    """
    key = groq_api_key or server_groq_key()
    if not key:
        return f"Plot {name} (value={value})"
    messages = [{"role": "system", "content": get_system_prompt_explanation_plot()}]
    if plot and not groq_supports_images():
        # Text-only model: explain from the name and value instead of the image
        plot = None
    if plot and value:
        add_image_message(messages, f"Here the graph {name} to explain and the value of the metric {value}", plot)
    elif plot:
        add_image_message(messages, f"Here the graph {name} to explain.", plot)
    else:
        messages.append({"role": "user", "content": f"Please explain the metric {name} with the value {value}."})
    
    try:
        return clean_markdown(groq_complete(key, messages, max_tokens=512, temperature=0.5))
    except Exception:
        logging.exception("Groq call failed for explanation plot")
        return f"Plot {name} (value={value})"

def generate_pdf(summary_results, dataset_preview, dataset_stats, preprocessing_summary,
                 target_column, dataset_name, shap_plot, output_dir="pdf_summary", groq_api_key=None):
    """
    Generates a PDF summary report of the training results, including dataset preview,
    statistics, preprocessing, metrics, plots, and explanations.
    """
    # === Prepare output directory ===
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{dataset_name}_{timestamp}.pdf"
    output_path = os.path.join(output_dir, pdf_filename)

    # === Create PDF document ===
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    # Title style and content
    title_style = styles['Heading1']
    title_style.alignment = 1

    today_str = datetime.now().strftime("%B %d, %Y")
    title_text = f"Training Summary - {dataset_name} ({today_str})"
    elements.append(Paragraph(title_text, title_style))
    elements.append(Spacer(1, 20))

    # === 1. Dataset Overview ===
    elements.append(Paragraph("1. Dataset Overview", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Dataset preview (markdown-like)
    elements.append(Paragraph("<b>Dataset Preview (first 5 rows)</b>", styles['Normal']))
    
    # Parse the markdown preview into a table
    preview_lines = dataset_preview.strip().split('\n')
    preview_data = [
        line.strip('|').split('|')
        for line in preview_lines
        if '|' in line and not re.match(r'^\s*-{2,}', line.strip().replace('|', '').strip())
    ]
    preview_data = [[cell.strip() for cell in row] for row in preview_data]

    # Detect the number of columns
    max_columns = len(preview_data[0])

    # Dynamically adjust column widths
    page_width = A4[0] - doc.leftMargin - doc.rightMargin
    col_width = page_width / max_columns
    col_widths = [col_width] * max_columns

    # Apply word wrap to table cells using Paragraph
    cell_style = ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7 if max_columns > 8 else 9,
        leading=10,
        alignment=TA_LEFT,
        wordWrap='CJK',  # force word wrap
    )

    # Replace cell values with Paragraphs for wrapping
    wrapped_data = []
    for row in preview_data:
        wrapped_row = [Paragraph(cell, cell_style) for cell in row]
        wrapped_data.append(wrapped_row)

    # Create the table for dataset preview
    table = Table(wrapped_data, colWidths=col_widths, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Dataset statistics table
    elements.append(Paragraph("<b>Dataset Statistics</b>", styles['Normal']))
    stat_table_data = [["Column", "Type", "Missing", "Mean", "Std"]]
    for col in dataset_stats:
        stat_table_data.append([
            col['name'],
            col['type'],
            str(col['missing']),
            col['mean'],
            col['std']
        ])
    stat_table = Table(stat_table_data, hAlign='LEFT')
    stat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 12))

    # Preprocessing summary section
    elements.append(Paragraph("<b>Preprocessing Summary</b>", styles['Normal']))
    elements.append(Paragraph(preprocessing_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Target column section
    elements.append(Paragraph(f"<b>Target Column:</b> {target_column}", styles['Normal']))
    elements.append(Spacer(1, 20))

    elements.append(PageBreak())

    # === 2. Training Results ===
    elements.append(Paragraph("2. Training Results", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Add task type, best model, and training time
    elements.append(Paragraph(f"- Detected Task Type: <b>{summary_results['task_type']}</b>", styles['Normal']))
    elements.append(Paragraph(f"- Best Model: <b>{summary_results['best_model']}</b>", styles['Normal']))
    elements.append(Paragraph(f"- Training Time: <b>{round(summary_results['train_time'], 2)} seconds</b>", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Metrics table
    metrics_data = [["Metric", "Score"]]
    for metric, data in summary_results['metrics_plot'].items():
        metrics_data.append([metric.upper(), f"{data['value']:.4f}"])
    metric_table = Table(metrics_data, hAlign='LEFT')
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(Paragraph("<b>Evaluation Metrics</b>", styles['Normal']))
    elements.append(metric_table)
    elements.append(Spacer(1, 20))

    # Metric explanations (one per metric, right after the table)
    for metric, data in summary_results['metrics_plot'].items():
        score = round(data.get('value', 0.0), 4)
        explanation = generate_explanation_plot(plot=None, name=metric, value=score, groq_api_key=groq_api_key)
        if explanation:
            elements.append(Paragraph(f"<b>{metric.upper()}:</b> {explanation}", styles['Normal']))
            elements.append(Spacer(1, 12))

    # === 3. Visual Results ===
    elements.append(Paragraph("3. Training Visualizations", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # === Visualization of metrics (plots if available) ===
    for metric, data in summary_results['metrics_plot'].items():
        # Title and score
        score = round(data.get('value', 0.0), 4)
        elements.append(Paragraph(f"<b>{metric.upper()} Score</b>: {score}", styles['Normal']))
        
        # Display all available plots (plot and/or plot_hist)
        for plot_type in ['plot', 'plot_hist']:
            if plot_type in data:
                image_data = base64.b64decode(data[plot_type])
                image_stream = io.BytesIO(image_data)
                img = Image(image_stream, width=400, height=250)
                elements.append(img)
                elements.append(Spacer(1, 6))

                # Explanation for each plot (uses Groq if key provided)
                explanation = generate_explanation_plot(
                    plot=data.get(plot_type),
                    name=f"{metric} ({plot_type})",
                    value=score,
                    groq_api_key=groq_api_key
                )
                if explanation:
                    elements.append(Paragraph(explanation, styles['Normal']))
                    elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 18))

    # Feature importance plot and explanation
    if 'feature_importance_plot' in summary_results:
        elements.append(Paragraph("<b>Feature Importance</b>", styles['Normal']))
        feature_img_data = base64.b64decode(summary_results['feature_importance_plot'])
        feature_img_stream = io.BytesIO(feature_img_data)
        feature_img = Image(feature_img_stream, width=400, height=250)
        elements.append(feature_img)

        explanation = generate_explanation_plot(plot=summary_results['feature_importance_plot'], name="Feature Importance", value=None, groq_api_key=groq_api_key)
        if explanation:
            elements.append(Paragraph(explanation, styles['Normal']))

        elements.append(Spacer(1, 18))
    if shap_plot:
        elements.append(Paragraph("<b>SHAP Summary Plot</b>", styles['Normal']))
        shap_img_data = base64.b64decode(shap_plot)
        shap_img_stream = io.BytesIO(shap_img_data)
        shap_img = Image(shap_img_stream, width=400, height=250)
        elements.append(shap_img)

        explanation = generate_explanation_plot(plot=shap_plot, name="SHAP Summary", value=None, groq_api_key=groq_api_key)
        if explanation:
            elements.append(Paragraph(explanation, styles['Normal']))

    # Build the PDF document and return the buffer
    doc.build(elements)
    buffer.seek(0)  
    return buffer
