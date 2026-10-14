from groq import Groq
from utils.assistant_tools import (NO_TOOLS_INSTRUCTIONS, TOOL_INSTRUCTIONS, build_tools,
                                   extract_text_actions, parse_tool_calls)
from utils.plots import plot_data_as_text
from utils.prompts import get_system_prompt_chat
import os
import logging
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

    plot_text = plot_data_as_text(data.get('plot_data'))
    if plot_text:
        append_if_exists(messages, "Exact data behind the result plots (use these numbers when explaining the plots):", plot_text)
    if not groq_supports_images():
        messages.append({"role": "system", "content":
                         "You cannot see the plot images. Describe plots only from the numbers given above; "
                         "never invent values. If the data for a plot is not provided, say so and explain "
                         "in general terms how to read it."})

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

