from groq import Groq
from utils.assistant_tools import (NO_TOOLS_INSTRUCTIONS, TOOL_INSTRUCTIONS, build_tools,
                                   extract_text_actions, parse_tool_calls)
from utils.plots import plot_data_as_text, shap_summary_as_text
from utils.prompts import get_system_prompt_chat
import os
import logging
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


def _groq_request(api_key, messages, max_tokens, temperature, tools=None):
    model = groq_model()
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
    messages = [{"role": "system", "content": get_system_prompt_chat(lang)}]
    if extra_system:
        messages.append({"role": "system", "content": extra_system})

    app_state = data.get('app_state') if isinstance(data.get('app_state'), dict) else None
    use_tools = bool(allow_actions and app_state and app_state.get('actions_enabled'))
    if app_state:
        messages.append({"role": "system", "content":
                         "Current state of the app the user is looking at (JSON). If no dataset is loaded, "
                         "tell the user to upload one first.\n" + _state_summary(app_state)})
    messages.append({"role": "system", "content": TOOL_INSTRUCTIONS if use_tools else NO_TOOLS_INSTRUCTIONS})

    context = results_context(data)
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "system", "content":
                     "You cannot see images. Explain plots only from the numbers in this context and never "
                     "invent values. If the data for something is not in the context, say so and explain how "
                     "to obtain it in the app (for example, the SHAP plot is computed with its button in the "
                     "results)."})

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


def _num(value, digits=4):
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def results_context(data):
    """Everything the user can see in the app, as plain text for a text-only model."""
    parts = []
    stats = data.get('stats')
    if isinstance(stats, list) and stats:
        lines = []
        for s in stats[:200]:
            if not isinstance(s, dict):
                continue
            line = f"- {s.get('name')} ({s.get('type')}): missing {s.get('missing')}, distinct {s.get('unique')}"
            if s.get('mean') is not None:
                line += f", mean {_num(s.get('mean'))}, sd {_num(s.get('std'))}, range {_num(s.get('min'))}–{_num(s.get('max'))}"
            elif s.get('top'):
                line += ", most frequent: " + ", ".join(f"{v} ({n})" for v, n in s['top'][:4])
            lines.append(line)
        parts.append("DATASET VARIABLES:\n" + "\n".join(lines))

    result = data.get('result') if isinstance(data.get('result'), dict) else None
    if result:
        lines = [f"Task: {result.get('task_type')}", f"Target: {result.get('target')}",
                 f"Best model: {result.get('best_model')}", f"Training time: {_num(result.get('train_time'), 3)} s",
                 f"Evaluation: {result.get('evaluation') or 'out-of-fold (cross-validation)'}"]
        for name, value in (result.get('metrics') or {}).items():
            lines.append(f"Metric {name}: {_num(value)}")
        if result.get('majority_share') is not None:
            lines.append(f"Accuracy of always predicting the most frequent class: {_num(result['majority_share'], 3)}")
        board = result.get('leaderboard') or []
        if board:
            lines.append("Models tried (validation score, higher is better): " + ", ".join(
                f"{r.get('model')} {_num(r.get('score_val'))}" for r in board[:10] if isinstance(r, dict)))
        parts.append("TRAINING RESULTS:\n" + "\n".join(lines))

    plot_text = plot_data_as_text(data.get('plot_data'))
    if plot_text:
        parts.append("PERFORMANCE PLOTS (exact numbers shown in the app):\n" + plot_text)

    importance = data.get('feature_importance') or {}
    if importance:
        lines = [f"{name}: {_num(value)}" for name, value in
                 sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:20]]
        parts.append("VARIABLE IMPORTANCE (permutation: drop in score when the variable is shuffled; "
                     "near 0 or negative = not useful):\n" + "\n".join(lines))

    shap_text = shap_summary_as_text(data.get('shap_summary'))
    if shap_text:
        parts.append("SHAP PLOT (already computed and visible to the user):\n" + shap_text)

    prediction = data.get('prediction_summary')
    if isinstance(prediction, dict) and prediction:
        parts.append("PREDICTIONS ON NEW DATA (visible to the user):\n" + json.dumps(prediction, ensure_ascii=False)[:3000])
    return "\n\n".join(parts)
