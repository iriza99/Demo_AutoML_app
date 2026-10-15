"""Entry point for a Hugging Face Space on the free tier (Gradio SDK).

Docker Spaces may need a paid plan, but a Gradio Space only installs
requirements.txt (and packages.txt) and runs this file, so the Flask app is
started here with gunicorn on the port the Space exposes. Gradio itself is not
used. The defaults are the public ones (see the Dockerfile and README).
"""
import os
import sys

DEFAULTS = {
    "AUTOML_REQUIRE_USER_KEY": "1",   # every user brings their own Groq key
    "MAX_CONCURRENT_TRAININGS": "1",
    "AUTOML_GPU": "0",
    "MPLCONFIGDIR": "/tmp/matplotlib",
}
for name, value in DEFAULTS.items():
    if not os.environ.get(name, "").strip():  # unset or empty in the Space settings
        os.environ[name] = value
port = os.environ.get("GRADIO_SERVER_PORT", "").strip() or "7860"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# One worker: training state lives in memory, so every request must reach the same process
os.execvp(sys.executable, [sys.executable, "-m", "gunicorn", "--workers", "1", "--threads", "8",
                           "--timeout", "600", "--graceful-timeout", "30",
                           "--bind", f"0.0.0.0:{port}", "app:app"])
