"""Entry point for a Hugging Face Space on the free tier (Gradio SDK).

Without a paid plan, Gradio Spaces run on ZeroGPU hardware. A ZeroGPU Space
must start through Gradio (that is how it registers with the platform),
otherwise it is stopped a few seconds after starting. So this file:

1. imports `spaces` first (it has to come before PyTorch),
2. launches a tiny, unused Gradio app on an internal port, which registers
   the Space,
3. serves the normal Flask app (the same interface as locally) on port 7860.

Training runs on CPU; the GPU is not used. The defaults are the public ones
(each user brings their own Groq key, one training at a time).
"""
import os

DEFAULTS = {
    "AUTOML_REQUIRE_USER_KEY": "1",   # every user brings their own Groq key
    "MAX_CONCURRENT_TRAININGS": "1",
    "AUTOML_GPU": "0",
    "MPLCONFIGDIR": "/tmp/matplotlib",
}
for name, value in DEFAULTS.items():
    if not os.environ.get(name, "").strip():  # unset or empty in the Space settings
        os.environ[name] = value
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import spaces  # noqa: E402  (must be imported before torch on ZeroGPU)
except ImportError:  # running outside Hugging Face
    spaces = None

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from a2wsgi import WSGIMiddleware  # noqa: E402

from app import app as flask_app  # noqa: E402

PORT = int(os.environ.get("GRADIO_SERVER_PORT", "").strip() or 7860)
INTERNAL_PORT = PORT + 1


def _gpu_probe():
    """Registered with ZeroGPU so the Space starts; the app never calls it."""
    import torch
    return torch.cuda.is_available()


if spaces is not None:
    _gpu_probe = spaces.GPU(duration=10)(_gpu_probe)

with gr.Blocks(title="AutoML Assistant") as _registration:
    gr.Markdown("AutoML Assistant")

# Launching any Gradio app is what registers a ZeroGPU Space; this one is not exposed
_registration.launch(server_name="127.0.0.1", server_port=INTERNAL_PORT, prevent_thread_lock=True,
                     show_error=False, quiet=True)

if __name__ == "__main__":
    uvicorn.run(WSGIMiddleware(flask_app, workers=16), host="0.0.0.0", port=PORT,
                timeout_keep_alive=30, log_level="info")
