"""Entry point for a Hugging Face Space on the free tier (Gradio SDK).

Without a paid plan, Gradio Spaces run on ZeroGPU hardware. A ZeroGPU Space
must start through Gradio (that is how it registers with the platform),
otherwise it is stopped a few seconds after starting. So this file:

1. imports `spaces` first (it has to come before PyTorch),
2. sends the ZeroGPU startup report that a Gradio launch would send (no extra
   server: Spaces only allow the public port),
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

import uvicorn  # noqa: E402
from a2wsgi import WSGIMiddleware  # noqa: E402

from app import app as flask_app  # noqa: E402

PORT = int(os.environ.get("GRADIO_SERVER_PORT", "").strip() or 7860)


def _gpu_probe():
    """Registered with ZeroGPU so the Space is accepted; the app never calls it."""
    import torch
    return torch.cuda.is_available()


def register_with_zerogpu():
    """What gradio.Blocks.launch() does first on ZeroGPU: report the Space's GPU functions."""
    global _gpu_probe
    if spaces is None:
        return
    _gpu_probe = spaces.GPU(duration=10)(_gpu_probe)
    startup = getattr(getattr(spaces, "zero", None), "startup", None)
    if startup is not None:  # only defined on ZeroGPU hardware
        startup()


if __name__ == "__main__":
    register_with_zerogpu()
    uvicorn.run(WSGIMiddleware(flask_app, workers=16), host="0.0.0.0", port=PORT,
                timeout_keep_alive=30, log_level="info")
