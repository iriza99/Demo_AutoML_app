# AutoML Assistant — image for Hugging Face Spaces (Docker SDK, paid) or any container host.
# Kept in docker/ so a free (Gradio SDK) Space does not mistake the repo for a Docker Space.
# Build from the project root: docker build -f docker/Dockerfile -t automl-assistant .
FROM python:3.12-slim

# LightGBM needs OpenMP
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /home/user/app

# CPU-only PyTorch first (the default wheel bundles CUDA and is several GB larger)
RUN pip install --no-cache-dir --user torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Public deployment defaults (override in the Space settings if needed):
# every user brings their own Groq key, one training at a time, CPU only.
ENV AUTOML_REQUIRE_USER_KEY=1 \
    MAX_CONCURRENT_TRAININGS=1 \
    AUTOML_GPU=0 \
    PORT=7860

EXPOSE 7860
# One worker: training state lives in memory, so every request must reach the same process.
# Threads serve the page, progress polling and downloads while a training runs.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 8 --timeout 600 --graceful-timeout 30 --bind 0.0.0.0:${PORT} app:app"]
