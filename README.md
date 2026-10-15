---
title: AutoML Assistant
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: hf_space.py
pinned: false
short_description: Train, evaluate and explain ML models without code
---

# AutoML Assistant

A user-friendly web interface to train, evaluate, and deploy machine learning models on any tabular or time series dataset — no data science background required. Powered by [AutoGluon](https://auto.gluon.ai/) with a conversational LLM assistant to guide you through every step.

---

## Features

- **Automated model training** — classification, regression, and time series forecasting
- **No-code, three-step interface** — *Data* (upload and review every variable) → *Train* (pick the target, time and optional settings) → *Results* (metrics, plots, downloads)
- **Honest evaluation** — out-of-fold metrics with optional holdout split
- **Results in plain language** — a one-line verdict, all metrics together, automatic checks (identifier-like or dominant variables, class imbalance) with one-click fixes, and interactive charts (confusion matrix, ROC, predicted vs actual)
- **Explainability** — variable importance and an interactive SHAP plot with a summary in words (TreeExplainer on the best tree model, about a second)
- **Expert mode** (settings) — technical details, configuration and every model tried, open by default
- **LLM assistant** — answers questions about your data and results in plain language using the exact numbers behind every chart (including SHAP and predictions), remembers the conversation and can *propose* actions (exclude variables, change the target, metric or time, start training, open a results section). Each proposal appears as a card and only runs when you click *Apply*
- **PDF report** — built from the app's own data, in the interface language, no assistant key needed
- **Example datasets** — breast cancer (classification) and diabetes (regression) one click away on the start screen ([sources](static/examples/README.md))
- **Picks up where you left off** — the dataset, settings and last results are kept in the browser (IndexedDB) and restored after a reload, also while a training is running; *Change file* starts from scratch. Trained models stay on the server for 24 h
- **Dark mode** and multilingual UI (English, French, Spanish)
- **Download** trained models, plots, a PDF report and predictions; predict straight away with the model you just trained

---

## Project Structure

```
├── app.py                   # Main Flask application
├── cleanup.py               # Periodic cleanup of old files
├── requirements.txt         # Python dependencies
│
├── models/                  # Trained AutoML models (.zip)
├── uploads/                 # User-uploaded datasets
├── decompression/           # Temporary folder for unzipped models
├── prediction/              # Prediction output files
├── logs/                    # Application logs
│
├── templates/
│   └── interface.html       # Main UI
│
├── static/
│   ├── lang.js              # i18n strings
│   ├── app.js               # Frontend logic
│   ├── style.css            # Design tokens, components, dark mode
│
└── utils/
    ├── model_utils.py       # Training, evaluation, SHAP
    ├── plots.py             # Figures in the app style and the data behind them
    ├── report.py            # PDF report
    ├── training_progress.py # Live training progress and GPU detection
    ├── assistant_tools.py   # Actions the assistant can propose
    ├── chatbot_utils.py     # LLM integration (Groq)
    └── prompts.py           # Prompt templates
```

---

## Deploy on Hugging Face Spaces

The repository is ready for a free Space. Hugging Face installs `requirements.txt` (CPU build of PyTorch) and `packages.txt`, then runs `hf_space.py`, which starts the Flask app with gunicorn on port 7860. The **Gradio** SDK is used only because it is free; Gradio itself is not used. The block at the top of this README is the Space configuration. (With a paid plan, the `Dockerfile` works too: change `sdk:` to `docker` and add `app_port: 7860`.)

1. On huggingface.co: **New Space** → name it → **SDK: Gradio** (template *Blank*) → hardware *CPU basic* (free) → create.
2. In the Space: **Settings → Variables and secrets → New secret** `AUTOML_SECRET_KEY` with a long random value (e.g. the output of `python -c "import secrets; print(secrets.token_hex(32))"`). It signs the trained models; without it, models downloaded before a restart cannot be used to predict.
3. Push the code (from the project folder; use a Hugging Face *write* token as the password). The Space starts with a sample commit, so the first push needs `--force`:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push --force space claude/wizardly-curie-egb6z4:main
   ```
   The first build takes about 10–15 minutes (*Logs* tab). Later pushes rebuild automatically.

What the public Space does differently from running it locally:

- **Each user enters their own Groq key** (`AUTOML_REQUIRE_USER_KEY=1`); a `GROQ_API_KEY` on the server is never used, so nobody spends your quota. The key stays in the user's browser.
- **One training at a time** (`MAX_CONCURRENT_TRAININGS=1`): a second user is asked to retry in a few minutes. Each training uses every CPU core.
- **Only models trained by this app can be used to predict.** Models are Python pickles, and loading one runs code, so every trained model is signed and an uploaded ZIP is loaded only if its signature and files check out.
- **Storage is temporary**: uploads and predictions are deleted after 1 h, models after 24 h, and everything when the Space restarts or sleeps (after 48 h without visits on free hardware). Results already on screen stay in the user's browser.
- Do not upload identifiable patient data to a public Space.

## Installation

### Prerequisites

- Python 3.10+
- pip
- Recommended: virtual environment

### Steps

```bash
git clone https://github.com/iriza99/Demo_AutoML_app.git
cd Demo_AutoML_app

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key
FLASK_DEBUG=1
```

Get a free API key at [console.groq.com/keys](https://console.groq.com/keys). Do **not** commit this file.
Without a key in `.env`, the app asks for one in the browser the first time the assistant is opened.

Optional settings (all can go in `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model for the assistant. Change it when Groq retires a model. |
| `AUTOML_GPU` | `auto` | `auto` trains XGBoost, CatBoost and the neural networks on the GPU when PyTorch sees one; `0` forces CPU. |
| `AUTOML_PARALLEL_FOLDS` | `0` | `1` fits cross-validation folds in parallel with Ray. Faster, but in a long-running server later trainings can lose most models to Ray "Broken pipe" errors. |
| `MAX_UPLOAD_MB` | `500` | Maximum upload size (trained model ZIPs are usually 10–50 MB). |
| `AUTOML_SECRET_KEY` | random, saved in `.secret_key` | Signs trained models; only signed models are accepted for prediction. Keep it stable. |
| `AUTOML_REQUIRE_USER_KEY` | `0` | `1` makes every user enter their own Groq key; `GROQ_API_KEY` is then ignored. |
| `MAX_CONCURRENT_TRAININGS` | `1` | Trainings allowed at the same time. |

**Using an NVIDIA GPU.** On Windows, `pip` installs a CPU-only PyTorch by default. Install a CUDA build that matches your driver, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu124`, and check it with
`python -c "import torch; print(torch.cuda.is_available())"`. When a GPU is used, the results show a *GPU* badge.

### Run

```bash
python app.py            # development
gunicorn --workers 1 --threads 8 --timeout 600 -b 0.0.0.0:5000 app:app   # production (Linux)
```

Use a single worker: training state lives in memory.

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## How to Use

### Training

1. Upload a CSV dataset
2. Preview the data, remove unwanted columns, handle missing values
3. Select the target column
4. Choose task type (tabular or time series)
5. Configure training options (time limit, cross-validation, etc.)
6. Launch training and review:
   - Performance metrics
   - Model leaderboard
   - Confusion matrix / ROC curve (classification)
   - Forecast plot (time series)
   - Feature importance and SHAP plots
7. Download the trained model

### Prediction

1. Upload a trained model (`.zip`)
2. Upload a new dataset (CSV)
3. Run predictions and download the output CSV

---

## LLM Assistant

The built-in chatbot can:

- Explain how to use the interface
- Describe metrics and plots in plain language
- Suggest improvements to your dataset or training setup
- Interpret feature importance and SHAP results

---

## Example Workflow

1. Upload `sales_data.csv`
2. Select `revenue` as the target column
3. Set training time to 120 seconds
4. Launch training
5. Review metrics and feature importance
6. Download the best model
7. Upload `new_data.csv` for predictions
8. Download results
