# AutoML Assistant

A user-friendly web interface to train, evaluate, and deploy machine learning models on any tabular or time series dataset — no data science background required. Powered by [AutoGluon](https://auto.gluon.ai/) with a conversational LLM assistant to guide you through every step.

---

## Features

- **Automated model training** — classification, regression, and time series forecasting
- **No-code, three-step interface** — *Data* (upload and review every variable) → *Train* (pick the target, time and optional settings) → *Results* (metrics, plots, downloads)
- **Honest evaluation** — out-of-fold metrics with optional holdout split
- **Explainability** — variable importance and SHAP beeswarm plots (TreeExplainer on the best tree model, about a second)
- **LLM assistant** — answers questions about your data and results in plain language, remembers the conversation and can *propose* actions (exclude variables, change the target, metric or time, start training, open a results tab). Each proposal appears as a card and only runs when you click *Apply*; actions are disabled in study mode
- **Dark mode** and multilingual UI (English, French, Spanish)
- **Download** trained models, plots, a PDF report and predictions; predict straight away with the model you just trained

---

## User study mode

The `study/` folder contains a controlled user study on **competence-aware adaptive guidance**: planted-trap clinical datasets, a study mode (`?study=1`) with telemetry, an implicit two-dimensional (ML × clinical domain) competence model, generic pitfall detectors and a local Jev-style typed decision layer. See [`study/README.md`](study/README.md) and [`study/TRAPS.md`](study/TRAPS.md).

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
│   ├── study.js / study.css # User-study mode (only with ?study=1)
│
└── utils/
    ├── model_utils.py       # Training, evaluation, SHAP, plots
    ├── chatbot_utils.py     # LLM integration (Groq)
    └── prompts.py           # Prompt templates
```

---

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
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model for the assistant and the PDF report. Change it when Groq retires a model. |
| `GROQ_VISION` | `0` | Set to `1` only if `GROQ_MODEL` accepts images; the assistant then also sees the plots. |
| `AUTOML_GPU` | `auto` | `auto` trains XGBoost, CatBoost and the neural networks on the GPU when PyTorch sees one; `0` forces CPU. |
| `AUTOML_PARALLEL_FOLDS` | `0` | `1` fits cross-validation folds in parallel with Ray. Faster, but in a long-running server later trainings can lose most models to Ray "Broken pipe" errors. |
| `MAX_UPLOAD_MB` | `500` | Maximum upload size (trained model ZIPs are usually 10–50 MB). |

**Using an NVIDIA GPU.** On Windows, `pip` installs a CPU-only PyTorch by default. Install a CUDA build that matches your driver, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu124`, and check it with
`python -c "import torch; print(torch.cuda.is_available())"`. When a GPU is used, the results show a *GPU* badge.

### Run

```bash
flask run
```

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
