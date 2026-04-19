/* global JSZip, XLSX, Papa */

// Global variables to store state across the app
let lastTrainedModelPath = null;
let lastUsedTargetColumn = null;
let lastCleanedCsvBlob = null;
let lastDatasetName = null;
let summaryResults = {};
let dataSetPreview = null;
let dataSetStats = null;
let dataPreprocessing = {};
let shapPlot = null;
let PlotsPredictionResults = null;
let trainingId = null;
let pollingIntervalTraining = null;
let currentShapRequestId = null;
let pngResultTrainingForPrediction = null;
let appMode = 1; // 1 = train, 2 = predict
let trainingCountdownInterval = null;
let trainingModelInterval = null;
let trainingTotalSeconds = 0;
let detectedTimeColumns = [];
let wizardCurrentStep = 1;
let wizardMaxStep = 1;
let datasetLoaded = false;
let totalDataRows = 0;

// ── Advanced Options ─────────────────────────────────────────────────────────
const ADVANCED_METRICS = {
  classification: [
    { value: 'accuracy',         label: 'Accuracy' },
    { value: 'roc_auc',          label: 'ROC AUC' },
    { value: 'f1_macro',         label: 'F1 Macro' },
  ],
  regression: [
    { value: 'root_mean_squared_error', label: 'RMSE' },
    { value: 'mean_absolute_error',     label: 'MAE' },
    { value: 'r2',                      label: 'R²' },
  ],
  timeseries: [
    { value: 'MASE', label: 'MASE (Mean Absolute Scaled Error)' },
    { value: 'MAPE', label: 'MAPE (Mean Absolute Percentage Error)' },
    { value: 'WQL',  label: 'WQL (Weighted Quantile Loss)' },
  ],
};

const ADVANCED_MODELS = {
  // Order matches AutoGluon's actual training sequence (KNN first, then boosters, then NNs)
  tabular: [
    { value: 'KNN',      label: 'KNeighbors' },
    { value: 'GBM',      label: 'LightGBM' },
    { value: 'XGB',      label: 'XGBoost' },
    { value: 'CAT',      label: 'CatBoost' },
    { value: 'RF',       label: 'Random Forest' },
    { value: 'XT',       label: 'Extra Trees' },
    { value: 'NN_TORCH', label: 'Neural Net (Torch)' },
    { value: 'FASTAI',   label: 'Neural Net (FastAI)' },
  ],
  // AutoGluon TimeSeriesPredictor medium_quality preset ("light" hyperparameters)
  timeseries: [
    { value: 'Naive',                    label: 'Naive' },
    { value: 'SeasonalNaive',            label: 'Seasonal Naive' },
    { value: 'ETS',                      label: 'ETS' },
    { value: 'Theta',                    label: 'Theta' },
    { value: 'RecursiveTabular',         label: 'Recursive Tabular' },
    { value: 'DirectTabular',            label: 'Direct Tabular' },
    { value: 'TemporalFusionTransformer',label: 'TFT' },
    { value: 'Chronos',                  label: 'Chronos' },
  ],
};

function toggleValSplit() {
  const checked = document.getElementById('adv-use-cv').checked;
  const container = document.getElementById('adv-val-split-container');
  if (container) container.style.display = checked ? 'none' : 'flex';
}

function toggleAdvancedOptions() {
  const body = document.getElementById('advanced-options-body');
  const arrow = document.getElementById('advanced-arrow');
  const isOpen = body.style.display !== 'none';
  body.style.display = isOpen ? 'none' : 'block';
  arrow.style.transform = isOpen ? '' : 'rotate(90deg)';
}

function openAdvancedOptions() {
  const body = document.getElementById('advanced-options-body');
  const arrow = document.getElementById('advanced-arrow');
  if (body.style.display === 'none') {
    body.style.display = 'block';
    arrow.style.transform = 'rotate(90deg)';
  }
}

function updateAdvancedOptions() {
  const isTimeSeries = detectedTimeColumns.length > 0;
  const targetCol = document.getElementById('target-column')?.value;
  const targetStat = (dataSetStats || []).find(s => s.name === targetCol);
  const isNumericType = targetStat?.type === 'Numeric';
  const uniqueCount = targetStat?.unique ?? Infinity;
  // Numeric columns with few unique values are likely multiclass (mirrors AutoGluon heuristic)
  const isNumeric = isNumericType && uniqueCount > 20;

  const taskType = isTimeSeries ? 'timeseries' : (isNumeric ? 'regression' : 'classification');

  // Metric dropdown
  const metricSelect = document.getElementById('adv-eval-metric');
  if (metricSelect) {
    metricSelect.innerHTML = '';
    ADVANCED_METRICS[taskType].forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.value;
      opt.textContent = m.label;
      metricSelect.appendChild(opt);
    });
  }

  // Model checkboxes
  const modelList = isTimeSeries ? ADVANCED_MODELS.timeseries : ADVANCED_MODELS.tabular;
  const container = document.getElementById('adv-models-container');
  if (container) {
    container.innerHTML = '';
    modelList.forEach(m => {
      const lbl = document.createElement('label');
      lbl.style.cssText = 'display:flex;align-items:center;gap:4px;padding:3px 8px;border:1px solid #ccc;border-radius:4px;cursor:pointer;font-size:0.88em;background:#fff;';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = m.value;
      cb.name = 'adv-model';
      cb.checked = true;
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(m.label));
      container.appendChild(lbl);
    });
  }

  // Cross-validation: only for tabular
  const cvContainer = document.getElementById('adv-cv-container');
  if (cvContainer) cvContainer.style.display = isTimeSeries ? 'none' : 'flex';

  // Prediction horizon: only for time series
  const predLenContainer = document.getElementById('pred-len-container');
  if (predLenContainer) predLenContainer.style.display = isTimeSeries ? 'flex' : 'none';

  // Auto-open panel when time series detected so user sees prediction horizon
  if (isTimeSeries) openAdvancedOptions();
}

function resetAdvancedOptions() {
  const body = document.getElementById('advanced-options-body');
  const arrow = document.getElementById('advanced-arrow');
  if (body) body.style.display = 'none';
  if (arrow) arrow.style.transform = '';
}
// ─────────────────────────────────────────────────────────────────────────────

function setUploadStatus(key) {
  const el = document.getElementById("upload-status-text");
  if (el) {
    el.textContent = t(key);
  }
}

function toggleFormatsIcon(show) {
  const icon = document.getElementById("upload-formats-icon");
  if (icon) {
    icon.style.display = show ? "inline-block" : "none";
  }
}

function setWizardStep(step) {
  if (!Number.isFinite(step) || step < 1) return;
  if (step > wizardMaxStep) return;
  wizardCurrentStep = step;
  syncWizardPanels(step);
  updateStepper(step);
}

function unlockWizardStep(step) {
  wizardMaxStep = Math.max(wizardMaxStep, step);
}

function syncWizardPanels(activeStep) {
  const panels = document.querySelectorAll(".wizard-panel");
  panels.forEach((panel) => {
    const s = parseInt(panel.dataset.step, 10) || 1;
    const isUnlocked = s <= wizardMaxStep;
    const isActive = s === activeStep;
    panel.classList.toggle("locked", !isUnlocked);
    panel.classList.toggle("active", isActive && isUnlocked);
    panel.style.display = isActive && isUnlocked ? "block" : "none";
  });
  const hero = document.getElementById("training-hero");
  if (hero) {
    hero.style.display = activeStep === 1 ? "block" : "none";
  }
  const uploadSection = document.getElementById("csv-upload-section");
  const preview = document.getElementById("csv-preview");
  if (uploadSection) uploadSection.style.display = activeStep === 1 ? "block" : "none";
  if (preview) preview.style.display = activeStep === 1 && datasetLoaded ? "block" : "none";
}

function updateStepper(stepNumber) {
  const steps = document.querySelectorAll(".stepper .step");
  if (!steps.length) return;
  steps.forEach((step) => {
    const idx = parseInt(step.dataset.step, 10);
    step.classList.toggle("active", idx === stepNumber);
    step.classList.toggle("done", idx < stepNumber && idx <= wizardMaxStep);
    step.classList.toggle("locked", idx > wizardMaxStep);
    const nextUnlocked = wizardMaxStep > wizardCurrentStep ? wizardCurrentStep + 1 : null;
    const shouldHighlight = nextUnlocked && idx === nextUnlocked;
    step.classList.toggle("highlight", shouldHighlight && !step.classList.contains("active") && !step.classList.contains("done"));
  });
}

function isProbablyDate(val) {
  if (val === null || val === undefined) return false;
  const s = `${val}`.trim();
  if (s.length < 6) return false;
  if (!/[\/\-:T]/.test(s)) return false;
  return !isNaN(Date.parse(s));
}

function miniHistogram(values) {
  if (!values || !values.length) return "";
  const bins = 5;
  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  if (vmin === vmax) return "▇▇▇▇▇";
  const width = (vmax - vmin) / bins || 1;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    let idx = Math.floor((v - vmin) / width);
    if (idx >= bins) idx = bins - 1;
    counts[idx] += 1;
  });
  const maxCount = Math.max(...counts) || 1;
  const blocks = ['▁','▂','▃','▅','▇'];
  return counts.map(c => {
    const level = Math.floor((c / maxCount) * (blocks.length - 1));
    return blocks[level];
  }).join('');
}



// On DOM ready: set language and update time display
function toggleDarkMode() {
  const enabled = document.body.classList.toggle("dark-mode");
  localStorage.setItem("darkMode", enabled ? "1" : "0");
  document.getElementById("dm-thumb").classList.toggle("dm-on", enabled);
}

window.addEventListener("DOMContentLoaded", () => {
  const savedLang = localStorage.getItem("selectedLanguage") || "en";
  document.getElementById("language-select").value = savedLang;
  changeLanguage(savedLang);

  if (localStorage.getItem("darkMode") === "1") {
    document.body.classList.add("dark-mode");
    const thumb = document.getElementById("dm-thumb");
    if (thumb) thumb.classList.add("dm-on");
  }

  updateTimeDisplay();
  setWizardStep(1);
  initCsvDropzone();
  const stepper = document.querySelector(".stepper");
  if (stepper) {
    stepper.addEventListener("click", (e) => {
      const stepEl = e.target.closest(".step");
      if (!stepEl) return;
      const step = parseInt(stepEl.dataset.step, 10);
      if (step <= wizardMaxStep) {
        setWizardStep(step);
      }
    });
  }
});

function initCsvDropzone() {
  const dz = document.getElementById("csv-dropzone");
  const input = document.getElementById("upload-csv");
  if (!dz || !input) return;

  const prevent = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ["dragenter", "dragover"].forEach(evt => {
    dz.addEventListener(evt, (e) => {
      prevent(e);
      dz.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach(evt => {
    dz.addEventListener(evt, (e) => {
      prevent(e);
      dz.classList.remove("drag-over");
    });
  });
  dz.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (!files || !files.length) return;
    const file = files[0];
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
  });

  dz.addEventListener("click", () => {
    input.click();
  });
}

// Show or hide the training panel
function showTrainingOverlay(show) {
  const panel = document.getElementById("training-panel");
  if (!panel) return;
  if (show) {
    panel.classList.add("visible");
  } else {
    panel.classList.remove("visible");
  }
}

function stopTrainingCountdown() {
  if (trainingCountdownInterval) {
    clearInterval(trainingCountdownInterval);
    trainingCountdownInterval = null;
  }
  const countdownEl = document.getElementById("training-countdown");
  if (countdownEl) countdownEl.textContent = "--:--";
  const ringEl = document.getElementById("tp-ring-fill");
  if (ringEl) ringEl.style.strokeDashoffset = "238.76";
  const fillEl = document.getElementById("tp-progress-fill");
  if (fillEl) fillEl.style.width = "0%";
  const pctEl = document.getElementById("tp-progress-pct");
  if (pctEl) pctEl.textContent = "0%";
  // Reset panel title and dot if they were changed to finalizing state
  const dot = document.querySelector('.tp-pulse-dot');
  if (dot) dot.classList.remove('finalizing');
  const title = document.querySelector('.tp-title');
  if (title) { title.removeAttribute('data-i18n-active'); title.textContent = t('trainingOverlayTitle') || 'Training model…'; }
}

function showFinalizingState() {
  stopModelAnimation();

  // Amber pulsing dot
  const dot = document.querySelector('.tp-pulse-dot');
  if (dot) dot.classList.add('finalizing');

  // Update title
  const title = document.querySelector('.tp-title');
  if (title) title.textContent = t('finalizingTitle') || 'Generating results…';

  // Keep timer at 0:00 and bar at 100%
  const countdownEl = document.getElementById("training-countdown");
  if (countdownEl) countdownEl.textContent = "0:00";
  const fillEl = document.getElementById("tp-progress-fill");
  if (fillEl) fillEl.style.width = "100%";
  const pctEl = document.getElementById("tp-progress-pct");
  if (pctEl) pctEl.textContent = "100%";

  // Append finalizing rows to model list
  const listEl = document.getElementById("tp-models-list");
  if (!listEl) return;
  const finalizingSteps = [
    t('finalizingOOF')               || 'Computing OOF predictions (unseen data)',
    t('finalizingFeatureImportance') || 'Computing feature importance',
    t('finalizingPlots')             || 'Generating validation plots',
    t('finalizingReport')            || 'Preparing leaderboard',
  ];
  const divider = document.createElement('div');
  divider.className = 'tp-divider';
  divider.id = 'tp-finalizing-divider';
  listEl.appendChild(divider);
  finalizingSteps.forEach((label, i) => {
    const row = document.createElement('div');
    row.className = 'tp-model-row training';
    row.id = `tp-finalizing-${i}`;
    row.innerHTML = `
      <span class="tp-model-icon spin">↻</span>
      <span class="tp-model-name">${label}</span>
    `;
    listEl.appendChild(row);
  });
  listEl.lastElementChild?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function startTrainingCountdown(totalSeconds) {
  stopTrainingCountdown();
  const countdownEl = document.getElementById("training-countdown");
  if (!countdownEl || !Number.isFinite(totalSeconds)) return;

  trainingTotalSeconds = Math.max(1, totalSeconds);
  let remaining = Math.floor(totalSeconds);
  const circumference = 238.76; // 2 * π * 38

  const render = () => {
    const mins = Math.floor(remaining / 60);
    const secs = remaining % 60;
    countdownEl.textContent = `${mins}:${secs.toString().padStart(2, "0")}`;

    const pct = remaining / trainingTotalSeconds;
    const ringEl = document.getElementById("tp-ring-fill");
    if (ringEl) ringEl.style.strokeDashoffset = circumference * (1 - pct);

    const elapsed = (1 - pct) * 100;
    const fillEl = document.getElementById("tp-progress-fill");
    if (fillEl) fillEl.style.width = `${elapsed.toFixed(1)}%`;
    const pctEl = document.getElementById("tp-progress-pct");
    if (pctEl) pctEl.textContent = `${Math.round(elapsed)}%`;
  };

  render();
  trainingCountdownInterval = setInterval(() => {
    remaining = Math.max(0, remaining - 1);
    render();
    if (remaining <= 0) {
      clearInterval(trainingCountdownInterval);
      trainingCountdownInterval = null;
      showFinalizingState();
    }
  }, 1000);
}

// ── Training Panel – model queue animation ────────────────────────────────────

function initTrainingPanel(selectedModels, datasetName, targetCol) {
  const nameEl = document.getElementById("tp-dataset-name");
  const targetEl = document.getElementById("tp-target-col");
  if (nameEl) nameEl.textContent = datasetName;
  if (targetEl) targetEl.textContent = `▸ ${targetCol}`;

  const listEl = document.getElementById("tp-models-list");
  if (!listEl) return;
  listEl.innerHTML = '';
  selectedModels.forEach((model, i) => {
    const row = document.createElement('div');
    row.className = 'tp-model-row pending';
    row.id = `tp-model-${i}`;
    row.innerHTML = `
      <span class="tp-model-icon">○</span>
      <span class="tp-model-name">${model.label}</span>
      <span class="tp-model-badge">${t('modelPending') || 'Pending'}</span>
    `;
    listEl.appendChild(row);
  });
}

function startModelAnimation(nModels, totalSeconds) {
  if (trainingModelInterval) clearInterval(trainingModelInterval);
  if (nModels === 0) return;
  let currentIdx = 0;
  const perModel = Math.max(3000, (totalSeconds * 1000) / nModels);
  _markModelTraining(currentIdx);
  trainingModelInterval = setInterval(() => {
    _markModelDone(currentIdx);
    currentIdx++;
    if (currentIdx < nModels) {
      _markModelTraining(currentIdx);
    } else {
      clearInterval(trainingModelInterval);
      trainingModelInterval = null;
    }
  }, perModel);
}

function _markModelTraining(idx) {
  const row = document.getElementById(`tp-model-${idx}`);
  if (!row) return;
  row.className = 'tp-model-row training';
  const icon = row.querySelector('.tp-model-icon');
  icon.className = 'tp-model-icon spin';
  icon.textContent = '↻';
  row.querySelector('.tp-model-badge').textContent = t('modelTraining') || 'Training';
  row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function _markModelDone(idx) {
  const row = document.getElementById(`tp-model-${idx}`);
  if (!row) return;
  row.className = 'tp-model-row done';
  const icon = row.querySelector('.tp-model-icon');
  icon.className = 'tp-model-icon';
  icon.textContent = '✓';
  row.querySelector('.tp-model-badge').textContent = t('modelDone') || 'Done';
}

function stopModelAnimation() {
  if (trainingModelInterval) {
    clearInterval(trainingModelInterval);
    trainingModelInterval = null;
  }
  document.querySelectorAll('.tp-model-row').forEach(row => {
    if (row.classList.contains('pending') || row.classList.contains('training')) {
      row.className = 'tp-model-row done';
      const icon = row.querySelector('.tp-model-icon');
      if (icon) { icon.className = 'tp-model-icon'; icon.textContent = '✓'; }
      const badge = row.querySelector('.tp-model-badge');
      if (badge) badge.textContent = t('modelDone') || 'Done';
    }
  });
}

function getGroqKey() {
  return localStorage.getItem("groq_api_key");
}

function showGroqModal() {
  const modal = document.getElementById("groq-modal");
  if (!modal) return;
  modal.classList.remove("hidden");
  const input = document.getElementById("groq-api-key-input");
  if (input) {
    input.value = "";
    input.focus();
  }
}

function hideGroqModal() {
  const modal = document.getElementById("groq-modal");
  if (modal) modal.classList.add("hidden");
}


// Update the time limit display based on slider value
function updateTimeDisplay() {
  const slider = document.getElementById("training-time-limit");
  const display = document.getElementById("time-limit-display");

  const minutes = parseInt(slider.value, 10);
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;

  let formatted = "";
  if (minutes === 0) {
    formatted = "0 min";
  } else if (mins === 0) {
    formatted = `${hours}h`;
  } else if (hours === 0) {
    formatted = `${mins} min`;
  } else {
    formatted = `${hours}h${mins}`;
  }

  display.textContent = formatted;
}

function updatePredLenDisplay() {
  const slider = document.getElementById("prediction-length-percent");
  const display = document.getElementById("pred-len-display");
  if (!slider || !display) return;
  display.textContent = `${slider.value}%`;
}

// Toggle settings menu visibility
const settingsToggle = document.getElementById("settings-toggle");
const settingsMenu = document.getElementById("settings-menu");

settingsToggle.addEventListener("click", () => {
  settingsMenu.classList.toggle("hidden");
});

// Close settings menu if clicking outside
document.addEventListener("click", (event) => {
  if (
    !settingsMenu.contains(event.target) &&
    !settingsToggle.contains(event.target)
  ) {
    settingsMenu.classList.add("hidden");
  }
});

// Change language and update UI translations
function changeLanguage(lang) {
  if (!lang) {
    lang = document.getElementById("language-select").value;
  }

  // Save selected language
  localStorage.setItem("selectedLanguage", lang);

  // Apply translations to elements
  const elements = document.querySelectorAll("[data-i18n]");
  elements.forEach((el) => {
    const key = el.getAttribute("data-i18n");
    const val = translations[lang] && translations[lang][key];
    if (!val) return;
    // For help icons, keep the "?" and set title only
    if (el.classList.contains("help-icon")) {
      el.setAttribute("title", val);
    } else {
      el.textContent = val;
    }
  });

  // Special placeholder for chat input
  const chatInput = document.getElementById("chat-input");
  if (translations[lang]["chatPlaceholder"]) {
    chatInput.placeholder = translations[lang]["chatPlaceholder"];
  }
}

// Translation helper
function t(key) {
  const lang = localStorage.getItem("selectedLanguage") || "en";
  return translations[lang] && translations[lang][key] ? translations[lang][key] : key;
}

// Switch between train and predict modes
function switchMode() {
  const mode = document.querySelector('input[name="mode"]:checked').value;

  document.getElementById('training-section').style.display = mode === 'train' ? 'block' : 'none';
  document.getElementById('predict-section').style.display = mode === 'predict' ? 'block' : 'none';

  appMode = mode === 'train' ? 1 : 2;
}

// Initial call to update time display on DOM load
document.addEventListener("DOMContentLoaded", updateTimeDisplay);

// Toggle between dataset and model upload sections
function toggleLoadChoice() {
  const choice = document.querySelector('input[name="load-choice"]:checked').value;
  document.getElementById('csv-upload-section').style.display = (choice === 'dataset') ? 'block' : 'none';
  document.getElementById('model-upload-section').style.display = (choice === 'model') ? 'block' : 'none';
}

// Remove uploaded model file and reset UI
function removeModelFile() {
  const input = document.getElementById('upload-model');
  input.value = '';
  input.style.display = 'inline';
  document.getElementById('model-file-info').style.display = 'none';
  document.getElementById('model-file-name').textContent = '';
  document.getElementById('model-status').style.display = 'none';
}

// Parse ARFF file content into a rows array (header row + data rows)
function parseArff(content) {
  const lines = content.split('\n');
  const columns = [];
  const dataLines = [];
  let inData = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('%')) continue;

    if (inData) {
      dataLines.push(trimmed);
    } else if (trimmed.toLowerCase().startsWith('@attribute')) {
      const match = trimmed.match(/@attribute\s+['"]?([^'"@\s]+)['"]?/i);
      if (match) columns.push(match[1]);
    } else if (trimmed.toLowerCase().startsWith('@data')) {
      inData = true;
    }
  }

  if (columns.length === 0 || dataLines.length === 0) return null;

  const parsed = Papa.parse(dataLines.join('\n'), { header: false, skipEmptyLines: true });
  return [columns, ...parsed.data];
}

// Handle CSV upload, preview, and stats
document.getElementById('upload-csv').addEventListener('change', function () {

  const file = this.files[0];
  const fileName = file ? file.name : "";
  const fileNameLower = fileName.toLowerCase();
  const nameWithoutExtension = fileName.replace(/\.[^/.]+$/, "");  // Supprime la dernière extension

  lastDatasetName = nameWithoutExtension;

  const acceptedExtensions = ['.csv', '.xls', '.xlsx', '.xlsm', '.arff'];
  
  if (file && acceptedExtensions.some(ext => fileNameLower.endsWith(ext))) {
    setUploadStatus("uploadStatusLoading");
    // Hide input and label
    document.querySelectorAll('#upload-csv-label').forEach(el => el.style.display = 'none');

    const fileNameSpan = document.getElementById('csv-file-name');
    const fileInfo = document.getElementById('csv-file-info');
    const selectedWrapper = document.getElementById('csv-selected');
    if (fileNameSpan) fileNameSpan.textContent = file.name;
    if (fileInfo) fileInfo.style.display = 'inline-flex';
    if (selectedWrapper) selectedWrapper.style.display = 'block';
    toggleFormatsIcon(false);
    const dropzone = document.getElementById("csv-dropzone");
    if (dropzone) dropzone.style.display = "none";

    const previewTable = document.getElementById('preview-table');
    previewTable.innerHTML = ''; // Reset table
    const previewLoading = document.getElementById('preview-loading');
    if (previewLoading) previewLoading.style.display = 'inline-block';
    document.getElementById('csv-preview').style.display = 'none';

    const ext = file.name.split('.').pop().toLowerCase();
    const reader = new FileReader();

    reader.onerror = function () {
      const previewLoading = document.getElementById('preview-loading');
      if (previewLoading) previewLoading.style.display = 'none';
      showAlert(t("pleaseSelectFile"), 'warning');
      setUploadStatus("uploadStatusIdle");
    };

    if (ext === 'csv') {
      reader.onload = function (e) {
        const content = e.target.result;
        const parsed = Papa.parse(content, {
          header: false,
          skipEmptyLines: true
        });
        buildPreview(parsed.data);
        setUploadStatus("uploadStatusLoaded");
        toggleFormatsIcon(false);
      };
      reader.readAsText(file);
    } else if (ext === 'arff') {
      reader.onload = function (e) {
        const rows = parseArff(e.target.result);
        if (rows) {
          buildPreview(rows);
          setUploadStatus("uploadStatusLoaded");
          toggleFormatsIcon(false);
        } else {
          const previewLoading = document.getElementById('preview-loading');
          if (previewLoading) previewLoading.style.display = 'none';
          showAlert(t("pleaseSelectFile"), 'warning');
          setUploadStatus("uploadStatusIdle");
        }
      };
      reader.readAsText(file);
    } else if (['xls', 'xlsx', 'xlsm'].includes(ext)) {
      reader.onload = function (e) {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const json = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });
        buildPreview(json);
        setUploadStatus("uploadStatusLoaded");
        toggleFormatsIcon(false);
      };
      reader.readAsArrayBuffer(file);
    }

  } else {
    const previewLoading = document.getElementById('preview-loading');
    if (previewLoading) previewLoading.style.display = 'none';
    showAlert(t("pleaseSelectFile"), 'warning');
    this.value = '';
    setUploadStatus("uploadStatusIdle");
    toggleFormatsIcon(true);
  }
});

// Build statistics for each column in the dataset
function buildStats(rows) {
  const header = rows[0];
  const dataRows = rows.slice(1);
  const numRows = dataRows.length || 1;

  const stats = header.map((col, idx) => {
    const colData = dataRows.map(row => row[idx]).filter(val => val !== "" && val !== null && val !== undefined);
    const isDate = colData.every(val => isProbablyDate(val));
    const isNumeric = !isDate && colData.every(val => !isNaN(parseFloat(val)));

    const parsedData = colData.map(val => isNumeric ? parseFloat(val) : val);
    const missing = dataRows.length - colData.length;

    const stat = {
      name: col,
      type: isDate ? "Date" : (isNumeric ? "Numeric" : "Categorical"),
      missing: missing,
      mean: isNumeric ? (parsedData.reduce((a, b) => a + b, 0) / parsedData.length).toFixed(2) : "-",
      std: isNumeric
        ? Math.sqrt(parsedData.map(x => (x - parsedData.reduce((a, b) => a + b, 0) / parsedData.length) ** 2).reduce((a, b) => a + b, 0) / parsedData.length).toFixed(2)
        : "-",
      unique: colData.length ? new Set(colData).size : 0,
      min: isNumeric ? Math.min(...parsedData).toFixed(2) : "-",
      max: isNumeric ? Math.max(...parsedData).toFixed(2) : "-",
      distribution: isNumeric ? miniHistogram(parsedData) : "",
      qualityClass: missing / numRows > 0.4 ? "quality-bad" : missing / numRows > 0.15 ? "quality-warn" : "quality-good",
    };
    return stat;
  });
  dataSetStats = stats; 
  displayStatsTable(stats);

}

// Display the statistics table in the UI
function displayStatsTable(stats) {
  const table = document.getElementById('stats-table');
  table.innerHTML = '';

  const headerRow = document.createElement('tr');
  const headers = [
    { key: 'statsColumn' },
    { key: 'statsType' },
    { key: 'statsMissing' },
    { key: 'statsUnique' },
    { key: 'statsMinMax' },
    { key: 'statsMeanStd' },
    { key: 'statsDistribution' }
  ];

  headers.forEach(header => {
    const th = document.createElement('th');
    th.setAttribute('data-i18n', header.key);
    th.textContent = t(header.key);
    headerRow.appendChild(th);
  });

  table.appendChild(headerRow);

  stats.forEach(stat => {
    const row = document.createElement('tr');
    row.classList.add(stat.qualityClass);

    const missingPct = totalDataRows ? ((stat.missing / totalDataRows) * 100).toFixed(1) : "0.0";
    const missingText = `${stat.missing} (${missingPct}%)`;
    const minMax = (stat.min !== "-" && stat.max !== "-") ? `${stat.min} / ${stat.max}` : "-";
    const meanStd = (stat.mean !== "-" && stat.std !== "-") ? `${stat.mean} ± ${stat.std}` : "-";

    [stat.name, stat.type, missingText, stat.unique, minMax, meanStd, stat.distribution || "—"].forEach(val => {
      const td = document.createElement('td');
      td.textContent = val;
      row.appendChild(td);
    });
    table.appendChild(row);
  });
}

// Build a preview of the uploaded dataset (first rows, columns, stats)
function buildPreview(rows) {
  const previewTable = document.getElementById('preview-table');
  previewTable.innerHTML = '';
  const previewLoading = document.getElementById('preview-loading');
  if (previewLoading) previewLoading.style.display = 'none';
  detectedTimeColumns = [];
  datasetLoaded = true;
  document.getElementById('csv-preview').style.display = 'block';
  unlockWizardStep(2);
  unlockWizardStep(3);
  updateStepper(wizardCurrentStep);

  const header = rows[0];
  const numColumns = header.length;
  const numRows = rows.length - 1;
  totalDataRows = numRows;

  let nanCount = 0;
  for (let i = 1; i < rows.length; i++) {
    nanCount += rows[i].filter(cell =>
      cell === null || cell === undefined || cell.toString().trim() === "" || cell.toString().toLowerCase() === "nan"
    ).length;
  }

  // Update stats display
  document.getElementById('csv-rows-count').textContent = numRows;
  document.getElementById('csv-columns-count').textContent = numColumns;
  document.getElementById('csv-nan-count').textContent = nanCount;

  // Build table header
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  header.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  previewTable.appendChild(thead);

  // Build table body (first 5 rows)
  const tbody = document.createElement("tbody");
  for (let i = 1; i < Math.min(rows.length, 6); i++) {
    const row = rows[i];
    const tr = document.createElement("tr");
    row.forEach(cell => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  previewTable.appendChild(tbody);

  // Update target column select
  refreshTargetOptions(header);
  // Detect time-like columns using a stricter heuristic
  detectedTimeColumns = header.filter((_, idx) => {
    let parsable = 0;
    let total = 0;
    for (let i = 1; i < rows.length; i++) {
      const v = rows[i][idx];
      if (v !== null && v !== undefined && `${v}`.trim() !== "") {
        total += 1;
        if (isProbablyDate(v)) {
          parsable += 1;
        }
      }
      if (total >= 10) break;
    }
    return total > 0 && parsable / total >= 0.7;
  });

  const timeSelect = document.getElementById("time-column");
  if (timeSelect) {
    timeSelect.innerHTML = "";
    if (detectedTimeColumns.length > 0) {
      detectedTimeColumns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        timeSelect.appendChild(opt);
      });
      document.getElementById("time-column-container").style.display = "block";
      const predLenContainer = document.getElementById("pred-len-container");
      if (predLenContainer) predLenContainer.style.display = "block";
      updatePredLenDisplay();
    } else {
      document.getElementById("time-column-container").style.display = "none";
      const predLenContainer = document.getElementById("pred-len-container");
      if (predLenContainer) predLenContainer.style.display = "none";
    }
  }
  buildStats(rows);
  updateAdvancedOptions();

  buildImputationControls(header, rows);
  document.getElementById("enable-imputation").checked = false;

  // Build drop columns checklist
  const dropListContainer = document.getElementById("drop-columns-list");
  dropListContainer.innerHTML = "";

  header.forEach(col => {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = col;
    checkbox.id = `drop-${col}`;

    const label = document.createElement("label");
    label.setAttribute("for", `drop-${col}`);
    label.textContent = col;
    label.style.marginRight = "15px";

    const container = document.createElement("div");
    container.appendChild(checkbox);
    container.appendChild(label);

    dropListContainer.appendChild(container);
  });

  document.querySelectorAll('#drop-columns-list input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => {
      buildImputationControls(header, rows);
      refreshTargetOptions(header);
    });
  });
}

// Rebuild target select excluding dropped columns
function refreshTargetOptions(header) {
  const targetSelect = document.getElementById("target-column");
  if (!targetSelect) return;

  targetSelect.onchange = updateAdvancedOptions;

  const dropped = new Set(
    Array.from(document.querySelectorAll('#drop-columns-list input[type="checkbox"]:checked')).map(cb => cb.value)
  );

  const current = targetSelect.value;
  targetSelect.innerHTML = "";
  header.forEach(col => {
    if (dropped.has(col)) return;
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    targetSelect.appendChild(option);
  });

  // If previous selection still valid, keep it
  if (current && !dropped.has(current)) {
    targetSelect.value = current;
  }
  updateAdvancedOptions();
}

// Remove uploaded CSV file and reset UI
function removeCSVFile() {
  const input = document.getElementById('upload-csv');
  input.value = '';
  const dropzone = document.getElementById("csv-dropzone");
  if (dropzone) dropzone.style.display = "block";

  const selectedWrapper = document.getElementById('csv-selected');
  const fileInfo = document.getElementById('csv-file-info');
  const fileNameSpan = document.getElementById('csv-file-name');
  if (fileInfo) fileInfo.style.display = 'none';
  if (fileNameSpan) fileNameSpan.textContent = '';
  if (selectedWrapper) selectedWrapper.style.display = 'none';
  document.querySelectorAll('#upload-csv-label').forEach(el => el.style.display = 'inline-block');
  document.getElementById('csv-preview').style.display = 'none';
  const previewTable = document.getElementById('preview-table');
  const statsTable = document.getElementById('stats-table');
  previewTable.innerHTML = '';
  statsTable.innerHTML = '';
  document.getElementById('target-column').innerHTML = '';
  const previewLoading = document.getElementById('preview-loading');
  if (previewLoading) previewLoading.style.display = 'none';
  document.getElementById('csv-preview').style.display = 'none';
  const timeContainer = document.getElementById('time-column-container');
  const predLenContainer = document.getElementById('pred-len-container');
  if (timeContainer) timeContainer.style.display = 'none';
  if (predLenContainer) predLenContainer.style.display = 'none';
  
  const resultsDiv = document.getElementById('training-results');
  resultsDiv.style.display = 'none';
  dataSetPreview = null;
  lastDatasetName = null;
  datasetLoaded = false;
  resetAdvancedOptions();

  currentShapRequestId = null;

  const shapPlot = document.getElementById("shap-plots");
  
  if (shapPlot) { 
    shapPlot.innerHTML = '';
  }
  wizardMaxStep = 1;
  setWizardStep(1);
  setUploadStatus("uploadStatusIdle");
  toggleFormatsIcon(true);
}

document.getElementById("enable-imputation").addEventListener("change", (e) => {
  const isChecked = e.target.checked;
  document.getElementById("imputation-controls").style.display = isChecked ? "block" : "none";
});

// Build imputation controls for missing values
function buildImputationControls(header, data) {
  const enableImputationCheckbox = document.getElementById("enable-imputation");
  const container = document.getElementById("imputation-controls");
  const imputationControlsEnable = document.getElementById("container-enable-imputation");
  container.innerHTML = "";

  const droppedColumns = new Set(
    Array.from(document.querySelectorAll('#drop-columns-list input[type="checkbox"]'))
      .filter(cb => cb.checked)
      .map(cb => cb.value)
  );

  let anyMissing = false;

  header.forEach((colName, colIndex) => {
    if (droppedColumns.has(colName)) return; 

    const missing = data.some((row, i) =>
      i > 0 && (row[colIndex] === "" || row[colIndex] === null || row[colIndex] === undefined || row[colIndex].toString().toLowerCase() === "nan")
    );

    if (!missing) return;

    anyMissing = true;

    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "10px";

    const label = document.createElement("label");
    label.textContent = colName + ": ";
    label.style.marginRight = "10px";
    wrapper.appendChild(label);

    const select = document.createElement("select");
    select.name = `impute-${colName}`;
    select.dataset.col = colName;

    // Only offer mean/median for numeric columns; categorical columns get mode/constant only
    const colValues = data.slice(1)
      .map(row => row[colIndex])
      .filter(v => v !== "" && v !== null && v !== undefined && v.toString().toLowerCase() !== "nan");
    const isNumeric = colValues.length > 0 && colValues.every(v => !isNaN(parseFloat(v)));
    const methods = isNumeric ? ["mean", "median", "mode", "constant"] : ["mode", "constant"];

    methods.forEach(method => {
      const option = document.createElement("option");
      option.value = method;
      option.textContent = t(method);
      option.setAttribute("data-i18n", method);
      select.appendChild(option);
    });

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Constant value";
    input.style.marginLeft = "10px";
    input.style.display = "none";

    select.addEventListener("change", () => {
      input.style.display = (select.value === "constant") ? "inline-block" : "none";
    });

    wrapper.appendChild(select);
    wrapper.appendChild(input);
    container.appendChild(wrapper);
  });

  if (!anyMissing) {
    container.textContent = t("noMissingValues");
    container.style.display = "block";
    container.setAttribute('data-i18n', 'noMissingValues');
    imputationControlsEnable.style.display = "none";
    return;
  }

  imputationControlsEnable.style.display = "block";
  container.style.display = enableImputationCheckbox.checked ? "block" : "none";
}

// Stop the training process on the server
function stopTraining() {
  fetch(`/stop_training/${trainingId}`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        showAlert(t("stopTrainingError"), 'error');
        return;
      }
      if (pollingIntervalTraining) {
        clearInterval(pollingIntervalTraining);
        pollingIntervalTraining = null;
      }

      showAlert(t("trainingStopped"), "info");
      resetTrainingButtons();
    })
    .catch(error => {
      console.error("Erreur lors de l'arrêt de l'entraînement :", error);
      showAlert(t("stopTrainingError"), 'error');
      resetTrainingButtons();
    });
}

// Reset training buttons to initial state
function resetTrainingButtons() {
  document.getElementById('start-training-btn').style.display = "inline-block";
  document.getElementById('stop-training-btn').style.display = "none";
  document.getElementById('training-spinner').style.display = "none";
  stopTrainingCountdown();
  stopModelAnimation();
  showTrainingOverlay(false);
}

// Poll server for training results until ready
function pollTrainingResult(trainingId) {
  pollingIntervalTraining = setInterval(() => {
    fetch(`/training_result/${trainingId}`)
      .then(res => {
        if (res.status === 202) {
          // Not ready yet
          return null;
        }
        return res.json();
      })
      .then(result => {
        if (!result) return;

        clearInterval(pollingIntervalTraining); 
        pollingIntervalTraining = null;

        if (result.error) {
          showAlert(t("trainingErrorGet"), 'error');
        } else {
          renderTrainingResults(result); 
        }
        resetTrainingButtons();
      })
      .catch(_ => {
        clearInterval(pollingIntervalTraining);
        showAlert(t("trainingErrorNetwork"), 'error');
        resetTrainingButtons();
      });
  }, 2000); 
}

function invalidateShapResult() {
  const shapPlot = document.getElementById("shap-plots");
  
  if (shapPlot) { 
    shapPlot.innerHTML = '';
  }
  currentShapRequestId = null;
}

// Start the training process: clean data, apply imputation, send to server
function startTraining() {
  const fileInput = document.getElementById('upload-csv');
  const file = fileInput.files[0];
  const targetColumn = document.getElementById('target-column').value;

  if (!file || !targetColumn) {
    showAlert(t("selectCSVAndTarget"), 'warning');
    return;
  }

  const slider = document.getElementById("training-time-limit");
  const sliderMinutes = slider ? parseInt(slider.value, 10) : 0;
  const totalSeconds = Number.isFinite(sliderMinutes) ? sliderMinutes * 60 : 0;

  invalidateShapResult();

  // Columns to drop
  const checkboxes = document.querySelectorAll('#drop-columns-list input[type="checkbox"]:checked');
  const columnsToDrop = Array.from(checkboxes).map(cb => cb.value);

  if (columnsToDrop.includes(targetColumn)) {
    showAlert(t("cannotDropTargetColumn"), 'error');
    return;
  }

  // Collect selected models for the progress panel
  const selectedModelEls = Array.from(document.querySelectorAll('#adv-models-container input[type="checkbox"]:checked'));
  const panelModels = selectedModelEls.length > 0
    ? selectedModelEls.map(cb => ({ value: cb.value, label: cb.closest('label')?.textContent?.trim() || cb.value }))
    : (detectedTimeColumns.length > 0 ? ADVANCED_MODELS.timeseries : ADVANCED_MODELS.tabular);

  document.getElementById('start-training-btn').style.display = "none";
  document.getElementById('stop-training-btn').style.display = "inline-block";
  document.getElementById('training-spinner').style.display = "inline-block";
  initTrainingPanel(panelModels, file.name, targetColumn);
  startTrainingCountdown(totalSeconds);
  startModelAnimation(panelModels.length, totalSeconds);
  showTrainingOverlay(true);
  unlockWizardStep(3);
  setWizardStep(3);

  const ext = file.name.split('.').pop().toLowerCase();
  const reader = new FileReader();

  reader.onload = function (e) {
    let data = [];
    let header = [];

    if (ext === 'csv') {
      const parsed = Papa.parse(e.target.result, { header: false, skipEmptyLines: true });
      data = parsed.data;
    } else if (ext === 'arff') {
      const rows = parseArff(e.target.result);
      if (!rows) {
        showAlert(t("unsupportedFormat"), 'warning');
        document.getElementById('training-spinner').style.display = "none";
        return;
      }
      data = rows;
    } else if (['xls', 'xlsx', 'xlsm'].includes(ext)) {
      const workbook = XLSX.read(new Uint8Array(e.target.result), { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];
      data = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });
    } else {
      showAlert(t("unsupportedFormat"), 'warning');
      document.getElementById('training-spinner').style.display = "none";
      return;
    }

    // Drop selected columns
    header = data[0];
    const dropIndexes = header.map((name, idx) => columnsToDrop.includes(name) ? idx : -1).filter(i => i !== -1);
    const cleanedData = data.map(row => row.filter((_, idx) => !dropIndexes.includes(idx)));

    // Apply imputation if enabled
    const imputationEnabled = document.getElementById("enable-imputation").checked;
    let imputationChoices = {};
    if (imputationEnabled) {
      document.querySelectorAll('#imputation-controls select').forEach(select => {
        const col = select.dataset.col;
        const method = select.value;
        let constant = null;
        if (method === "constant") {
          const input = select.nextElementSibling;
          constant = input.value;
        }
        imputationChoices[col] = { method, constant };
      });

      // Apply imputation to missing values
      const newHeader = cleanedData[0];
      const colIndexes = Object.keys(imputationChoices).map(col =>
        ({ col, idx: newHeader.indexOf(col) })
      );

      for (let rowIdx = 1; rowIdx < cleanedData.length; rowIdx++) {
        const row = cleanedData[rowIdx];

        for (const { col, idx } of colIndexes) {
          if (row[idx] === "" || row[idx] === null || row[idx] === undefined) {
            const { method, constant } = imputationChoices[col];
            const colValues = cleanedData
              .slice(1)
              .map(r => r[idx])
              .filter(v => v !== "" && v !== null && v !== undefined);

            let fillValue;
            if (method === "mean") {
              const nums = colValues.map(Number).filter(n => !isNaN(n));
              const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
              fillValue = mean.toFixed(2);
            } else if (method === "median") {
              const nums = colValues.map(Number).filter(n => !isNaN(n)).sort((a, b) => a - b);
              const mid = Math.floor(nums.length / 2);
              fillValue = nums.length % 2 === 0
                ? ((nums[mid - 1] + nums[mid]) / 2).toFixed(2)
                : nums[mid].toFixed(2);
            } else if (method === "mode") {
              const freq = {};
              colValues.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
              fillValue = Object.entries(freq).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
            } else if (method === "constant") {
              fillValue = constant;
            }

            row[idx] = fillValue;
          }
        }
      }
    }

    dataPreprocessing = {
      data_preprocessing: {
        columns_drop: columnsToDrop,
        imputation_missing_value: imputationChoices
      }
    };

    const csv = Papa.unparse(cleanedData, { quotes: false });

    // Prepare cleaned file for upload
    const formData = new FormData();
  const blob = new Blob([csv], { type: 'text/csv' });
  formData.append('file', blob, `${file.name}.csv`);
  formData.append('target_column', targetColumn);
  const timeSlider = document.getElementById("training-time-limit");
  const timeLimitSeconds = timeSlider ? parseInt(timeSlider.value, 10) * 60 : 60;
  formData.append('time_limit', timeLimitSeconds);
  const timeColumnSelect = document.getElementById('time-column');
  if (timeColumnSelect && detectedTimeColumns.length > 0) {
    formData.append('time_column', timeColumnSelect.value);
    const predLenSlider = document.getElementById('prediction-length-percent');
    if (predLenSlider) {
      formData.append('prediction_length_percent', predLenSlider.value);
    }
  }

  // Advanced options
  const advMetric = document.getElementById('adv-eval-metric');
  if (advMetric && advMetric.value) formData.append('eval_metric', advMetric.value);

  const excludedModels = Array.from(document.querySelectorAll('#adv-models-container input[type="checkbox"]'))
    .filter(cb => !cb.checked).map(cb => cb.value);
  if (excludedModels.length > 0) formData.append('excluded_model_types', excludedModels.join(','));

  const useCvCb = document.getElementById('adv-use-cv');
  const useCv = useCvCb && useCvCb.checked;
  formData.append('use_cv', useCv ? '1' : '0');
  if (!useCv) {
    const valSize = document.getElementById('adv-val-size');
    const parsed = valSize ? parseInt(valSize.value, 10) : NaN;
    const safeVal = (!isNaN(parsed) && parsed >= 10 && parsed <= 50) ? parsed : 20;
    formData.append('val_size', (safeVal / 100).toFixed(2));
  }

    lastUsedTargetColumn = targetColumn;
    lastCleanedCsvBlob = blob;  

    // Send to server for training
    fetch('/train', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {

      if (data.error) {
        showAlert(t("beginTrainingError"), 'error');
        resetTrainingButtons();
        return;
      }

      const tempTrainingId = data.training_id;
      trainingId = tempTrainingId
      pollTrainingResult(tempTrainingId);

    })
.catch(_ => {
      resetTrainingButtons();
      document.getElementById('training-spinner').style.display = "none";
      showTrainingOverlay(false);
      stopTrainingCountdown();
      showAlert(t("trainingError"), 'error');
    });
  };

  if (ext === 'csv' || ext === 'arff') {
    reader.readAsText(file);
  } else {
    reader.readAsArrayBuffer(file);
  }
}


// Render training results and plots in the UI
function renderTrainingResults(data) {
  lastTrainedModelPath = data.model_path;
  summaryResults["summary"] = data.summary_LLM;
  summaryResults["feature_importance_plot"] = data.feature_importance_plot;
  summaryResults["metrics_plot"] = data.metrics;
  summaryResults["forecast_plot"] = data.forecast_plot || null;
  summaryResults["task_type"] = data.task_type;
  summaryResults["best_model"] = data.best_model;
  summaryResults["prediction_length"] = data.prediction_length || null;
  summaryResults["train_time"] = data.train_time;
  unlockWizardStep(4);
  unlockWizardStep(5);
  setWizardStep(4);
  dataSetPreview = data.markdown_preview; 
  const resultsDiv = document.getElementById('training-results');
  resultsDiv.innerHTML = `<h2 data-i18n="trainingResults">${t("trainingResults")}</h2>`;

  // -------- Training Summary Section --------
  let summaryHTML = `
    <div class="result-section" id="training-summary-section">
      <h3 data-i18n="trainingSummary">${t("trainingSummary")}</h3>

      <div class="subsection">
        <h4 data-i18n="generalInfo">${t("generalInfo")}</h4>
        <p><strong data-i18n="detectedTask">${t("detectedTask")}</strong> ${data.task_type}</p>
        <p><strong data-i18n="selectedModel">${t("selectedModel")}</strong> ${data.best_model}</p>
        <p><strong data-i18n="trainingTime">${t("trainingTime")}</strong> ${data.train_time.toFixed(2)} seconds</p>
      </div>
  `;

  const metrics = data.metrics;
  let metricsHTML = '';
  let plotsHTML = `
      <div class="subsection">
        <h4 data-i18n="plots">${t("plots")}</h4>
  `;

  for (const metric in metrics) {
    const value = metrics[metric].value;
    const plotBase64 = metrics[metric].plot;
    const plotHist = metrics[metric].plot_hist || null;
    const metricLabel = metric.toUpperCase();
    if (value == null) continue;
    const formattedValue = value.toFixed(4);

    metricsHTML += `<tr><td>${metricLabel}</td><td>${formattedValue}</td></tr>`;

    if (metricLabel === "RMSE" && plotBase64 && plotHist) {
      plotsHTML += `
        <div class="plot-card">
          <img src="data:image/png;base64,${plotBase64}" alt="${metricLabel} Plot" />
          <img src="data:image/png;base64,${plotHist}" alt="${metricLabel} Error Distribution" />
        </div>
      `;
    } else if (plotBase64) {
      plotsHTML += `
        <div class="plot-card">
          <img src="data:image/png;base64,${plotBase64}" alt="${metricLabel} Plot" />
        </div>
      `;
    }
  }

  if (data.task_type === "timeseries" && data.forecast_plot) {
    plotsHTML += `
      <div class="plot-card">
        <img src="data:image/png;base64,${data.forecast_plot}" alt="Forecast vs Actual" />
      </div>
    `;
  }

  let metricSectionTitle = "Classification Metrics";
  if (data.task_type === "regression") {
    metricSectionTitle = "Regression Metrics";
  } else if (data.task_type === "timeseries") {
    metricSectionTitle = "Time Series Metrics";
  }

  summaryHTML += `
      <div class="subsection">
        <h4>${metricSectionTitle}</h4>
        <table class="metrics-table">
          <thead><tr><th>${t("metric")}</th><th>${t("value")}</th></tr></thead>
          <tbody>${metricsHTML}</tbody>
        </table>
      </div>
  `;

    if (data.leaderboard && Array.isArray(data.leaderboard)) {
    const leaderboardHTML = `
      <div class="result-section" id="leaderboard-section">
        <h3 data-i18n="leaderboard">${t("leaderboard")}</h3>
        <div class="leaderboard-table-container">
          <table class="metrics-table">
            <thead>
              <tr>
                <th data-i18n="model">${t("model")}</th>
                <th data-i18n="scoreVal">${t("scoreVal")}</th>
                <th data-i18n="fitTime">${t("fitTime")}</th>
                <th data-i18n="predictTime">${t("predictTime")}</th>
              </tr>
            </thead>
            <tbody>
              ${data.leaderboard.map(row => `
                <tr>
                  <td>${row.model}</td>
                  <td>${row.score_val != null ? row.score_val.toFixed(4) : 'N/A'}</td>
                  <td>${row.fit_time != null ? row.fit_time.toFixed(2) + 's' : 'N/A'}</td>
                  <td>${row.pred_time_val != null ? row.pred_time_val.toFixed(2) + 's' : 'N/A'}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
    summaryHTML += leaderboardHTML;
  }

  plotsHTML += `</div>`; // End plots
  summaryHTML += plotsHTML + `</div>`; // End Training Summary
  resultsDiv.innerHTML += summaryHTML;

  // -------- Model Explainability Section --------
  const isTimeseries = data.task_type === "timeseries";
  const hasExplainability = data.feature_importance_plot || !isTimeseries;

  if (hasExplainability) {
    let explainabilityHTML = `
      <div class="result-section" id="model-explainability-section">
        <h3 data-i18n="modelExplainability">${t("modelExplainability")}</h3>
    `;

    if (data.feature_importance_plot) {
      explainabilityHTML += `
        <div class="plot-card">
          <img src="data:image/png;base64,${data.feature_importance_plot}" alt="Feature Importance Plot" />
        </div>
      `;
    }

    if (!isTimeseries) {
      explainabilityHTML += `
        <div class="button-container">
          <div style="display: flex; align-items: center; gap: 15px;">
            <button id="generate-shap-button" onclick="generateShapPlot()" class="generate-shap-button" data-i18n="generateShap">
              ${t("generateShap")}
            </button>
            <span class="help-icon" title="${t('shapExpensiveInfo')}" data-i18n="shapExpensiveInfo">i</span>
            <div id="training-spinner-shap" class="spinner" style="display: none;"></div>
          </div>
        </div>
        <div class="plot-card" id="shap-plots"></div>`;
    }

    explainabilityHTML += `</div>`; // End Model Explainability
    resultsDiv.innerHTML += explainabilityHTML;
  }

  resultsDiv.style.display = 'block';

  // Mirror downloads into the dedicated panel (wizard step 5)
  const downloadsPanel = document.getElementById("downloads-panel");
  if (downloadsPanel) {
    downloadsPanel.innerHTML = `
      <h3 data-i18n="downloadSection">${t("downloadSection") || "Downloads"}</h3>
      <div class="download-buttons-container">
        <a id="download-link" href="${data.download_url || "#"}" data-i18n="downloadModel" class="download-button download-model" download>
          📥 ${t("downloadModel")}
        </a>
        <button class="download-button download-plots" onclick="downloadAllPlots()" data-i18n="downloadPlots">
          📊 ${t("downloadPlots")}
        </button>
        <button class="download-button download-pdf" onclick="downloadPDF()" data-i18n="downloadPDF">
          📄 ${t("downloadPDF")}
        </button>
      </div>
    `;
  }
}

function renderShapResult(result){
  
  const tempShapPlot = result.shap_summary_plot
  shapPlot = tempShapPlot;

  const plotCard = document.getElementById("shap-plots");
  plotCard.innerHTML = `
    <img src="data:image/png;base64,${tempShapPlot}" alt="SHAP Summary" />
  `;
}

function resetShapButtons() {
  const button_generate = document.getElementById("generate-shap-button");
  const spinner = document.getElementById('training-spinner-shap');

  button_generate.disabled = false;
  button_generate.style.display = "inline-block";
  if (spinner) spinner.style.display = "none";
}

// Generate SHAP plot for model explainability
async function generateShapPlot() {
  if (!lastTrainedModelPath || !lastCleanedCsvBlob || !lastUsedTargetColumn) {
    showAlert(t("missingSHAPInfo"), 'error');
    return;
  }

  const shapRequestId = crypto.randomUUID();
  currentShapRequestId = shapRequestId;

  const button_generate = document.getElementById("generate-shap-button");
  const spinner = document.getElementById("training-spinner-shap");

  button_generate.disabled = true;
  spinner.style.display = "inline-block";

  const formData = new FormData();
  formData.append("training_id", trainingId || '');
  formData.append("model_path", lastTrainedModelPath || '');
  formData.append("target_column", lastUsedTargetColumn);
  formData.append("dataset", lastCleanedCsvBlob, "cleaned_data.csv");

  try {
    const response = await fetch("/generate_shap_plot", { method: "POST", body: formData });
    if (!response.ok) {
      resetShapButtons();
      throw new Error(await response.text());
    }

    const { shap_id, error: initError } = await response.json();
    if (initError) throw new Error(initError);

    // Poll until done
    await new Promise((resolve, reject) => {
      const interval = setInterval(async () => {
        if (currentShapRequestId !== shapRequestId) {
          clearInterval(interval);
          return resolve();
        }
        try {
          const pr = await fetch(`/shap_progress/${shap_id}`);
          const data = await pr.json();

          if (data.done) {
            clearInterval(interval);
            if (data.error) return reject(new Error(data.error));
            renderShapResult(data.result);
            button_generate.style.display = "none";
            resolve();
          }
        } catch (e) {
          clearInterval(interval);
          reject(e);
        }
      }, 500);
    });

  } catch (error) {
    showAlert(t("shapGenerationError") + ": " + error.message, "error");
    resetShapButtons();
  } finally {
    spinner.style.display = "none";
  }
}





// Download all plots as a ZIP file
function downloadAllPlots() {
  const zip = new JSZip();
  const images = document.querySelectorAll('.plot-card img');

  images.forEach((img, index) => {
    const base64 = img.src.split(',')[1]; // Get only the base64 part
    const alt = img.alt.replace(/\s+/g, '_').toLowerCase(); // Safe filename
    zip.file(`${alt || 'plot_' + index}.png`, base64, { base64: true });
  });

  // Get dataset name
  const fileInput = document.getElementById('upload-csv');
  const fileName = fileInput.files.length > 0 ? fileInput.files[0].name.replace(/\.csv$/, '') : 'dataset';

  // Generate readable timestamp
  const now = new Date();
  const timestamp = now.toISOString().replace(/[:\-T]/g, '_').split('.')[0]; // ex: 2025_05_30_14_45_12

  // Final ZIP filename
  const finalFileName = `${fileName}_plots_${timestamp}.zip`;

  zip.generateAsync({ type: "blob" })
    .then(function (content) {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(content);
      link.download = finalFileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
}

// Download training results as a PDF
async function downloadPDF() {
  // Check that all required information is available before generating the PDF
  if (!dataSetPreview || !dataSetStats || Object.keys(summaryResults).length === 0 || Object.keys(dataPreprocessing).length === 0) {
    showAlert(t("missingPDFInfo"), 'error');
    return;
  }

  try {
    // Prepare the payload to send to the backend for PDF generation
    const payload = {
      summary: summaryResults,
      preview: dataSetPreview,
      stats: dataSetStats,
      data_preprocessing: dataPreprocessing,
      target_column: lastUsedTargetColumn,
      dataset_name: lastDatasetName
    };
    const groqKey = getGroqKey();
    if (groqKey) {
      payload.groq_api_key = groqKey;
    }
    if (shapPlot) {
      payload.shap_summary_plot = shapPlot; // Include SHAP plot if available
    }
    
    // Send the request to the backend to generate the PDF
    const response = await fetch('/download_pdf', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    // Check if the response is OK
    if (!response.ok) {
      showAlert(t("pdfGenerationError"), 'error');
    }

    // Retrieve the PDF file as a blob
    const blob = await response.blob();

    // Generate a readable filename using the dataset name and current timestamp
    const fileInput = document.getElementById('upload-csv');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name.replace(/\.csv$/, '') : 'dataset';

    const now = new Date();
    const timestamp = now.toISOString().replace(/[:\-T]/g, '_').split('.')[0];
    const finalFileName = `${fileName}_training_summary_${timestamp}.pdf`;
    
    // Create a temporary link to trigger the PDF download
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = finalFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

  } catch (error) {
    // Show an error if PDF generation fails
    showAlert(t("trainingErrorNetwork"), 'error');
  }
}

// Handle prediction CSV upload and show next step
document.getElementById('predict-csv').addEventListener('change', function () {
  const file = this.files[0];
  const acceptedExtensions = ['.csv', '.xls', '.xlsx', '.xlsm', '.arff'];
  if (file && acceptedExtensions.some(ext => file.name.endsWith(ext))) {
    this.style.display = 'none';
    document.getElementById('predict-csv-label').style.display = 'none';
    document.getElementById('predict-file-name').textContent = file.name;
    document.getElementById('predict-file-info').style.display = 'inline-block';

    // Show next step
    document.getElementById('step-2-model').style.display = 'block';
  } else {
    showAlert(t("pleaseSelectCFile"), 'warning');
    this.value = '';
  }
});

// Remove prediction file and reset UI
function removePredictFile() {
  const input = document.getElementById('predict-csv');
  input.value = '';

  document.getElementById('predict-csv-label').style.display = 'inline-block';
  document.getElementById('predict-file-info').style.display = 'none';
  document.getElementById('predict-file-name').textContent = '';

  removePredictModel(); 

  // Hide next steps
  document.getElementById('step-2-model').style.display = 'none';
  document.getElementById('step-3-predict').style.display = 'none';
  document.getElementById('prediction-results').style.display = 'none';
}

// Handle prediction model ZIP upload and show next step
document.getElementById('predict-model-zip').addEventListener('change', function () {
  const file = this.files[0];
  if (file && file.name.endsWith('.zip')) {
    this.style.display = 'none';
    document.getElementById('predict-zip-label').style.display = 'none';
    document.getElementById('predict-model-name').textContent = file.name;
    document.getElementById('predict-model-info').style.display = 'inline-block';

    // Show predict button
    document.getElementById('step-3-predict').style.display = 'block';
  } else {
    showAlert(t("pleaseSelectZIP"), 'warning');
    this.value = '';
  }
});

// Remove prediction model and reset UI
function removePredictModel() {
  const input = document.getElementById('predict-model-zip');
  input.value = '';
  document.getElementById('predict-zip-label').style.display = 'inline-block';
  document.getElementById('predict-model-info').style.display = 'none';
  document.getElementById('predict-model-name').textContent = '';

  document.getElementById('step-3-predict').style.display = 'none';
}

// Run prediction: send dataset and model to server, display results and plots
async function runPrediction() {
  const datasetInput = document.getElementById('predict-csv');
  const modelInput = document.getElementById('predict-model-zip');
  const dataset = datasetInput.files[0];
  const model = modelInput.files[0];

  if (!dataset || !model) {
    showAlert(t("selectDatasetAndModel"), 'warning');
    return;
  }

  const zip = await JSZip.loadAsync(model);
  const formData = new FormData();

  formData.append('dataset', dataset);
  formData.append('zip_model', model);

  const pngFiles = [];

  zip.forEach((relativePath, zipEntry) => {
    if (relativePath.startsWith("plot_train_results/") && relativePath.endsWith(".png")) {
      pngFiles.push(zipEntry);
    }
  });

  pngResultTrainingForPrediction = pngFiles;

  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        showAlert(t(data.error), 'error');
        return;
      }

      if (!data.preview || data.preview.length === 0) {
        showAlert("No prediction available.", 'warning');
        return;
      }

      const plots = data.plots;
      PlotsPredictionResults = plots;

      const predictionResults = document.getElementById('prediction-results');
      predictionResults.innerHTML = `
        <h2 data-i18n="predictionResults">${t("predictionResults")}</h2>
        <div class="result-section">
          <h3 data-i18n="preview">${t("preview")}</h3>
          <div id="prediction-table-container"></div>
        </div>
      `;

      // Build preview table
      const tableContainer = document.getElementById('prediction-table-container');
      const table = document.createElement('table');
      table.id = 'prediction-table';
      table.className = 'preview-table';

      const header = Object.keys(data.preview[0]);
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      header.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      data.preview.forEach(row => {
        const tr = document.createElement('tr');
        header.forEach(col => {
          const td = document.createElement('td');
          td.textContent = row[col];
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      tableContainer.appendChild(table);

      // Display prediction plots
      if (data.plots && Object.keys(data.plots).length > 0) {
        let plotsHTML = `
          <div class="result-section">
            <h3 data-i18n="predictionPlots">${t("predictionPlots")}</h3>
        `;

        for (const [title, base64] of Object.entries(data.plots)) {
          const formattedTitle = title.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          plotsHTML += `
            <div class="plot-card">
              <p><strong data-i18n="formattedTitle">${formattedTitle}</strong></p>
              <img src="data:image/png;base64,${base64}" alt="${formattedTitle}" />
            </div>
          `;
        }

        plotsHTML += `</div>`;
        predictionResults.innerHTML += plotsHTML;
      }

      // Download predictions button
      const downloadUrl = data.download_url || '#';
      predictionResults.innerHTML += `
        <div class="download-buttons-container">
          <a id="prediction-download-link" href="${downloadUrl}" data-i18n="downloadPredictions" class="download-button download-model" download>
            📥 ${t("downloadPredictions")}
          </a>
          <button class="download-button download-plots" data-i18n="downloadPredictionPlots" onclick="downloadAllPredictionPlots()">
            📊 ${t("downloadPredictionPlots")}
          </button>
        </div>
      `;

      predictionResults.style.display = 'block';
    })
    .catch(_ => {
      showAlert(t("predictionError"), 'error');
    });
}

// Download all prediction plots as a ZIP file
function downloadAllPredictionPlots() {
  const zip = new JSZip();
  const images = document.querySelectorAll('#prediction-results .plot-card img');

  images.forEach((img, index) => {
    const base64 = img.src.split(',')[1];
    const alt = img.alt.replace(/\s+/g, '_').toLowerCase();
    zip.file(`${alt || 'plot_' + index}.png`, base64, { base64: true });
  });

  const fileInput = document.getElementById('predict-csv');
  const fileName = fileInput.files.length > 0 ? fileInput.files[0].name.replace(/\.csv$/, '') : 'dataset';

  const now = new Date();
  const timestamp = now.toISOString().replace(/[:\-T]/g, '_').split('.')[0];
  const finalFileName = `${fileName}_prediction_plots_${timestamp}.zip`;

  zip.generateAsync({ type: "blob" }).then(function (content) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(content);
    link.download = finalFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
}

// Send chat message to backend and display AI response
async function sendChat() {
  const input = document.getElementById('chat-input');
  const sendButton = document.querySelector('#chat-footer button');
  const chatBox = document.getElementById('chat-box');
  const lang = localStorage.getItem("selectedLanguage") || "en";
  const message = input.value.trim();

  if (!message) return;

  input.disabled = true;
  sendButton.disabled = true;

  const userMsg = document.createElement('div');
  userMsg.className = 'chat-message user';
  userMsg.textContent = message;
  chatBox.appendChild(userMsg);
  chatBox.scrollTop = chatBox.scrollHeight;
  input.value = '';

  try {
    const payload = {
      message,
      lang
    };
    const groqKey = getGroqKey();
    if (groqKey) {
      payload.groq_api_key = groqKey;
    }
    if (appMode === 1){
      if (Object.keys(summaryResults).length !== 0) {
        payload.summary = {
          text: summaryResults.summary, 
          feature_importance_plot: summaryResults.feature_importance_plot, 
          metrics_plot: summaryResults.metrics_plot,
          forecast_plot: summaryResults.forecast_plot
        };
      if (shapPlot){
          payload.shap_summary_plot = shapPlot
        }
      payload.model_metadata = {
        task_type: summaryResults.task_type,
        best_model: summaryResults.best_model,
        target_column: lastUsedTargetColumn || null,
        prediction_length: summaryResults.prediction_length || null
      };
      }
      if (dataSetStats){
        payload.stats = dataSetStats
      }
      if (dataSetPreview) {
        payload.markdown_preview = dataSetPreview;
      }
      if (dataPreprocessing) {
        payload.data_preprocessing = dataPreprocessing;
      }
      } 
    if (appMode === 2) {  
      if (PlotsPredictionResults) {
        payload.plots_prediction_results = PlotsPredictionResults;
      }
      if (pngResultTrainingForPrediction) {
        const entries = await Promise.all(
          pngResultTrainingForPrediction.map(async entry => ({
            name: entry.name.split('/').pop(),
            base64: await entry.async("base64")
          }))
        );
        payload.png_result_training_for_prediction = Object.fromEntries(
          entries.map(e => [e.name, e.base64])
        );
      }
    }
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorData = await response.json();
      showAlert(t(errorData.error), 'error');
      pngResultTrainingForPrediction = null;
      return;
    }

    const data = await response.json();
    const aiReply = data.response;

    const aiMsg = document.createElement('div');
    aiMsg.className = 'chat-message ai';
    aiMsg.innerHTML = marked.parse(aiReply);
    chatBox.appendChild(aiMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

  } catch (error) {
    const errorMsg = document.createElement('div');
    errorMsg.className = 'chat-message ai';
    errorMsg.textContent = t('error_chat', 'error');
    chatBox.appendChild(errorMsg);
    chatBox.scrollTop = chatBox.scrollHeight;
    pngResultTrainingForPrediction = null;
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
}

// Send chat on Enter (without Shift)
document.getElementById('chat-input').addEventListener('keydown', function (event) {
  // If Enter pressed without Shift
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault(); // prevent newline
    sendChat(); 
  }
});

// Groq API key modal handlers
const groqSaveBtn = document.getElementById("groq-save-btn");
const groqCancelBtn = document.getElementById("groq-cancel-btn");
const groqInput = document.getElementById("groq-api-key-input");

if (groqSaveBtn) {
  groqSaveBtn.addEventListener("click", () => {
    const val = (groqInput?.value || "").trim();
    if (!val) {
      showAlert(t("groqMissing"), "warning", 4000);
      return;
    }
    localStorage.setItem("groq_api_key", val);
    hideGroqModal();
    // Open chat if it was requested
    const sidebar = document.getElementById('chat-sidebar');
    if (sidebar && sidebar.classList.contains('collapsed')) {
      toggleChatSidebar();
    }
  });
}

if (groqCancelBtn) {
  groqCancelBtn.addEventListener("click", hideGroqModal);
}

if (groqInput) {
  groqInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      groqSaveBtn?.click();
    }
  });
}

// Toggle chat sidebar open/collapsed
function toggleChatSidebar() {
  const sidebar = document.getElementById('chat-sidebar');
  const toggleButton = document.getElementById('toggle-chat-btn');
  const wrapper = document.querySelector('.page-wrapper');

  // Only allow opening if Groq API key is set
  if (sidebar.classList.contains('collapsed') && !getGroqKey()) {
    showGroqModal();
    return;
  }

  const isCollapsed = sidebar.classList.toggle('collapsed');

  if (isCollapsed) {
    toggleButton.classList.remove('hidden');
    wrapper.classList.remove('chat-open'); 
  } else {
    toggleButton.classList.add('hidden');
    wrapper.classList.add('chat-open'); 
  }
}

// Show alert message in the UI
function showAlert(message, type = 'error', duration = 10000) {
  const alertBox = document.getElementById('custom-alert');
  alertBox.textContent = message;

  alertBox.className = 'custom-alert'; // reset
  if (type === 'success') alertBox.classList.add('success');
  else if (type === 'warning') alertBox.classList.add('warning');
  else alertBox.classList.add('error');

  alertBox.classList.remove('hidden');

  setTimeout(() => {
    alertBox.classList.add('hidden');
  }, duration);
}
