// AutoML Assistant — frontend logic.
// Flow: 1 · Data (upload, review variables) → 2 · Train (target, time, options) → 3 · Results.

// ── State ────────────────────────────────────────────────────────────────────
let datasetRows = null;          // parsed rows, header first
let datasetFile = null;
let dataSetStats = null;         // per-column stats (also sent to the assistant / PDF)
let dataSetPreview = null;       // markdown preview returned by the server
let dataPreprocessing = {};
let summaryResults = {};
let lastResult = null;
let lastTrainedModelPath = null;
let lastUsedTargetColumn = null;
let lastCleanedCsvBlob = null;
let lastDatasetName = null;
let shapPlot = null;
let PlotsPredictionResults = null;
let pngResultTrainingForPrediction = null;
let trainingId = null;
let pollingIntervalTraining = null;
let currentShapRequestId = null;
let appMode = 1;                 // 1 = train, 2 = predict
let detectedTimeColumns = [];
let wizardCurrentStep = 1;
let wizardMaxStep = 1;
let datasetLoaded = false;
let totalDataRows = 0;

const STEP = { DATA: 1, TRAIN: 2, RESULTS: 3 };
const CDN = {
  xlsx: "https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js",
  jszip: "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js",
};

const ADVANCED_METRICS = {
  classification: [
    { value: "accuracy", label: "Accuracy" },
    { value: "roc_auc", label: "ROC AUC" },
    { value: "f1_macro", label: "F1 macro" },
  ],
  regression: [
    { value: "root_mean_squared_error", label: "RMSE" },
    { value: "mean_absolute_error", label: "MAE" },
    { value: "r2", label: "R²" },
  ],
  timeseries: [
    { value: "MASE", label: "MASE" },
    { value: "MAPE", label: "MAPE" },
    { value: "WQL", label: "WQL" },
  ],
};

const ADVANCED_MODELS = {
  tabular: [
    { value: "GBM", label: "LightGBM" },
    { value: "CAT", label: "CatBoost" },
    { value: "XGB", label: "XGBoost" },
    { value: "RF", label: "Random Forest" },
    { value: "XT", label: "Extra Trees" },
    { value: "NN_TORCH", label: "Neural net" },
    { value: "FASTAI", label: "FastAI" },
  ],
  timeseries: [
    { value: "Naive", label: "Naive" },
    { value: "SeasonalNaive", label: "Seasonal naive" },
    { value: "ETS", label: "ETS" },
    { value: "AutoARIMA", label: "AutoARIMA" },
    { value: "Theta", label: "Theta" },
    { value: "DirectTabular", label: "Direct tabular" },
    { value: "RecursiveTabular", label: "Recursive tabular" },
    { value: "DeepAR", label: "DeepAR" },
    { value: "TemporalFusionTransformer", label: "TFT" },
    { value: "PatchTST", label: "PatchTST" },
  ],
};

// ── Small helpers ────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function currentLang() {
  try { return localStorage.getItem("selectedLanguage") || "es"; } catch (_) { return "es"; }
}

function t(key, params) {
  const lang = currentLang();
  let text = (translations[lang] && translations[lang][key]) || (translations.en && translations.en[key]) || key;
  if (params) Object.entries(params).forEach(([k, v]) => { text = text.replace(`{${k}}`, v); });
  return text;
}

function fmt(value, digits = 3) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(currentLang(), { maximumFractionDigits: digits });
}

function pct(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `${Number(value * 100).toLocaleString(currentLang(), { maximumFractionDigits: digits })} %`;
}

const _scripts = {};
function loadScript(src) {
  if (!_scripts[src]) {
    _scripts[src] = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => { delete _scripts[src]; reject(new Error(`Could not load ${src}`)); };
      document.head.appendChild(s);
    });
  }
  return _scripts[src];
}

function isMissing(v) {
  return v === null || v === undefined || `${v}`.trim() === "" || `${v}`.toLowerCase() === "nan";
}

function isProbablyDate(val) {
  if (typeof val !== "string") return false;
  const s = val.trim();
  if (!/[-/:T]/.test(s) || s.length < 6) return false;
  return !isNaN(Date.parse(s));
}

// Safe inline-handler argument: a JSON string literal, HTML-escaped for the attribute
function jsArg(value) {
  return esc(JSON.stringify(String(value)));
}

// crypto.randomUUID only exists on https/localhost; the app is often opened via a LAN IP
function uid() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function downloadBlob(blob, filename) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 5000);
}

function timestamp() {
  return new Date().toISOString().replace(/[:\-T]/g, "_").split(".")[0];
}

function datasetBaseName() {
  return (lastDatasetName || "dataset").replace(/\.[^/.]+$/, "");
}

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  const lang = currentLang();
  $("language-select").value = lang;
  changeLanguage(lang);
  $("theme-toggle").checked = document.documentElement.dataset.theme === "dark";

  initSettingsMenu();
  initDropzone();
  initPrediction();
  initChat();
  initTimePresets();

  $("stepper").addEventListener("click", (e) => {
    const step = parseInt(e.target.closest(".step")?.dataset.step, 10);
    if (step && step <= wizardMaxStep) setWizardStep(step);
  });
  $("target-column").addEventListener("change", onTargetChange);
  $("enable-imputation").addEventListener("change", (e) => {
    $("imputation-controls").hidden = !e.target.checked;
  });
  $("adv-eval-metric").addEventListener("change", updateTrainSummary);
  $("time-column").addEventListener("change", updateTrainSummary);

  updateTimeDisplay();
  setWizardStep(STEP.DATA);
});

function initSettingsMenu() {
  const toggle = $("settings-toggle");
  const menu = $("settings-menu");
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.classList.toggle("hidden");
  });
  document.addEventListener("click", (e) => {
    if (!menu.contains(e.target) && !toggle.contains(e.target)) menu.classList.add("hidden");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      menu.classList.add("hidden");
      if (!$("chat-sidebar").classList.contains("collapsed")) toggleChatSidebar();
    }
  });
}

function toggleDarkMode() {
  const dark = $("theme-toggle").checked;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (_) { /* private mode */ }
}

function changeLanguage(lang) {
  lang = lang || $("language-select").value;
  try { localStorage.setItem("selectedLanguage", lang); } catch (_) { /* private mode */ }
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const val = translations[lang]?.[node.dataset.i18n] || translations.en?.[node.dataset.i18n];
    if (val) node.textContent = val;
  });
  $("chat-input").placeholder = t("chatPlaceholder");
  // Re-render dynamic parts in the new language
  if (datasetLoaded) {
    renderDatasetChips();
    renderVariablesTable();
    renderTargetInsight();
    updateTrainSummary();
  }
  if (lastResult) renderTrainingResults(lastResult, { silent: true });
  updateTimeDisplay();
  renderChatSuggestions();
  updateStepSubtitles();
}

function setMode(mode) {
  const train = mode === "train";
  $("training-section").hidden = !train;
  $("predict-section").hidden = train;
  $("mode-train").classList.toggle("active", train);
  $("mode-predict").classList.toggle("active", !train);
  $("mode-train").setAttribute("aria-selected", String(train));
  $("mode-predict").setAttribute("aria-selected", String(!train));
  appMode = train ? 1 : 2;
  renderChatSuggestions();
}

// ── Wizard ───────────────────────────────────────────────────────────────────
function setWizardStep(step) {
  if (!Number.isFinite(step) || step < 1 || step > wizardMaxStep) return;
  wizardCurrentStep = step;
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = parseInt(panel.dataset.step, 10) !== step;
  });
  updateStepper();
  if (step === STEP.TRAIN) updateTrainSummary();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function unlockWizardStep(step) {
  wizardMaxStep = Math.max(wizardMaxStep, step);
  updateStepper();
}

function updateStepper() {
  document.querySelectorAll("#stepper .step").forEach((node) => {
    const idx = parseInt(node.dataset.step, 10);
    node.classList.toggle("active", idx === wizardCurrentStep);
    node.classList.toggle("done", idx !== wizardCurrentStep && idx < wizardMaxStep);
    node.classList.toggle("locked", idx > wizardMaxStep);
  });
  updateStepSubtitles();
}

function updateStepSubtitles() {
  const sub1 = $("step-sub-1"), sub2 = $("step-sub-2"), sub3 = $("step-sub-3");
  if (!sub1) return;
  sub1.textContent = datasetLoaded
    ? `${lastDatasetName} · ${totalDataRows} × ${datasetRows[0].length}` : t("stepDataSub");
  const target = $("target-column").value;
  sub2.textContent = datasetLoaded && target ? `${t("target")}: ${target}` : t("stepTrainSub");
  const main = lastResult ? mainMetric(lastResult) : null;
  sub3.textContent = main ? `${main.label} ${fmt(main.value)}` : t("stepResultsSub");
}

// ── Step 1 · Data ────────────────────────────────────────────────────────────
function initDropzone() {
  const dz = $("csv-dropzone");
  const input = $("upload-csv");
  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  ["dragenter", "dragover"].forEach(evt => dz.addEventListener(evt, (e) => { stop(e); dz.classList.add("drag-over"); }));
  ["dragleave", "drop"].forEach(evt => dz.addEventListener(evt, (e) => { stop(e); dz.classList.remove("drag-over"); }));
  dz.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
  });
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { if (input.files[0]) loadDataFile(input.files[0]); });
}

async function parseTableFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "csv") {
    return new Promise((resolve, reject) => {
      Papa.parse(file, { skipEmptyLines: true, complete: (r) => resolve(r.data), error: reject });
    });
  }
  if (ext === "arff") {
    const rows = parseArff(await file.text());
    if (!rows) throw new Error("arff");
    return rows;
  }
  if (["xls", "xlsx", "xlsm"].includes(ext)) {
    await loadScript(CDN.xlsx);
    const workbook = XLSX.read(new Uint8Array(await file.arrayBuffer()), { type: "array" });
    const sheet = workbook.Sheets[workbook.SheetNames[0]];
    return XLSX.utils.sheet_to_json(sheet, { header: 1, defval: "" });
  }
  throw new Error("format");
}

function parseArff(content) {
  const columns = [];
  const dataLines = [];
  let inData = false;
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("%")) continue;
    if (inData) dataLines.push(trimmed);
    else if (trimmed.toLowerCase().startsWith("@attribute")) {
      const match = trimmed.match(/@attribute\s+['"]?([^'"@\s]+)['"]?/i);
      if (match) columns.push(match[1]);
    } else if (trimmed.toLowerCase().startsWith("@data")) inData = true;
  }
  if (!columns.length || !dataLines.length) return null;
  return [columns, ...Papa.parse(dataLines.join("\n"), { skipEmptyLines: true }).data];
}

async function loadDataFile(file) {
  const ok = [".csv", ".xls", ".xlsx", ".xlsm", ".arff"].some(ext => file.name.toLowerCase().endsWith(ext));
  if (!ok) {
    showAlert(t("pleaseSelectFile"), "warning");
    $("upload-csv").value = "";
    return;
  }
  $("preview-loading").hidden = false;
  try {
    const rows = await parseTableFile(file);
    if (!rows || rows.length < 2) throw new Error("empty");
    datasetFile = file;
    lastDatasetName = file.name;
    onDataLoaded(rows);
  } catch (err) {
    console.error(err);
    showAlert(t("pleaseSelectFile"), "warning");
    $("upload-csv").value = "";
  } finally {
    $("preview-loading").hidden = true;
  }
}

function onDataLoaded(rows) {
  const width = rows[0].length;
  datasetRows = rows.map(r => (r.length === width ? r : Array.from({ length: width }, (_, i) => r[i] ?? "")));
  totalDataRows = datasetRows.length - 1;
  datasetLoaded = true;
  invalidateResults();
  unlockWizardStep(STEP.TRAIN);

  buildStats();
  detectTimeColumns();
  $("csv-file-name").textContent = lastDatasetName;
  $("csv-dropzone").hidden = true;
  $("how-it-works").hidden = true;
  $("dataset-loaded").hidden = false;

  renderDatasetChips();
  renderVariablesTable();
  renderPreviewRows();
  refreshTargetOptions(true);
  buildImputationControls();
  const imputeToggle = $("enable-imputation");
  if (imputeToggle) imputeToggle.checked = false;
  $("imputation-controls").hidden = true;
  setDataView("vars");
  updateStepper();
}

function buildStats() {
  const header = datasetRows[0];
  const body = datasetRows.slice(1);
  dataSetStats = header.map((name, idx) => {
    const values = body.map(r => r[idx]).filter(v => !isMissing(v));
    const sample = values.slice(0, 200);
    const isDate = sample.length > 0 && sample.filter(isProbablyDate).length / sample.length >= 0.7;
    const isNumeric = !isDate && values.length > 0 && values.every(v => !isNaN(parseFloat(v)) && isFinite(v));
    const missing = body.length - values.length;
    const counts = new Map();
    values.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
    const stat = { name, missing, unique: counts.size, type: isDate ? "Date" : (isNumeric ? "Numeric" : "Categorical") };
    if (isNumeric) {
      const nums = values.map(Number);
      let min = Infinity, max = -Infinity, sum = 0;
      nums.forEach(n => { if (n < min) min = n; if (n > max) max = n; sum += n; });
      const mean = sum / nums.length;
      const std = Math.sqrt(nums.reduce((a, n) => a + (n - mean) ** 2, 0) / nums.length);
      Object.assign(stat, { min, max, mean: +mean.toFixed(3), std: +std.toFixed(3), hist: histogram(nums, min, max, 18) });
    } else {
      stat.top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
    }
    return stat;
  });
}

function histogram(nums, min, max, bins) {
  const counts = new Array(bins).fill(0);
  const span = max - min || 1;
  nums.forEach(n => { counts[Math.min(bins - 1, Math.floor(((n - min) / span) * bins))] += 1; });
  return counts;
}

function sparkline(hist) {
  const max = Math.max(...hist) || 1;
  const w = 90, h = 26, bw = w / hist.length;
  const bars = hist.map((c, i) => {
    const bh = Math.max(1, (c / max) * h);
    return `<rect x="${(i * bw).toFixed(1)}" y="${(h - bh).toFixed(1)}" width="${(bw - 1).toFixed(1)}" height="${bh.toFixed(1)}"/>`;
  }).join("");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true">${bars}</svg>`;
}

function detectTimeColumns() {
  detectedTimeColumns = dataSetStats.filter(s => s.type === "Date").map(s => s.name);
  const select = $("time-column");
  select.innerHTML = detectedTimeColumns.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  $("time-column-container").hidden = detectedTimeColumns.length === 0;
  $("pred-len-container").hidden = detectedTimeColumns.length === 0;
  updatePredLenDisplay();
}

function renderDatasetChips() {
  const totalCells = totalDataRows * datasetRows[0].length;
  const missing = dataSetStats.reduce((a, s) => a + s.missing, 0);
  $("dataset-chips").innerHTML = [
    `<span class="pill">${fmt(totalDataRows, 0)} ${t("rowsLower")}</span>`,
    `<span class="pill">${datasetRows[0].length} ${t("columnsLower")}</span>`,
    `<span class="pill ${missing ? "pill-warn" : "pill-ok"}">${pct(missing / (totalCells || 1))} ${t("missingLower")}</span>`,
  ].join("");
}

function getDroppedColumns() {
  return Array.from(document.querySelectorAll("#vars-table .var-include"))
    .filter(cb => !cb.checked).map(cb => cb.value);
}

function renderVariablesTable() {
  const dropped = new Set(getDroppedColumns());
  const target = $("target-column").value;
  const typeLabel = { Numeric: t("typeNumeric"), Categorical: t("typeCategorical"), Date: t("typeDate") };
  const rows = dataSetStats.map((s) => {
    const missRate = s.missing / (totalDataRows || 1);
    const missClass = missRate > 0.4 ? "bad" : missRate > 0.15 ? "warn" : "ok";
    let summary;
    if (s.type === "Numeric") {
      summary = `<div class="var-dist">${sparkline(s.hist)}<span class="muted">${fmt(s.min)} – ${fmt(s.max)}</span></div>`;
    } else {
      summary = `<div class="var-top">${(s.top || []).map(([v, c]) =>
        `<span class="pill pill-soft" title="${c}">${esc(String(v).slice(0, 18))} · ${pct(c / (totalDataRows - s.missing || 1), 0)}</span>`).join("")}</div>`;
    }
    const isTarget = s.name === target;
    return `<tr class="${dropped.has(s.name) ? "is-dropped" : ""}${isTarget ? " is-target" : ""}">
      <td class="col-use"><input type="checkbox" class="var-include" value="${esc(s.name)}" ${dropped.has(s.name) ? "" : "checked"} ${isTarget ? "disabled" : ""} aria-label="${esc(t("useVariable"))} ${esc(s.name)}"></td>
      <td class="col-name"><span class="var-name">${esc(s.name)}</span>${isTarget ? `<span class="pill pill-primary">${t("target")}</span>` : ""}</td>
      <td><span class="type-badge type-${s.type.toLowerCase()}">${typeLabel[s.type]}</span></td>
      <td class="col-missing"><div class="meter meter-${missClass}"><span style="width:${Math.min(100, missRate * 100).toFixed(1)}%"></span></div><span class="muted">${pct(missRate, 0)}</span></td>
      <td class="num">${fmt(s.unique, 0)}</td>
      <td class="col-dist">${summary}</td>
    </tr>`;
  }).join("");
  $("vars-table").innerHTML = `<thead><tr>
      <th class="col-use">${t("use")}</th><th>${t("variable")}</th><th>${t("type")}</th>
      <th>${t("missingShort")}</th><th class="num">${t("uniqueShort")}</th><th>${t("distribution")}</th>
    </tr></thead><tbody>${rows}</tbody>`;
  $("vars-table").querySelectorAll(".var-include").forEach(cb => cb.addEventListener("change", onVariableToggle));
}

function onVariableToggle(e) {
  e.target.closest("tr").classList.toggle("is-dropped", !e.target.checked);
  buildImputationControls();
  updateTrainSummary();
}

const PREVIEW_ROWS = 10;

function renderPreviewRows() {
  const header = datasetRows[0];
  $("rows-caption").textContent = t("rowsCaption", { n: Math.min(PREVIEW_ROWS, totalDataRows), total: fmt(totalDataRows, 0) });
  const head = `<thead><tr>${header.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>`;
  const body = datasetRows.slice(1, PREVIEW_ROWS + 1).map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("");
  $("preview-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

function setDataView(view) {
  $("vars-view").hidden = view !== "vars";
  $("rows-panel").hidden = view !== "rows";
  document.querySelectorAll(".seg-btn").forEach(b => b.classList.toggle("active", b.dataset.view === view));
}

function removeCSVFile() {
  $("upload-csv").value = "";
  datasetRows = null;
  datasetFile = null;
  datasetLoaded = false;
  dataSetStats = null;
  lastDatasetName = null;
  invalidateResults();
  $("dataset-loaded").hidden = true;
  $("csv-dropzone").hidden = false;
  $("how-it-works").hidden = false;
  $("vars-table").innerHTML = "";
  $("preview-table").innerHTML = "";
  $("target-column").innerHTML = "";
  $("time-column-container").hidden = true;
  $("pred-len-container").hidden = true;
  resetAdvancedOptions();
  wizardMaxStep = 1;
  setWizardStep(STEP.DATA);
}

function invalidateResults() {
  lastResult = null;
  shapPlot = null;
  currentShapRequestId = null;
  $("training-results").innerHTML = "";
  if (wizardMaxStep > STEP.TRAIN) wizardMaxStep = STEP.TRAIN;
}

// ── Step 2 · Train ───────────────────────────────────────────────────────────
function guessTarget(header) {
  const hints = /(target|label|outcome|class|y|result|diagnos|event|death|mortal|readmi|sepsis)/i;
  const named = [...header].reverse().find(h => hints.test(h));
  return named || header[header.length - 1];
}

function refreshTargetOptions(initial = false) {
  const select = $("target-column");
  const header = datasetRows[0];
  const current = select.value;
  select.innerHTML = header.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  select.value = !initial && current && header.includes(current) ? current : guessTarget(header);
  onTargetChange();
}

function onTargetChange() {
  // The target can never be excluded
  const target = $("target-column").value;
  document.querySelectorAll("#vars-table .var-include").forEach((cb) => {
    if (cb.value === target) cb.checked = true;
  });
  renderVariablesTable();
  updateAdvancedOptions();
  renderTargetInsight();
  buildImputationControls();
  updateTrainSummary();
}

function taskTypeFor(target) {
  if (detectedTimeColumns.length > 0) return "timeseries";
  const stat = (dataSetStats || []).find(s => s.name === target);
  if (!stat) return "classification";
  return stat.type === "Numeric" && stat.unique > 20 ? "regression" : "classification";
}

function renderTargetInsight() {
  const box = $("target-insight");
  const target = $("target-column").value;
  const stat = (dataSetStats || []).find(s => s.name === target);
  if (!stat) { box.innerHTML = ""; return; }
  const task = taskTypeFor(target);
  const taskLabel = { classification: t("taskClassification"), regression: t("taskRegression"), timeseries: t("taskTimeseries") }[task];
  let detail = "";
  if (task === "classification") {
    const idx = datasetRows[0].indexOf(target);
    const counts = new Map();
    datasetRows.slice(1).forEach(r => { if (!isMissing(r[idx])) counts.set(r[idx], (counts.get(r[idx]) || 0) + 1); });
    const total = [...counts.values()].reduce((a, b) => a + b, 0) || 1;
    detail = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([v, c]) => `
      <div class="class-row"><span class="class-name">${esc(v)}</span>
        <div class="meter meter-primary"><span style="width:${(c / total * 100).toFixed(1)}%"></span></div>
        <span class="muted num">${pct(c / total)}</span></div>`).join("");
  } else if (stat.hist) {
    detail = `<div class="var-dist">${sparkline(stat.hist)}<span class="muted">${fmt(stat.min)} – ${fmt(stat.max)}</span></div>`;
  }
  box.innerHTML = `<div class="insight-head"><span class="type-badge type-task">${taskLabel}</span></div>${detail}`;
}

function updateAdvancedOptions() {
  const task = taskTypeFor($("target-column").value);
  const metricSelect = $("adv-eval-metric");
  const previousMetric = metricSelect.value;
  metricSelect.innerHTML = ADVANCED_METRICS[task].map(m => `<option value="${m.value}">${m.label}</option>`).join("");
  if (ADVANCED_METRICS[task].some(m => m.value === previousMetric)) metricSelect.value = previousMetric;

  const models = task === "timeseries" ? ADVANCED_MODELS.timeseries : ADVANCED_MODELS.tabular;
  const container = $("adv-models-container");
  const unchecked = new Set(Array.from(container.querySelectorAll("input:not(:checked)")).map(cb => cb.value));
  container.innerHTML = models.map(m => `<label class="chip-check"><input type="checkbox" value="${m.value}" ${unchecked.has(m.value) ? "" : "checked"}><span>${m.label}</span></label>`).join("");

  $("adv-cv-container").hidden = task === "timeseries";
  if (task === "timeseries") $("advanced-options-section").open = true;
}

function resetAdvancedOptions() {
  $("advanced-options-section").open = false;
  $("adv-eval-metric").innerHTML = "";
  $("adv-models-container").innerHTML = "";
  $("adv-use-cv").checked = true;
  toggleValSplit();
}

function toggleValSplit() {
  $("adv-val-split-container").hidden = $("adv-use-cv").checked;
}

function initTimePresets() {
  $("time-presets").addEventListener("click", (e) => {
    const btn = e.target.closest(".chip-btn");
    if (!btn) return;
    $("training-time-limit").value = btn.dataset.min;
    $("training-time-limit").dispatchEvent(new Event("change", { bubbles: true }));
    updateTimeDisplay();
  });
}

function formatMinutes(minutes) {
  const h = Math.floor(minutes / 60), m = minutes % 60;
  if (!h) return `${m} min`;
  return m ? `${h} h ${m} min` : `${h} h`;
}

function updateTimeDisplay() {
  const minutes = parseInt($("training-time-limit").value, 10) || 1;
  $("time-limit-display").textContent = formatMinutes(minutes);
  document.querySelectorAll("#time-presets .chip-btn").forEach(b => b.classList.toggle("active", parseInt(b.dataset.min, 10) === minutes));
  updateTrainSummary();
}

function updatePredLenDisplay() {
  $("pred-len-display").textContent = `${$("prediction-length-percent").value}%`;
}

function buildImputationControls() {
  const container = $("imputation-controls");
  const toggleRow = $("container-enable-imputation");
  if (!datasetRows) { container.innerHTML = ""; return; }
  const dropped = new Set(getDroppedColumns());
  const target = $("target-column").value;
  const previous = {};
  container.querySelectorAll("select").forEach(s => { previous[s.dataset.col] = s.value; });

  const rows = dataSetStats.filter(s => s.missing > 0 && !dropped.has(s.name) && s.name !== target).map((s) => {
    const methods = s.type === "Numeric" ? ["median", "mean", "mode", "constant"] : ["mode", "constant"];
    const opts = methods.map(m => `<option value="${m}" ${previous[s.name] === m ? "selected" : ""}>${t(m)}</option>`).join("");
    return `<div class="impute-row"><span class="var-name">${esc(s.name)}</span>
      <span class="muted">${pct(s.missing / totalDataRows, 0)}</span>
      <select data-col="${esc(s.name)}">${opts}</select>
      <input type="text" class="impute-constant" placeholder="${esc(t("constantValue"))}" hidden></div>`;
  });
  if (!rows.length) {
    container.innerHTML = "";
    toggleRow.innerHTML = `<p class="muted">${t("noMissingValues")}</p>`;
    return;
  }
  if (!$("enable-imputation")) {
    toggleRow.innerHTML = `<label class="check"><input type="checkbox" id="enable-imputation"><span data-i18n="enableImputation">${t("enableImputation")}</span></label>`;
    $("enable-imputation").addEventListener("change", (e) => { $("imputation-controls").hidden = !e.target.checked; });
  }
  container.innerHTML = rows.join("");
  container.querySelectorAll("select").forEach((select) => {
    const input = select.nextElementSibling;
    const sync = () => { input.hidden = select.value !== "constant"; };
    select.addEventListener("change", sync);
    sync();
  });
}

function updateTrainSummary() {
  const list = $("train-summary");
  if (!list || !datasetLoaded) return;
  const target = $("target-column").value;
  const dropped = getDroppedColumns();
  const used = datasetRows[0].length - 1 - dropped.filter(c => c !== target).length;
  const task = taskTypeFor(target);
  const metric = $("adv-eval-metric").selectedOptions[0]?.textContent || "—";
  const items = [
    [t("data"), `${lastDatasetName}`],
    [t("target"), target || "—"],
    [t("task"), { classification: t("taskClassification"), regression: t("taskRegression"), timeseries: t("taskTimeseries") }[task]],
    [t("variablesUsed"), `${used}${dropped.length ? ` <span class="muted">(${dropped.length} ${t("excluded")})</span>` : ""}`],
    [t("metric"), metric],
    [t("time"), $("time-limit-display").textContent],
  ];
  list.innerHTML = items.map(([k, v]) => `<dt>${k}</dt><dd>${typeof v === "string" && v.includes("<span") ? v : esc(v)}</dd>`).join("");
  updateStepSubtitles();
}

function prepareTrainingTable(dropped) {
  const header = datasetRows[0];
  const keep = header.map((_, i) => i).filter(i => !dropped.includes(header[i]));
  const table = datasetRows.map(r => keep.map(i => r[i]));
  const imputationChoices = {};
  if ($("enable-imputation")?.checked) {
    $("imputation-controls").querySelectorAll("select").forEach((select) => {
      imputationChoices[select.dataset.col] = {
        method: select.value,
        constant: select.value === "constant" ? select.nextElementSibling.value : null,
      };
    });
    const newHeader = table[0];
    Object.entries(imputationChoices).forEach(([col, { method, constant }]) => {
      const idx = newHeader.indexOf(col);
      if (idx < 0) return;
      const present = table.slice(1).map(r => r[idx]).filter(v => !isMissing(v));
      let fill;
      if (method === "mean" || method === "median") {
        const nums = present.map(Number).filter(n => !isNaN(n)).sort((a, b) => a - b);
        if (!nums.length) return;
        if (method === "mean") fill = (nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(4);
        else {
          const mid = Math.floor(nums.length / 2);
          fill = String(nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2);
        }
      } else if (method === "mode") {
        const freq = {};
        present.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
        fill = Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0];
      } else {
        fill = constant;
      }
      if (fill === undefined) return;
      for (let r = 1; r < table.length; r++) if (isMissing(table[r][idx])) table[r][idx] = fill;
    });
  }
  return { table, imputationChoices };
}

function startTraining() {
  const target = $("target-column").value;
  if (!datasetRows || !target) {
    showAlert(t("selectCSVAndTarget"), "warning");
    return;
  }
  const dropped = getDroppedColumns();
  if (dropped.includes(target)) {
    showAlert(t("cannotDropTargetColumn"), "error");
    return;
  }
  const minutes = parseInt($("training-time-limit").value, 10) || 1;
  const task = taskTypeFor(target);
  const selectedModels = Array.from($("adv-models-container").querySelectorAll("input:checked"));
  if (!selectedModels.length) {
    showAlert(t("selectAtLeastOneModel"), "warning");
    return;
  }

  const { table, imputationChoices } = prepareTrainingTable(dropped);
  dataPreprocessing = { data_preprocessing: { columns_drop: dropped, imputation_missing_value: imputationChoices } };
  const blob = new Blob([Papa.unparse(table, { quotes: false })], { type: "text/csv" });
  lastCleanedCsvBlob = blob;
  lastUsedTargetColumn = target;

  const formData = new FormData();
  formData.append("file", blob, `${datasetBaseName()}.csv`);
  formData.append("target_column", target);
  formData.append("time_limit", String(Math.max(60, minutes * 60)));
  if (task === "timeseries") {
    formData.append("time_column", $("time-column").value);
    formData.append("prediction_length_percent", $("prediction-length-percent").value);
  }
  const metric = $("adv-eval-metric").value;
  if (metric) formData.append("eval_metric", metric);
  const excluded = Array.from($("adv-models-container").querySelectorAll("input:not(:checked)")).map(cb => cb.value);
  if (excluded.length) formData.append("excluded_model_types", excluded.join(","));
  const useCv = $("adv-use-cv").checked;
  formData.append("use_cv", useCv ? "1" : "0");
  if (!useCv) formData.append("val_size", (parseInt($("adv-val-size").value, 10) / 100).toFixed(2));

  invalidateResults();
  $("start-training-btn").hidden = true;
  $("stop-training-btn").hidden = false;
  $("stop-training-btn").disabled = false;
  $("stop-training-btn").textContent = t("stopTraining");
  startProgress(Math.max(60, minutes * 60));
  $("training-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });

  fetch("/train", { method: "POST", body: formData })
    .then(r => r.json())
    .then((data) => {
      if (data.error) {
        showAlert(t(data.error) !== data.error ? t(data.error) : t("beginTrainingError"), "error");
        resetTrainingButtons();
        return;
      }
      trainingId = data.training_id;
      pollTrainingResult(trainingId);
    })
    .catch(() => {
      resetTrainingButtons();
      showAlert(t("trainingError"), "error");
    });
}

function stopTraining() {
  if (!trainingId) { resetTrainingButtons(); return; }
  // AutoGluon cannot interrupt the model it is fitting: the run ends after it
  $("stop-training-btn").disabled = true;
  $("stop-training-btn").textContent = t("stoppingTraining");
  fetch(`/stop_training/${trainingId}`, { method: "POST" })
    .then(r => r.json())
    .then((data) => { if (data.error) resetTrainingButtons(); })
    .catch(() => { showAlert(t("stopTrainingError"), "error"); });
}

function resetTrainingButtons() {
  $("start-training-btn").hidden = false;
  $("stop-training-btn").hidden = true;
  stopProgress();
}

function pollTrainingResult(id) {
  clearInterval(pollingIntervalTraining);
  pollingIntervalTraining = setInterval(() => {
    fetch(`/training_progress/${id}`).then(r => (r.ok ? r.json() : null)).then(p => { if (p) renderProgress(p); }).catch(() => {});
    fetch(`/training_result/${id}`)
      .then(res => (res.status === 202 ? null : res.json()))
      .then((result) => {
        if (!result) return;
        clearInterval(pollingIntervalTraining);
        pollingIntervalTraining = null;
        resetTrainingButtons();
        if (result.error === "training_interrupted") showAlert(t("trainingStopped"), "warning", 5000);
        else if (result.error) showAlert(t(result.error) !== result.error ? t(result.error) : t("trainingErrorGet"), "error");
        else renderTrainingResults(result);
      })
      .catch(() => {
        clearInterval(pollingIntervalTraining);
        showAlert(t("trainingErrorNetwork"), "error");
        resetTrainingButtons();
      });
  }, 1500);
}

// ── Training progress (reported by AutoGluon, nothing simulated) ──────────────
let progressTimer = null;
let progressStart = 0;
let progressLimit = 0;
let lastProgress = null;

const MODEL_LABELS = [
  [/^LightGBMXT/, "LightGBM (extra trees)"], [/^LightGBMLarge/, "LightGBM (large)"], [/^LightGBM/, "LightGBM"],
  [/^CatBoost/, "CatBoost"], [/^XGBoost/, "XGBoost"], [/^RandomForest/, "Random Forest"], [/^ExtraTrees/, "Extra Trees"],
  [/^NeuralNetTorch/, "nnTorch"], [/^NeuralNetFastAI/, "nnFastAI"], [/^KNeighbors/, "KNN"], [/^LinearModel/, "linear"],
  [/^WeightedEnsemble/, "ensemble"],
];

function modelLabel(name) {
  const match = MODEL_LABELS.find(([re]) => re.test(name));
  if (!match) return name;
  const label = match[1];
  return { nnTorch: t("modelNeuralNet"), nnFastAI: `${t("modelNeuralNet")} (FastAI)`, linear: t("modelLinear"), ensemble: t("modelEnsemble") }[label] || label;
}

function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function startProgress(limitSeconds) {
  stopProgress();
  progressStart = Date.now();
  progressLimit = limitSeconds;
  lastProgress = null;
  $("tp-models-list").innerHTML = "";
  $("tp-gpu").hidden = true;
  $("tp-limit").textContent = t("ofMaximum", { limit: clock(limitSeconds) });
  renderProgress({ phase: "preparing", elapsed: 0, models: [] });
  $("training-panel").classList.add("visible");
  progressTimer = setInterval(() => renderProgressClock(), 1000);
}

function stopProgress() {
  clearInterval(progressTimer);
  progressTimer = null;
  $("training-panel").classList.remove("visible");
}

function renderProgressClock() {
  const elapsed = (Date.now() - progressStart) / 1000;
  $("tp-elapsed").textContent = clock(elapsed);
  const evaluating = lastProgress && lastProgress.phase === "evaluating";
  const bar = $("tp-progress-fill");
  bar.classList.toggle("indeterminate", Boolean(evaluating));
  bar.style.width = evaluating ? "" : `${Math.min(100, (elapsed / progressLimit) * 100).toFixed(1)}%`;
}

function renderProgress(p) {
  lastProgress = p;
  if (typeof p.elapsed === "number" && p.elapsed > 0) progressStart = Date.now() - p.elapsed * 1000;
  const done = p.models.filter(m => m.status === "done" && !m.ensemble).length;
  const phaseText = p.stopping ? t("phaseStopping")
    : p.phase === "fitting" ? t("phaseFitting", { n: done })
    : p.phase === "evaluating" ? t("phaseEvaluating")
    : p.phase === "done" ? t("phaseDone")
    : t("phasePreparing");
  $("tp-phase").textContent = phaseText;
  $("tp-note").textContent = p.phase === "evaluating" ? t("progressNoteEvaluating") : t("progressNote");
  $("tp-gpu").hidden = !p.gpu;
  const metricLabel = METRIC_INFO[p.metric]?.label || (p.metric || "").toUpperCase();
  $("tp-models-list").innerHTML = p.models.slice().reverse().map((m) => {
    const icon = m.status === "running" ? '<span class="tp-model-icon spin">↻</span>'
      : m.status === "done" ? '<span class="tp-model-icon">✓</span>'
      : m.status === "failed" ? '<span class="tp-model-icon">✕</span>' : '<span class="tp-model-icon">–</span>';
    const detail = m.status === "running" ? t("modelTraining")
      : m.status === "failed" ? t("modelFailed")
      : m.status === "unavailable" ? t("modelUnavailable")
      : m.status === "skipped" ? t("modelSkipped")
      : `${m.score != null ? `${fmt(m.score)} ${metricLabel}` : ""}${m.fit_time != null ? ` · ${fmt(m.fit_time, 1)} s` : ""}`;
    return `<div class="tp-model-row ${m.status}" title="${esc(m.name)}">${icon}<span class="tp-model-name">${esc(modelLabel(m.name))}</span><span class="tp-model-badge">${detail}</span></div>`;
  }).join("");
  renderProgressClock();
}

// ── Step 3 · Results ─────────────────────────────────────────────────────────
const METRIC_INFO = {
  roc_auc: { label: "ROC AUC", higher: true, bands: [[0.9, "excellent"], [0.8, "good"], [0.7, "fair"], [0, "weak"]] },
  accuracy: { label: "Accuracy", higher: true },
  f1_macro: { label: "F1 macro", higher: true, bands: [[0.85, "excellent"], [0.7, "good"], [0.55, "fair"], [0, "weak"]] },
  r2: { label: "R²", higher: true, bands: [[0.8, "excellent"], [0.6, "good"], [0.3, "fair"], [-Infinity, "weak"]] },
  rmse: { label: "RMSE", higher: false },
  mae: { label: "MAE", higher: false },
  mape: { label: "MAPE", higher: false },
};
const METRIC_ORDER = ["roc_auc", "f1_macro", "accuracy", "r2", "rmse", "mae", "mape"];

function metricEntries(result) {
  return Object.entries(result.metrics || {})
    .filter(([, m]) => m && typeof m.value === "number")
    .sort((a, b) => METRIC_ORDER.indexOf(a[0]) - METRIC_ORDER.indexOf(b[0]));
}

function mainMetric(result) {
  const first = metricEntries(result)[0];
  return first ? { key: first[0], label: METRIC_INFO[first[0]]?.label || first[0].toUpperCase(), value: first[1].value } : null;
}

function qualityBand(key, value) {
  const bands = METRIC_INFO[key]?.bands;
  if (!bands) return null;
  return bands.find(([min]) => value >= min)?.[1] || null;
}

function majorityShare(target) {
  if (!datasetRows || !target) return null;
  const idx = datasetRows[0].indexOf(target);
  if (idx < 0) return null;
  const counts = new Map();
  datasetRows.slice(1).forEach(r => { if (!isMissing(r[idx])) counts.set(r[idx], (counts.get(r[idx]) || 0) + 1); });
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  return total ? Math.max(...counts.values()) / total : null;
}

function renderTrainingResults(data, { silent = false } = {}) {
  lastResult = data;
  lastTrainedModelPath = data.model_path;
  dataSetPreview = data.markdown_preview || dataSetPreview;
  summaryResults = {
    summary: data.summary_LLM,
    feature_importance_plot: data.feature_importance_plot,
    metrics_plot: data.metrics,
    forecast_plot: data.forecast_plot || null,
    task_type: data.task_type,
    best_model: data.best_model,
    prediction_length: data.prediction_length || null,
    train_time: data.train_time,
  };
  if (!silent) {
    unlockWizardStep(STEP.RESULTS);
    setWizardStep(STEP.RESULTS);
    $("use-last-model-btn").hidden = !data.download_url;
  }

  const taskLabel = { binary: t("taskBinary"), multiclass: t("taskMulticlass"), regression: t("taskRegression"), timeseries: t("taskTimeseries") }[data.task_type] || data.task_type;
  const baseline = ["binary", "multiclass"].includes(data.task_type) ? majorityShare(lastUsedTargetColumn) : null;

  const kpis = metricEntries(data).map(([key, m]) => {
    const info = METRIC_INFO[key] || { label: key.toUpperCase() };
    const band = qualityBand(key, m.value);
    let note = band ? `<span class="badge badge-${band}">${t(`quality_${band}`)}</span>` : "";
    if (key === "accuracy" && baseline != null) {
      note = `<span class="kpi-note">${t("baselineAccuracy")}: ${pct(baseline)}</span>`;
    }
    const value = ["accuracy"].includes(key) ? pct(m.value) : fmt(m.value);
    return `<div class="kpi card"><span class="kpi-label">${info.label}</span><span class="kpi-value">${value}</span>${note}<span class="kpi-help">${t(`help_${key}`) !== `help_${key}` ? t(`help_${key}`) : ""}</span></div>`;
  }).join("");

  const plots = metricEntries(data).flatMap(([key, m]) => [m.plot, m.plot_hist].filter(Boolean).map((p, i) =>
    `<figure class="plot card"><img src="data:image/png;base64,${p}" alt="${esc(key)}${i ? "_errors" : ""}" loading="lazy"><figcaption>${METRIC_INFO[key]?.label || key}</figcaption></figure>`));
  if (data.forecast_plot) plots.push(`<figure class="plot card"><img src="data:image/png;base64,${data.forecast_plot}" alt="forecast" loading="lazy"><figcaption>${t("forecast")}</figcaption></figure>`);

  const tabs = [
    { id: "perf", label: t("tabPerformance"), html: plots.length ? `<div class="plot-grid">${plots.join("")}</div>` : `<p class="muted">${t("noPlots")}</p>` },
  ];
  if (data.feature_importance && Object.keys(data.feature_importance).length) {
    tabs.push({ id: "importance", label: t("tabImportance"), html: renderImportance(data.feature_importance) });
  }
  if (data.task_type !== "timeseries") {
    tabs.push({ id: "shap", label: t("tabShap"), html: `
      <div class="card">
        <p class="muted">${t("shapIntro")}</p>
        <div class="row-gap">
          <button id="generate-shap-button" class="btn btn-secondary" type="button" onclick="generateShapPlot()">${t("generateShap")}</button>
          <div id="training-spinner-shap" class="spinner" hidden></div>
        </div>
        <div id="shap-plots">${shapPlot ? `<img src="data:image/png;base64,${shapPlot}" alt="shap_summary">` : ""}</div>
      </div>` });
  }
  if (Array.isArray(data.leaderboard) && data.leaderboard.length) {
    tabs.push({ id: "models", label: t("tabModels"), html: renderLeaderboard(data.leaderboard, data.best_model) });
  }

  $("training-results").innerHTML = `
    <div class="results-head card">
      <div>
        <span class="eyebrow">${t("bestModel")}</span>
        <h2 class="mono">${esc(data.best_model)}</h2>
        <p class="muted">${taskLabel} · ${t("target")}: <strong>${esc(lastUsedTargetColumn || "")}</strong> · ${fmt(data.train_time, 0)} s ${data.used_gpu ? '<span class="pill pill-ok">GPU</span>' : ""}</p>
      </div>
      <div class="results-actions">
        <a class="btn btn-primary" href="${data.download_url || "#"}" download>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v12M7 11l5 5 5-5M4 20h16"/></svg>${t("downloadModel")}</a>
        <button class="btn btn-secondary" type="button" onclick="downloadPDF()">${t("downloadPDF")}</button>
        <button class="btn btn-ghost" type="button" onclick="downloadAllPlots()">${t("downloadPlots")}</button>
        <button class="btn btn-ghost" type="button" onclick="useLastTrainedModel(true)">${t("predictWithModel")}</button>
      </div>
    </div>
    <div class="kpi-grid">${kpis}</div>
    <div class="tabs" role="tablist">${tabs.map((tab, i) => `<button class="tab ${i === 0 ? "active" : ""}" role="tab" data-tab="${tab.id}" onclick="setResultsTab('${tab.id}')">${tab.label}</button>`).join("")}</div>
    ${tabs.map((tab, i) => `<div class="tab-panel" data-tab="${tab.id}" ${i === 0 ? "" : "hidden"}>${tab.html}</div>`).join("")}
    <div class="ask-card card">
      <div><strong>${t("askTitle")}</strong><p class="muted">${t("askText")}</p></div>
      <button class="btn btn-secondary" type="button" onclick="openChatWith(${jsArg(t("suggestGood"))})">${t("askButton")}</button>
    </div>`;
  updateStepSubtitles();
}

function setResultsTab(id) {
  document.querySelectorAll("#training-results .tab").forEach(b => b.classList.toggle("active", b.dataset.tab === id));
  document.querySelectorAll("#training-results .tab-panel").forEach(p => { p.hidden = p.dataset.tab !== id; });
}

function renderImportance(importance) {
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 15);
  const max = Math.max(...entries.map(([, v]) => Math.abs(v))) || 1;
  const rows = entries.map(([name, v]) => `
    <div class="imp-row"><span class="imp-name" title="${esc(name)}">${esc(name)}</span>
      <div class="imp-bar"><span class="${v < 0 ? "neg" : ""}" style="width:${(Math.abs(v) / max * 100).toFixed(1)}%"></span></div>
      <span class="imp-val num">${fmt(v, 4)}</span></div>`).join("");
  return `<div class="card"><p class="muted">${t("importanceIntro")}</p><div class="imp-list">${rows}</div></div>`;
}

function renderLeaderboard(rows, best) {
  const scores = rows.map(r => r.score_val).filter(v => typeof v === "number");
  const min = Math.min(...scores), max = Math.max(...scores), span = (max - min) || 1;
  const body = rows.map(r => `<tr class="${r.model === best ? "is-best" : ""}">
      <td>${esc(r.model)}${r.model === best ? ` <span class="badge badge-excellent">${t("best")}</span>` : ""}</td>
      <td class="num"><div class="score-cell"><div class="meter meter-primary"><span style="width:${typeof r.score_val === "number" ? (15 + 85 * (r.score_val - min) / span).toFixed(1) : 0}%"></span></div>${typeof r.score_val === "number" ? fmt(r.score_val, 4) : "—"}</div></td>
      <td class="num">${typeof r.fit_time === "number" ? `${fmt(r.fit_time, 1)} s` : "—"}</td>
      <td class="num">${typeof r.pred_time_val === "number" ? `${fmt(r.pred_time_val, 2)} s` : "—"}</td></tr>`).join("");
  return `<div class="card table-wrap"><table class="table">
    <thead><tr><th>${t("model")}</th><th class="num">${t("scoreVal")}</th><th class="num">${t("fitTime")}</th><th class="num">${t("predictTime")}</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

async function generateShapPlot() {
  if (!lastTrainedModelPath || !lastCleanedCsvBlob || !lastUsedTargetColumn) {
    showAlert(t("missingSHAPInfo"), "error");
    return;
  }
  const requestId = uid();
  currentShapRequestId = requestId;
  const button = $("generate-shap-button");
  const spinner = $("training-spinner-shap");
  button.disabled = true;
  spinner.hidden = false;
  const formData = new FormData();
  formData.append("training_id", trainingId || "");
  formData.append("model_path", lastTrainedModelPath || "");
  formData.append("target_column", lastUsedTargetColumn);
  formData.append("dataset", lastCleanedCsvBlob, "cleaned_data.csv");
  try {
    const response = await fetch("/generate_shap_plot", { method: "POST", body: formData });
    if (!response.ok) throw new Error(await response.text());
    const { shap_id, error } = await response.json();
    if (error) throw new Error(error);
    await new Promise((resolve, reject) => {
      const interval = setInterval(async () => {
        if (currentShapRequestId !== requestId) { clearInterval(interval); resolve(); return; }
        try {
          const data = await (await fetch(`/shap_progress/${shap_id}`)).json();
          if (!data.done) return;
          clearInterval(interval);
          if (data.error) { reject(new Error(data.error)); return; }
          shapPlot = data.result.shap_summary_plot;
          $("shap-plots").innerHTML = `<img src="data:image/png;base64,${shapPlot}" alt="shap_summary">`;
          button.hidden = true;
          resolve();
        } catch (e) { clearInterval(interval); reject(e); }
      }, 700);
    });
  } catch (err) {
    showAlert(`${t("shapGenerationError")}: ${err.message}`, "error");
    button.disabled = false;
  } finally {
    spinner.hidden = true;
  }
}

async function downloadAllPlots() {
  if (!lastResult) return;
  await loadScript(CDN.jszip);
  const zip = new JSZip();
  const add = (name, b64) => { if (b64) zip.file(`${name}.png`, b64, { base64: true }); };
  metricEntries(lastResult).forEach(([key, m]) => { add(key, m.plot); add(`${key}_errors`, m.plot_hist); });
  add("feature_importance", lastResult.feature_importance_plot);
  add("forecast", lastResult.forecast_plot);
  add("shap_summary", shapPlot);
  downloadBlob(await zip.generateAsync({ type: "blob" }), `${datasetBaseName()}_plots_${timestamp()}.zip`);
}

async function downloadPDF() {
  if (!lastResult || !dataSetStats) {
    showAlert(t("missingPDFInfo"), "error");
    return;
  }
  showAlert(t("pdfGenerating"), "info", 4000);
  try {
    const payload = {
      summary: summaryResults, preview: dataSetPreview, stats: dataSetStats,
      data_preprocessing: dataPreprocessing, target_column: lastUsedTargetColumn, dataset_name: lastDatasetName,
    };
    const groqKey = getGroqKey();
    if (groqKey) payload.groq_api_key = groqKey;
    if (shapPlot) payload.shap_summary_plot = shapPlot;
    const response = await fetch("/download_pdf", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!response.ok) { showAlert(t("pdfGenerationError"), "error"); return; }
    downloadBlob(await response.blob(), `${datasetBaseName()}_report_${timestamp()}.pdf`);
  } catch (_) {
    showAlert(t("trainingErrorNetwork"), "error");
  }
}

// ── Prediction ───────────────────────────────────────────────────────────────
function initPrediction() {
  $("predict-csv").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;
    if (![".csv", ".xls", ".xlsx", ".xlsm", ".arff"].some(ext => file.name.toLowerCase().endsWith(ext))) {
      showAlert(t("pleaseSelectFile"), "warning");
      this.value = "";
      return;
    }
    $("predict-file-name").textContent = file.name;
    $("predict-file-info").hidden = false;
    $("predict-csv-label").hidden = true;
    updatePredictButton();
  });
  $("predict-model-zip").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      showAlert(t("pleaseSelectZIP"), "warning");
      this.value = "";
      return;
    }
    setPredictModelChip(file.name);
  });
}

function setPredictModelChip(name) {
  $("predict-model-name").textContent = name;
  $("predict-model-info").hidden = false;
  $("predict-zip-label").hidden = true;
  $("use-last-model-btn").hidden = true;
  updatePredictButton();
}

function updatePredictButton() {
  $("predict-btn").disabled = !($("predict-csv").files[0] && $("predict-model-zip").files[0]);
}

function removePredictFile() {
  $("predict-csv").value = "";
  $("predict-file-info").hidden = true;
  $("predict-csv-label").hidden = false;
  $("prediction-results").hidden = true;
  updatePredictButton();
}

function removePredictModel() {
  $("predict-model-zip").value = "";
  $("predict-model-info").hidden = true;
  $("predict-zip-label").hidden = false;
  $("use-last-model-btn").hidden = !lastResult?.download_url;
  updatePredictButton();
}

async function useLastTrainedModel(switchMode = false) {
  if (!lastResult?.download_url) return;
  try {
    const blob = await (await fetch(lastResult.download_url)).blob();
    const name = lastResult.download_url.split("/").pop();
    const dt = new DataTransfer();
    dt.items.add(new File([blob], name, { type: "application/zip" }));
    $("predict-model-zip").files = dt.files;
    setPredictModelChip(name);
    if (switchMode) setMode("predict");
  } catch (_) {
    showAlert(t("trainingErrorNetwork"), "error");
  }
}

async function runPrediction() {
  const dataset = $("predict-csv").files[0];
  const model = $("predict-model-zip").files[0];
  if (!dataset || !model) { showAlert(t("selectDatasetAndModel"), "warning"); return; }
  $("predict-btn").disabled = true;
  $("predict-spinner").hidden = false;
  try {
    await loadScript(CDN.jszip);
    const zip = await JSZip.loadAsync(model);
    pngResultTrainingForPrediction = [];
    zip.forEach((path, entry) => {
      if (path.startsWith("plot_train_results/") && path.endsWith(".png")) pngResultTrainingForPrediction.push(entry);
    });
    const formData = new FormData();
    formData.append("dataset", dataset);
    formData.append("zip_model", model);
    const data = await (await fetch("/predict", { method: "POST", body: formData })).json();
    if (data.error) { showAlert(t(data.error), "error"); return; }
    if (!data.preview?.length) { showAlert(t("noPrediction"), "warning"); return; }
    PlotsPredictionResults = data.plots;
    renderPredictionResults(data);
  } catch (_) {
    showAlert(t("predictionError"), "error");
  } finally {
    $("predict-spinner").hidden = true;
    updatePredictButton();
  }
}

function renderPredictionResults(data) {
  const header = Object.keys(data.preview[0]);
  const table = `<table class="table table-compact"><thead><tr>${header.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${data.preview.map(r => `<tr>${header.map(h => `<td>${esc(r[h])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  const plots = Object.entries(data.plots || {}).map(([title, b64]) =>
    `<figure class="plot card"><img src="data:image/png;base64,${b64}" alt="${esc(title)}" loading="lazy"><figcaption>${esc(title.replace(/_/g, " "))}</figcaption></figure>`).join("");
  const box = $("prediction-results");
  box.innerHTML = `
    <div class="results-head card">
      <div><span class="eyebrow">${t("predictionResults")}</span><h2>${t("preview")}</h2></div>
      <div class="results-actions">
        <a class="btn btn-primary" href="${data.download_url || "#"}" download>${t("downloadPredictions")}</a>
        ${plots ? `<button class="btn btn-ghost" type="button" onclick="downloadAllPredictionPlots()">${t("downloadPredictionPlots")}</button>` : ""}
      </div>
    </div>
    <div class="card table-wrap">${table}</div>
    ${plots ? `<div class="plot-grid">${plots}</div>` : ""}`;
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function downloadAllPredictionPlots() {
  await loadScript(CDN.jszip);
  const zip = new JSZip();
  Object.entries(PlotsPredictionResults || {}).forEach(([name, b64]) => zip.file(`${name}.png`, b64, { base64: true }));
  downloadBlob(await zip.generateAsync({ type: "blob" }), `predictions_plots_${timestamp()}.zip`);
}

// ── Assistant ────────────────────────────────────────────────────────────────
function initChat() {
  const input = $("chat-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(140, input.scrollHeight)}px`;
  });
  $("groq-save-btn").addEventListener("click", () => {
    const val = $("groq-api-key-input").value.trim();
    if (!val) { showAlert(t("groqMissing"), "warning", 4000); return; }
    try { localStorage.setItem("groq_api_key", val); } catch (_) { /* private mode */ }
    hideGroqModal();
    if ($("chat-sidebar").classList.contains("collapsed")) toggleChatSidebar();
  });
  $("groq-cancel-btn").addEventListener("click", hideGroqModal);
  $("groq-api-key-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("groq-save-btn").click(); } });
}

function getGroqKey() {
  try { return localStorage.getItem("groq_api_key"); } catch (_) { return null; }
}

function showGroqModal() {
  $("settings-menu").classList.add("hidden");
  $("groq-modal").classList.remove("hidden");
  $("groq-api-key-input").value = "";
  $("groq-api-key-input").focus();
}

function hideGroqModal() {
  $("groq-modal").classList.add("hidden");
}

let serverHasAssistantKey = null;

async function assistantReady() {
  if (getGroqKey()) return true;
  if (serverHasAssistantKey === null) {
    try {
      serverHasAssistantKey = (await (await fetch("/assistant_status")).json()).server_key === true;
    } catch (_) { serverHasAssistantKey = false; }
  }
  return serverHasAssistantKey;
}

async function toggleChatSidebar() {
  const sidebar = $("chat-sidebar");
  const opening = sidebar.classList.contains("collapsed");
  // Ask for a key first when neither the browser nor the server has one
  if (opening && !(await assistantReady())) { showGroqModal(); return; }
  sidebar.classList.toggle("collapsed", !opening);
  document.body.classList.toggle("chat-open", opening);
  if (opening) { renderChatSuggestions(); setTimeout(() => $("chat-input").focus(), 200); }
}

async function openChatWith(text) {
  if ($("chat-sidebar").classList.contains("collapsed")) await toggleChatSidebar();
  if ($("chat-sidebar").classList.contains("collapsed")) return;
  $("chat-input").value = text;
  $("chat-input").focus();
}

function renderChatSuggestions() {
  const box = $("chat-suggestions");
  if (!box) return;
  const keys = appMode === 2 ? ["suggestPredict"]
    : lastResult ? ["suggestGood", "suggestImportant", "suggestImprove"]
    : datasetLoaded ? ["suggestData", "suggestTarget", "suggestMetric"]
    : ["suggestStart", "suggestFormat"];
  const hasMessages = $("chat-box").children.length > 0;
  box.hidden = hasMessages;
  box.innerHTML = keys.map(k => `<button type="button" class="chip-btn" onclick="sendChat(${jsArg(t(k))})">${esc(t(k))}</button>`).join("");
}

function addChatMessage(role, html) {
  const node = document.createElement("div");
  node.className = `chat-message ${role}`;
  node.innerHTML = html;
  $("chat-box").appendChild(node);
  $("chat-box").scrollTop = $("chat-box").scrollHeight;
  $("chat-suggestions").hidden = true;
  return node;
}

async function sendChat(preset) {
  const input = $("chat-input");
  const message = (typeof preset === "string" ? preset : input.value).trim();
  if (!message) return;
  const sendButton = document.querySelector("#chat-footer button");
  input.disabled = true;
  sendButton.disabled = true;
  addChatMessage("user", esc(message));
  input.value = "";
  input.style.height = "auto";
  const typing = addChatMessage("ai typing", "<span></span><span></span><span></span>");

  try {
    const payload = { message, lang: currentLang(), app_state: appState(), history: chatHistory.slice(-12) };
    if (typeof STUDY !== "undefined" && STUDY && STUDY.sessionId) payload.study_session_id = STUDY.sessionId;
    const groqKey = getGroqKey();
    if (groqKey) payload.groq_api_key = groqKey;
    if (appMode === 1) {
      if (lastResult) {
        payload.summary = {
          text: summaryResults.summary, feature_importance_plot: summaryResults.feature_importance_plot,
          metrics_plot: summaryResults.metrics_plot, forecast_plot: summaryResults.forecast_plot,
        };
        if (shapPlot) payload.shap_summary_plot = shapPlot;
        if (lastResult.feature_importance) payload.feature_importance = lastResult.feature_importance;
        payload.model_metadata = {
          task_type: summaryResults.task_type, best_model: summaryResults.best_model,
          target_column: lastUsedTargetColumn || null, prediction_length: summaryResults.prediction_length || null,
        };
      }
      if (dataSetStats) payload.stats = dataSetStats.map(({ hist, top, ...rest }) => rest);
      if (dataSetPreview) payload.markdown_preview = dataSetPreview;
      else if (datasetRows) payload.markdown_preview = datasetRows.slice(0, 6).map(r => r.join(" | ")).join("\n");
      if (dataPreprocessing) payload.data_preprocessing = dataPreprocessing;
    } else {
      if (PlotsPredictionResults) payload.plots_prediction_results = PlotsPredictionResults;
      if (pngResultTrainingForPrediction?.length) {
        const entries = await Promise.all(pngResultTrainingForPrediction.map(async e => [e.name.split("/").pop(), await e.async("base64")]));
        payload.png_result_training_for_prediction = Object.fromEntries(entries);
      }
    }
    const response = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    typing.remove();
    if (!response.ok) {
      if (data.error === "groqMissing") {
        serverHasAssistantKey = false;
        addChatMessage("ai error", esc(t("groqMissing")));
        showGroqModal();
        return;
      }
      addChatMessage("ai error", esc(t(data.error || "error_chat")) + (data.detail ? `<br><small>${esc(data.detail)}</small>` : ""));
      return;
    }
    const actions = Array.isArray(data.actions) ? data.actions : [];
    const text = data.response || (actions.length ? t("actionsIntro") : "");
    if (text) addChatMessage("ai", renderMarkdown(text));
    if (actions.length) renderActionCards(actions);
    chatHistory.push({ role: "user", content: message });
    chatHistory.push({ role: "assistant", content: text + (actions.length ? `\n[${t("actionsProposedNote")}: ${actions.map(describeAction).join("; ")}]` : "") });
  } catch (_) {
    typing.remove();
    addChatMessage("ai error", esc(t("error_chat")));
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
}

// Markdown from the model, with anything executable removed (it may quote dataset values)
const SAFE_TAGS = new Set(["P", "BR", "STRONG", "B", "EM", "I", "CODE", "PRE", "UL", "OL", "LI", "BLOCKQUOTE",
  "H1", "H2", "H3", "H4", "H5", "H6", "TABLE", "THEAD", "TBODY", "TR", "TH", "TD", "A", "HR", "DEL", "SPAN"]);

function renderMarkdown(text) {
  if (typeof marked === "undefined") return esc(text).replace(/\n/g, "<br>");
  const tpl = document.createElement("template");
  tpl.innerHTML = marked.parse(text);
  const walk = (node) => {
    Array.from(node.children).forEach((child) => {
      if (!SAFE_TAGS.has(child.tagName)) {
        child.replaceWith(document.createTextNode(child.textContent));
        return;
      }
      Array.from(child.attributes).forEach((attr) => {
        const keep = child.tagName === "A" && attr.name === "href" && /^https?:/i.test(attr.value);
        if (!keep) child.removeAttribute(attr.name);
      });
      if (child.tagName === "A") { child.target = "_blank"; child.rel = "noopener"; }
      walk(child);
    });
  };
  walk(tpl.content);
  return tpl.innerHTML;
}

// ── Assistant actions ────────────────────────────────────────────────────────
// The assistant only proposes; nothing changes until the user clicks "Apply".
const chatHistory = [];
const RESULT_TABS = { performance: "perf", importance: "importance", shap: "shap", models: "models" };

function appState() {
  const state = {
    mode: appMode === 1 ? "train" : "predict",
    step: ["", "data", "train", "results"][wizardCurrentStep] || "data",
    actions_enabled: !(typeof STUDY !== "undefined" && STUDY),
    dataset: null,
    // Labels exactly as the user sees them, so the assistant names buttons correctly
    ui_labels: {
      steps: [t("stepData"), t("stepTrain"), t("stepResults")],
      continue_button: t("continue"), train_button: t("startTraining"), stop_button: t("stopTraining"),
      advanced_options: t("advancedOptions"), metric_field: t("evalMetric"), time_field: t("timeTitle"),
      variables_table: t("variablesTitle"), use_column: t("use"), results_tabs: [t("tabPerformance"), t("tabImportance"), t("tabShap"), t("tabModels")],
      downloads: [t("downloadModel"), t("downloadPDF"), t("downloadPlots")], modes: [t("tabTrain"), t("tabPredict")],
    },
  };
  if (datasetLoaded) {
    const target = $("target-column").value;
    state.dataset = {
      name: lastDatasetName, rows: totalDataRows,
      columns: dataSetStats.slice(0, 300).map(s => ({
        name: s.name, type: s.type, missing_pct: +(100 * s.missing / (totalDataRows || 1)).toFixed(1), unique: s.unique,
      })),
    };
    state.target = target;
    state.task = taskTypeFor(target);
    state.excluded = getDroppedColumns();
    state.metric = $("adv-eval-metric").value;
    state.metric_options = Array.from($("adv-eval-metric").options).map(o => o.value);
    state.time_minutes = parseInt($("training-time-limit").value, 10);
    state.cross_validation = $("adv-use-cv").checked;
    state.training_running = !$("stop-training-btn").hidden;
  }
  if (lastResult) {
    const importance = Object.entries(lastResult.feature_importance || {}).sort((a, b) => b[1] - a[1]).slice(0, 10);
    state.results = {
      best_model: lastResult.best_model,
      task_type: lastResult.task_type,
      metrics: Object.fromEntries(metricEntries(lastResult).map(([k, m]) => [k, +m.value.toFixed(4)])),
      top_importance: Object.fromEntries(importance.map(([k, v]) => [k, +v.toFixed(4)])),
      tabs: Array.from(document.querySelectorAll("#training-results .tab")).map(b =>
        Object.keys(RESULT_TABS).find(k => RESULT_TABS[k] === b.dataset.tab)).filter(Boolean),
    };
  }
  return state;
}

function describeAction({ name, args }) {
  const list = (cols) => cols.map(c => `«${c}»`).join(", ");
  const label = {
    exclude_variables: () => t("act_exclude", { cols: list(args.columns) }),
    include_variables: () => t("act_include", { cols: list(args.columns) }),
    set_target: () => t("act_target", { col: args.column }),
    set_metric: () => t("act_metric", { metric: METRIC_INFO[args.metric]?.label || $("adv-eval-metric").querySelector(`option[value="${CSS.escape(args.metric)}"]`)?.textContent || args.metric }),
    set_training_time: () => t("act_time", { time: formatMinutes(args.minutes) }),
    set_cross_validation: () => t(args.enabled ? "act_cv_on" : "act_cv_off"),
    start_training: () => t("act_train"),
    go_to_step: () => t("act_step", { step: t({ data: "stepData", train: "stepTrain", results: "stepResults" }[args.step]) }),
    show_results_tab: () => t("act_tab", { tab: t({ performance: "tabPerformance", importance: "tabImportance", shap: "tabShap", models: "tabModels" }[args.tab]) }),
    generate_shap: () => t("act_shap"),
    switch_mode: () => t(args.mode === "predict" ? "act_mode_predict" : "act_mode_train"),
  }[name];
  return label ? label() : name;
}

function setVariablesIncluded(columns, include) {
  const target = $("target-column").value;
  let changed = 0;
  columns.forEach((col) => {
    if (col === target) return;
    const cb = document.querySelector(`#vars-table .var-include[value="${CSS.escape(col)}"]`);
    if (cb && cb.checked !== include) {
      cb.checked = include;
      cb.dispatchEvent(new Event("change", { bubbles: true }));
      changed += 1;
    }
  });
  if (!changed) throw new Error(t("actNothingToChange"));
  setWizardStep(STEP.DATA);
}

function setSelect(id, value) {
  const select = $(id);
  if (!Array.from(select.options).some(o => o.value === value)) throw new Error(t("actInvalid"));
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

const ACTIONS = {
  exclude_variables: ({ columns }) => setVariablesIncluded(columns, false),
  include_variables: ({ columns }) => setVariablesIncluded(columns, true),
  set_target: ({ column }) => { setSelect("target-column", column); setWizardStep(STEP.TRAIN); },
  set_metric: ({ metric }) => { setSelect("adv-eval-metric", metric); setWizardStep(STEP.TRAIN); $("advanced-options-section").open = true; },
  set_training_time: ({ minutes }) => {
    const slider = $("training-time-limit");
    slider.value = String(Math.max(1, Math.min(180, minutes)));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
    updateTimeDisplay();
    setWizardStep(STEP.TRAIN);
  },
  set_cross_validation: ({ enabled }) => {
    const cb = $("adv-use-cv");
    cb.checked = enabled;
    cb.dispatchEvent(new Event("change", { bubbles: true }));
    toggleValSplit();
    setWizardStep(STEP.TRAIN);
    $("advanced-options-section").open = true;
  },
  start_training: () => { setWizardStep(STEP.TRAIN); startTraining(); },
  go_to_step: ({ step }) => setWizardStep({ data: STEP.DATA, train: STEP.TRAIN, results: STEP.RESULTS }[step]),
  show_results_tab: ({ tab }) => { setWizardStep(STEP.RESULTS); setResultsTab(RESULT_TABS[tab]); },
  generate_shap: () => { setWizardStep(STEP.RESULTS); setResultsTab("shap"); generateShapPlot(); },
  switch_mode: ({ mode }) => setMode(mode),
};

function applyAction(action, card) {
  const status = card.querySelector(".action-status");
  try {
    if (!ACTIONS[action.name]) throw new Error(t("actInvalid"));
    ACTIONS[action.name](action.args || {});
    card.classList.add("applied");
    status.textContent = `✓ ${t("actApplied")}`;
    chatHistory.push({ role: "user", content: `[${t("actionAppliedNote")}: ${describeAction(action)}]` });
  } catch (err) {
    card.classList.add("failed");
    status.textContent = `✕ ${err.message || t("actInvalid")}`;
  }
  card.querySelectorAll("button").forEach(b => { b.disabled = true; });
}

function renderActionCards(actions) {
  const box = document.createElement("div");
  box.className = "action-list";
  actions.forEach((action) => {
    const card = document.createElement("div");
    card.className = "action-card";
    card.innerHTML = `<span class="action-label">${esc(describeAction(action))}</span>
      <span class="action-status muted"></span>
      <div class="action-buttons">
        <button type="button" class="btn btn-primary btn-sm">${t("actApply")}</button>
        <button type="button" class="btn btn-ghost btn-sm">${t("actDismiss")}</button>
      </div>`;
    const [apply, dismiss] = card.querySelectorAll("button");
    apply.addEventListener("click", () => applyAction(action, card));
    dismiss.addEventListener("click", () => {
      card.classList.add("dismissed");
      card.querySelector(".action-status").textContent = t("actDismissed");
      card.querySelectorAll("button").forEach(b => { b.disabled = true; });
    });
    box.appendChild(card);
  });
  if (actions.length > 1) {
    const all = document.createElement("button");
    all.type = "button";
    all.className = "btn btn-secondary btn-sm action-apply-all";
    all.textContent = t("actApplyAll");
    all.addEventListener("click", () => {
      box.querySelectorAll(".action-card:not(.applied):not(.dismissed):not(.failed)").forEach((card, i) => {
        applyAction(actions[Array.from(box.querySelectorAll(".action-card")).indexOf(card)], card);
      });
      all.disabled = true;
    });
    box.appendChild(all);
  }
  $("chat-box").appendChild(box);
  $("chat-box").scrollTop = $("chat-box").scrollHeight;
}

// ── Toast ────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showAlert(message, type = "error", duration = 7000) {
  const box = $("custom-alert");
  box.textContent = message;
  box.className = `toast toast-${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => box.classList.add("hidden"), duration);
}
