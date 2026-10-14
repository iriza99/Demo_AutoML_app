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
let trainingId = null;
let pollingIntervalTraining = null;
let currentShapRequestId = null;
let appMode = 1;                 // 1 = train, 2 = predict
let detectedTimeColumns = [];
let wizardCurrentStep = 1;
let wizardMaxStep = 1;
let datasetLoaded = false;
let totalDataRows = 0;
let lastTrainConfig = null;      // options used for the last training (shown in the technical details)
let lastPredictionSummary = null;
let expertMode = (() => { try { return localStorage.getItem("expertMode") === "1"; } catch (_) { return false; } })();

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
  $("expert-toggle").checked = expertMode;
  applyExpertMode();

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

function toggleExpertMode() {
  expertMode = $("expert-toggle").checked;
  try { localStorage.setItem("expertMode", expertMode ? "1" : "0"); } catch (_) { /* private mode */ }
  applyExpertMode();
}

// Expert mode shows technical details (model names, configuration, all models) opened by default
function applyExpertMode() {
  document.body.classList.toggle("expert", expertMode);
  const adv = $("advanced-options-section");
  if (adv) adv.open = expertMode;
  const details = $("res-models");
  if (details) details.open = expertMode;
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
  shapSummary = null;
  currentShapRequestId = null;
  $("training-results").innerHTML = "";
  if (wizardMaxStep > STEP.TRAIN) wizardMaxStep = STEP.TRAIN;
}

// ── Step 2 · Train ───────────────────────────────────────────────────────────
// Likely outcome columns, offered as suggestions; the user always makes the choice
function suggestTargets(header) {
  const hints = /(target|label|outcome|class|result|diagnos|event|death|mortal|readmi|sepsis|status|survival|y$)/i;
  const named = [...header].reverse().filter(h => hints.test(h));
  return [...new Set([...named, header[header.length - 1]])].slice(0, 3);
}

function refreshTargetOptions(initial = false) {
  const select = $("target-column");
  const header = datasetRows[0];
  const current = select.value;
  select.innerHTML = `<option value="">${esc(t("chooseTarget"))}</option>` +
    header.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  select.value = !initial && current && header.includes(current) ? current : "";
  onTargetChange();
}

function chooseTarget(col) {
  $("target-column").value = col;
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
  $("target-column").classList.toggle("is-empty", !target);
  if (!stat) {
    const suggestions = datasetRows ? suggestTargets(datasetRows[0]) : [];
    box.innerHTML = suggestions.length ? `<div class="target-suggest"><span class="muted">${t("targetSuggestions")}</span>
      ${suggestions.map(c => `<button type="button" class="chip-btn" onclick="chooseTarget(${jsArg(c)})">${esc(c)}</button>`).join("")}</div>` : "";
    return;
  }
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
  // Recommended default: ROC AUC for two classes (not fooled by imbalance), the first option otherwise
  const targetStat = (dataSetStats || []).find(st => st.name === $("target-column").value);
  const binary = task === "classification" && targetStat?.unique === 2;
  let options = ADVANCED_METRICS[task];
  if (task === "classification" && !binary) options = options.filter(m => m.value !== "roc_auc");
  const recommended = binary ? "roc_auc" : options[0].value;
  metricSelect.innerHTML = options.map(m => `<option value="${m.value}">${m.label}${m.value === recommended ? ` (${t("recommended")})` : ""}</option>`).join("");
  const key = task + (binary ? ":2" : "");
  metricSelect.value = options.some(m => m.value === previousMetric) && metricSelect.dataset.task === key ? previousMetric : recommended;
  metricSelect.dataset.task = key;

  const models = task === "timeseries" ? ADVANCED_MODELS.timeseries : ADVANCED_MODELS.tabular;
  const container = $("adv-models-container");
  const unchecked = new Set(Array.from(container.querySelectorAll("input:not(:checked)")).map(cb => cb.value));
  container.innerHTML = models.map(m => `<label class="chip-check"><input type="checkbox" value="${m.value}" ${unchecked.has(m.value) ? "" : "checked"}><span>${m.label}</span></label>`).join("");

  $("adv-cv-container").hidden = task === "timeseries";
  if (task === "timeseries") $("advanced-options-section").open = true;
}

function resetAdvancedOptions() {
  $("advanced-options-section").open = expertMode;
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
  const used = datasetRows[0].length - (target ? 1 : 0) - dropped.filter(c => c !== target).length;
  const task = taskTypeFor(target);
  const metric = target ? ($("adv-eval-metric").selectedOptions[0]?.textContent || "—") : "—";
  const items = [
    [t("data"), `${lastDatasetName}`],
    [t("target"), target || "—"],
    [t("task"), target ? { classification: t("taskClassification"), regression: t("taskRegression"), timeseries: t("taskTimeseries") }[task] : "—"],
    [t("variablesUsed"), `${used}${dropped.length ? ` <span class="muted">(${dropped.length} ${t("excluded")})</span>` : ""}`],
    [t("metric"), metric],
    [t("time"), $("time-limit-display").textContent],
  ];
  list.innerHTML = items.map(([k, v]) => `<dt>${k}</dt><dd>${typeof v === "string" && v.includes("<span") ? v : esc(v)}</dd>`).join("");
  $("start-training-btn").disabled = !target;
  $("start-training-btn").title = target ? "" : t("chooseTargetFirst");
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
  formData.append("lang", currentLang());
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
  lastTrainConfig = {
    metricLabel: Object.values(ADVANCED_METRICS).flat().find(m => m.value === metric)?.label || metric || "—",
    cv: useCv, valPct: parseInt($("adv-val-size").value, 10), minutes,
    excluded: dropped.filter(c => c !== target),
    families: Array.from($("adv-models-container").querySelectorAll("input:checked")).map(cb => cb.parentElement.textContent.trim()),
  };

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

// ── Native charts (same tokens as the UI, light and dark) ───────────────────
const SEQ_STEPS = 7;

function chartTip(evt, html) {
  let tip = document.getElementById("viz-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "viz-tip";
    tip.className = "viz-tip";
    document.body.appendChild(tip);
  }
  if (!html) { tip.hidden = true; return; }
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: w, innerHeight: h } = window;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(evt.clientX + pad, w - r.width - 8)}px`;
  tip.style.top = `${Math.min(evt.clientY + pad, h - r.height - 8)}px`;
}

function renderNativeCharts(pd) {
  const charts = [];
  if (pd.confusion) charts.push(confusionMatrixChart(pd));
  if (pd.roc) charts.push(rocChart(pd.roc));
  if (pd.y_true) {
    charts.push(scatterChart(pd));
    charts.push(errorHistogramChart(pd.error_hist));
  }
  return charts;
}

function confusionMatrixChart(pd) {
  const labels = pd.labels;
  const m = pd.confusion;
  const n = labels.length;
  const rowTotals = m.map(r => r.reduce((a, b) => a + b, 0) || 1);
  const head = `<div class="cm-corner"><span>${t("cmActual")} ↓ · ${t("cmPredicted")} →</span></div>` +
    labels.map(l => `<div class="cm-colhead">${esc(l)}</div>`).join("");
  const rows = labels.map((label, i) => `<div class="cm-rowhead">${esc(label)}</div>` + m[i].map((v, j) => {
    const share = v / rowTotals[i];
    const step = Math.min(SEQ_STEPS - 1, Math.round(share * (SEQ_STEPS - 1)));
    const tip = esc(t("cmTip", { n: fmt(v, 0), actual: label, predicted: labels[j], pct: pct(share) }));
    return `<div class="cm-cell seq-${step}${i === j ? " cm-diag" : ""}" data-tip="${tip}"><strong>${fmt(v, 0)}</strong><small>${pct(share, 0)}</small></div>`;
  }).join("")).join("");
  let stats = "";
  if (n === 2) {
    // Rates for the positive class (second label, as in the ROC curve)
    const [[tn, fp], [fn, tp]] = m;
    const rate = (a, b) => (a + b ? a / (a + b) : null);
    const items = [["sensitivity", rate(tp, fn)], ["specificity", rate(tn, fp)], ["ppv", rate(tp, fp)], ["npv", rate(tn, fn)]];
    stats = `<dl class="cm-stats">${items.map(([k, v]) => `<div><dt>${t(`stat_${k}`)}</dt><dd>${pct(v)}</dd></div>`).join("")}</dl>
      <p class="muted cm-note">${t("cmPositive", { label: labels[1] })}</p>`;
  }
  return `<figure class="plot card chart">
    <figcaption class="chart-title">${t("cmTitle")}</figcaption>
    <p class="muted chart-sub">${t("cmSub")}</p>
    <div class="cm" style="--n:${n}" onmousemove="const c=event.target.closest('.cm-cell'); chartTip(event, c ? c.dataset.tip : '')" onmouseleave="chartTip(event, '')">${head}${rows}</div>
    ${stats}
  </figure>`;
}

function axisTicks(min, max, count = 5) {
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + i * step);
}

// Round tick values (1, 2, 2.5, 5 × 10^k) that fall inside [min, max]
function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  return [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw) || raw;
}
function niceTicks(min, max, count = 5) {
  const step = niceStep(max - min, count);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}
function niceDomain(min, max, count = 5) {
  const step = niceStep(max - min, count);
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}

function svgFrame({ w, h, m, xTicks, yTicks, x, y, xLabel, yLabel, xFmt = v => fmt(v, 2), yFmt = v => fmt(v, 2) }) {
  const grid = yTicks.map(v => `<line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${y(v)}" y2="${y(v)}"/>
      <text class="tick" x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${yFmt(v)}</text>`).join("") +
    xTicks.map(v => `<text class="tick" x="${x(v)}" y="${h - m.b + 18}" text-anchor="middle">${xFmt(v)}</text>`).join("");
  return `${grid}
    <line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}"/>
    <text class="axis-label" x="${(m.l + w - m.r) / 2}" y="${h - 6}" text-anchor="middle">${esc(xLabel)}</text>
    <text class="axis-label" transform="translate(14 ${(m.t + h - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
}

function rocChart(roc) {
  const w = 420, h = 360, m = { l: 52, r: 16, t: 12, b: 48 };
  const x = v => m.l + v * (w - m.l - m.r);
  const y = v => h - m.b - v * (h - m.t - m.b);
  const pts = roc.fpr.map((f, i) => [f, roc.tpr[i]]);
  const line = pts.map(([f, tp], i) => `${i ? "L" : "M"}${x(f).toFixed(1)},${y(tp).toFixed(1)}`).join("");
  const area = `${line}L${x(1)},${y(0)}L${x(0)},${y(0)}Z`;
  const ticks = axisTicks(0, 1, 5);
  const id = `roc-${uid()}`;
  window[id] = roc;
  return `<figure class="plot card chart">
    <figcaption class="chart-title">${t("rocTitle")} · AUC ${fmt(roc.auc, 3)}</figcaption>
    <p class="muted chart-sub">${t("rocSub", { label: roc.positive_label })}</p>
    <svg class="viz" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(t("rocTitle"))}"
         onmousemove="rocHover(event, '${id}')" onmouseleave="rocHover(event, '${id}', true)">
      ${svgFrame({ w, h, m, xTicks: ticks, yTicks: ticks, x, y, xLabel: t("rocX"), yLabel: t("rocY") })}
      <line class="ref" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>
      <text class="ref-label" x="${x(0.62)}" y="${y(0.56)}">${t("rocChance")}</text>
      <path class="area" d="${area}"/>
      <path class="series" d="${line}"/>
      <g class="hover" hidden><line class="crosshair"/><circle r="5" class="dot"/></g>
      <rect class="hit" x="${m.l}" y="${m.t}" width="${w - m.l - m.r}" height="${h - m.t - m.b}"/>
    </svg>
  </figure>`;
}

function rocHover(evt, id, leave = false) {
  const svg = evt.currentTarget;
  const g = svg.querySelector(".hover");
  if (leave) { g.hidden = true; chartTip(evt, ""); return; }
  const roc = window[id];
  const w = 420, h = 360, m = { l: 52, r: 16, t: 12, b: 48 };
  const box = svg.getBoundingClientRect();
  const fx = Math.max(0, Math.min(1, ((evt.clientX - box.left) * (w / box.width) - m.l) / (w - m.l - m.r)));
  let best = 0;
  roc.fpr.forEach((f, i) => { if (Math.abs(f - fx) < Math.abs(roc.fpr[best] - fx)) best = i; });
  const px = m.l + roc.fpr[best] * (w - m.l - m.r), py = h - m.b - roc.tpr[best] * (h - m.t - m.b);
  g.hidden = false;
  const cross = g.querySelector(".crosshair");
  cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("y1", m.t); cross.setAttribute("y2", h - m.b);
  const dot = g.querySelector(".dot");
  dot.setAttribute("cx", px); dot.setAttribute("cy", py);
  const thr = roc.threshold[best];
  chartTip(evt, `<strong>${t("stat_sensitivity")} ${pct(roc.tpr[best])}</strong><br>${t("stat_specificity")} ${pct(1 - roc.fpr[best])}` +
    (thr != null && thr <= 1 ? `<br><span class="muted">${t("rocThreshold")} ${fmt(thr, 2)}</span>` : ""));
}

function niceRange(values) {
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.05;
  return [min - pad, max + pad];
}

function scatterChart(pd) {
  const w = 420, h = 360, m = { l: 60, r: 16, t: 12, b: 48 };
  const [lo, hi] = niceDomain(...niceRange(pd.y_true.concat(pd.y_pred)));
  const x = v => m.l + (v - lo) / (hi - lo) * (w - m.l - m.r);
  const y = v => h - m.b - (v - lo) / (hi - lo) * (h - m.t - m.b);
  const ticks = niceTicks(lo, hi, 5);
  const dots = pd.y_true.map((a, i) => `<circle class="pt" r="4" cx="${x(a).toFixed(1)}" cy="${y(pd.y_pred[i]).toFixed(1)}"><title>${t("cmActual")} ${fmt(a)} · ${t("cmPredicted")} ${fmt(pd.y_pred[i])}</title></circle>`).join("");
  return `<figure class="plot card chart">
    <figcaption class="chart-title">${t("scatterTitle")}</figcaption>
    <p class="muted chart-sub">${t("scatterSub", { n: fmt(pd.y_true.length, 0), total: fmt(pd.n, 0) })}</p>
    <svg class="viz" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(t("scatterTitle"))}">
      ${svgFrame({ w, h, m, xTicks: ticks, yTicks: ticks, x, y, xLabel: t("cmActual"), yLabel: t("cmPredicted"), xFmt: v => fmt(v, 3), yFmt: v => fmt(v, 3) })}
      <line class="ref" x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}"/>
      <g class="pts">${dots}</g>
    </svg>
  </figure>`;
}

function errorHistogramChart(hist) {
  const w = 420, h = 300, m = { l: 52, r: 16, t: 12, b: 48 };
  const edges = hist.edges, counts = hist.counts;
  const lo = edges[0], hi = edges[edges.length - 1];
  const maxC = niceDomain(0, Math.max(...counts) || 1, 4)[1];
  const x = v => m.l + (v - lo) / (hi - lo || 1) * (w - m.l - m.r);
  const y = v => h - m.b - v / maxC * (h - m.t - m.b);
  const bars = counts.map((c, i) => {
    const x0 = x(edges[i]) + 1, x1 = x(edges[i + 1]) - 1, top = y(c), base = h - m.b;
    if (c === 0 || x1 <= x0) return "";
    const r = Math.min(4, (x1 - x0) / 2, base - top);
    return `<path class="bar" d="M${x0},${base}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x1 - r}Q${x1},${top} ${x1},${top + r}V${base}Z"><title>${fmt(edges[i], 3)} – ${fmt(edges[i + 1], 3)}: ${fmt(c, 0)}</title></path>`;
  }).join("");
  const zero = lo < 0 && hi > 0 ? `<line class="ref" x1="${x(0)}" x2="${x(0)}" y1="${m.t}" y2="${h - m.b}"/>` : "";
  return `<figure class="plot card chart">
    <figcaption class="chart-title">${t("errorsTitle")}</figcaption>
    <p class="muted chart-sub">${t("errorsSub")}</p>
    <svg class="viz" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(t("errorsTitle"))}">
      ${svgFrame({ w, h, m, xTicks: niceTicks(lo, hi, 5), yTicks: niceTicks(0, maxC, 4), x, y, xLabel: t("errorsX"), yLabel: t("errorsY"), xFmt: v => fmt(v, 3), yFmt: v => fmt(v, 0) })}
      ${bars}${zero}
    </svg>
  </figure>`;
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

  const target = lastUsedTargetColumn || "";
  const taskLabel = { binary: t("taskBinary"), multiclass: t("taskMulticlass"), regression: t("taskRegression"), timeseries: t("taskTimeseries") }[data.task_type] || data.task_type;
  const baseline = ["binary", "multiclass"].includes(data.task_type) ? majorityShare(target) : null;
  const main = metricEntries(data)[0];
  const band = main ? qualityBand(main[0], main[1].value) : null;
  const labels = data.plot_data?.labels;
  const headline = !band ? t("verdictDone", { target })
    : data.task_type === "binary" && labels?.length === 2 ? t(`verdictBin_${band}`, { a: labels[0], b: labels[1] })
    : t(`verdict_${band}`, { target });

  const tiles = metricEntries(data).map(([key, m]) => {
    const info = METRIC_INFO[key] || { label: key.toUpperCase() };
    const b = qualityBand(key, m.value);
    const value = key === "accuracy" ? pct(m.value) : fmt(m.value);
    const help = t(`help_${key}`) !== `help_${key}` ? t(`help_${key}`) : "";
    let note = b ? `<span class="badge badge-${b}">${t(`quality_${b}`)}</span>` : "";
    if (key === "accuracy" && baseline != null) note = `<span class="metric-note">${t("vsBaseline", { pct: pct(baseline) })}</span>`;
    return `<div class="metric" title="${esc(help)}"><span class="metric-label">${info.label}</span><span class="metric-value">${value}</span>${note}<span class="metric-help">${esc(help)}</span></div>`;
  }).join("");

  // Native charts when the raw data is available; the PNGs remain for downloads, the PDF and old results
  const plots = data.plot_data ? renderNativeCharts(data.plot_data) : metricEntries(data).flatMap(([key, m]) => [m.plot, m.plot_hist].filter(Boolean).map((p, i) =>
    `<figure class="plot card"><img src="data:image/png;base64,${p}" alt="${esc(key)}${i ? "_errors" : ""}" loading="lazy"><figcaption>${METRIC_INFO[key]?.label || key}</figcaption></figure>`));
  if (data.forecast_plot) plots.push(`<figure class="plot card"><img src="data:image/png;base64,${data.forecast_plot}" alt="forecast" loading="lazy"><figcaption>${t("forecast")}</figcaption></figure>`);

  const hasImportance = data.feature_importance && Object.keys(data.feature_importance).length;
  const hasShap = data.task_type !== "timeseries";
  const sections = [{ id: "performance", label: t("secPerformance") }];
  if (hasImportance || hasShap) sections.push({ id: "importance", label: t("secVariables") });
  sections.push({ id: "models", label: t("secDetails") });

  $("training-results").innerHTML = `
    <div class="verdict card">
      <div class="verdict-top">
        <div class="verdict-text">
          <span class="eyebrow">${taskLabel}</span>
          <h2>${esc(headline)}</h2>
          <p class="muted">${t("verdictSub", { rows: fmt(totalDataRows, 0), time: fmtDuration(data.train_time) })}
            <span class="expert-only"> · ${t("bestModel")}: <span class="mono">${esc(data.best_model)}</span></span>
            ${data.used_gpu ? ' <span class="pill pill-ok">GPU</span>' : ""}</p>
        </div>
        <div class="verdict-actions">
          <button class="btn btn-primary" type="button" onclick="useLastTrainedModel(true)">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>${t("predictWithModel")}</button>
          <a class="btn btn-secondary" href="${data.download_url || "#"}" download>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v12M7 11l5 5 5-5M4 20h16"/></svg>${t("downloadModel")}</a>
          <button class="btn btn-ghost" type="button" onclick="downloadPDF()">${t("downloadPDF")}</button>
        </div>
      </div>
      ${tiles ? `<div class="metric-strip">${tiles}</div>` : ""}
      ${renderResultChecks(data, baseline)}
    </div>
    <nav class="section-nav" aria-label="${esc(t("stepResults"))}">
      ${sections.map(s => `<button type="button" class="chip-btn" onclick="showResultsSection('${s.id}')">${s.label}</button>`).join("")}
      <button type="button" class="chip-btn chip-accent" onclick="openChatWith(${jsArg(t("suggestGood"))})">${t("askButton")}</button>
    </nav>

    <section class="res-section" id="res-performance">
      <div class="res-head"><h3>${t("secPerformance")}</h3><p class="muted">${t("secPerformanceSub")}</p></div>
      ${plots.length ? `<div class="plot-grid">${plots.join("")}</div>` : `<p class="muted">${t("noPlots")}</p>`}
    </section>

    ${hasImportance || hasShap ? `
    <section class="res-section" id="res-importance">
      <div class="res-head"><h3>${t("secVariables")}</h3><p class="muted">${t("secVariablesSub")}</p></div>
      <div class="var-grid">
        ${hasImportance ? renderImportance(data.feature_importance) : ""}
        ${hasShap ? `<div class="card shap-card" id="res-shap">${renderShapBlock()}</div>` : ""}
      </div>
    </section>` : ""}

    <details class="card res-details" id="res-models" ${expertMode ? "open" : ""}>
      <summary><span>${t("secDetails")}</span><span class="muted">${t("secDetailsSub")}</span></summary>
      ${renderTrainingDetails(data)}
      ${Array.isArray(data.leaderboard) && data.leaderboard.length ? renderLeaderboard(data.leaderboard, data.best_model) : ""}
      <div class="row-gap"><button class="btn btn-ghost btn-sm" type="button" onclick="downloadAllPlots()">${t("downloadPlots")}</button></div>
    </details>`;
  updateStepSubtitles();
}

function fmtDuration(seconds) {
  if (typeof seconds !== "number") return "—";
  if (seconds < 90) return `${fmt(seconds, 0)} s`;
  return `${fmt(seconds / 60, 1)} min`;
}

// Automatic sanity checks on the results, in plain words, with a one-click fix when there is one
function renderResultChecks(data, baseline) {
  const checks = [];
  const imp = Object.entries(data.feature_importance || {}).sort((a, b) => b[1] - a[1]);
  const statsByName = Object.fromEntries((dataSetStats || []).map(s => [s.name, s]));
  const rows = totalDataRows || 1;
  imp.slice(0, 3).forEach(([col, v]) => {
    const s = statsByName[col];
    if (v > 0 && s && s.unique >= 0.95 * rows && s.type !== "Date" && (s.type === "Categorical" || /(^|_|\b)(id|code|cod|num|no|nr|number|record|episode|patient|nhc)(_|\b|$)/i.test(col) || s.unique === rows)) {
      checks.push({ text: t("checkIdentifier", { col }), fix: col });
    }
  });
  const main = metricEntries(data)[0];
  const band = main ? qualityBand(main[0], main[1].value) : null;
  if (imp.length > 2 && imp[0][1] > 0 && imp[0][1] > 4 * Math.max(imp[1][1], 1e-9) && band === "excellent" && !checks.some(c => c.fix === imp[0][0])) {
    checks.push({ text: t("checkDominant", { col: imp[0][0] }), fix: imp[0][0] });
  }
  const acc = data.metrics?.accuracy?.value;
  if (baseline != null && baseline >= 0.75 && typeof acc === "number") {
    checks.push({ text: t(acc - baseline < 0.03 ? "checkNoBetter" : "checkImbalance", { pct: pct(baseline) }) });
  }
  if (band === "weak") checks.push({ text: t("checkWeak") });
  if (!checks.length) return "";
  return `<ul class="checks">${checks.map(c => `<li>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l10 18H2zM12 10v5M12 18h.01"/></svg>
      <span>${esc(c.text)}</span>
      ${c.fix ? `<button class="btn btn-ghost btn-sm" type="button" onclick="excludeAndRetrain(${jsArg(c.fix)})">${t("checkFix", { col: c.fix })}</button>` : ""}
    </li>`).join("")}</ul>`;
}

function excludeAndRetrain(col) {
  try { setVariablesIncluded([col], false); } catch (_) { /* already excluded */ }
  setWizardStep(STEP.TRAIN);
  showAlert(t("checkFixDone", { col }), "info", 5000);
}

function showResultsSection(id) {
  setWizardStep(STEP.RESULTS);
  const node = $({ performance: "res-performance", importance: "res-importance", shap: "res-shap", models: "res-models" }[id] || "res-performance");
  if (!node) return;
  if (node.tagName === "DETAILS") node.open = true;
  setTimeout(() => node.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
}
// Kept for older callers
function setResultsTab(id) { showResultsSection({ perf: "performance" }[id] || id); }

function renderTrainingDetails(data) {
  const c = lastTrainConfig || {};
  const rows = [
    [t("task"), { binary: t("taskBinary"), multiclass: t("taskMulticlass"), regression: t("taskRegression"), timeseries: t("taskTimeseries") }[data.task_type] || data.task_type],
    [t("target"), lastUsedTargetColumn],
    [t("bestModel"), `<span class="mono">${esc(data.best_model)}</span>`],
    [t("evalMetric"), c.metricLabel || "—"],
    [t("evaluation"), c.cv === false ? t("evalHoldout", { pct: c.valPct }) : t("evalCv")],
    [t("timeTitle"), `${fmtDuration(data.train_time)} / ${formatMinutes(c.minutes || 0)}`],
    [t("excludedVars"), c.excluded?.length ? c.excluded.map(esc).join(", ") : t("none")],
    [t("modelFamilies"), c.families?.length ? c.families.map(esc).join(", ") : "—"],
    ["GPU", data.used_gpu ? t("yes") : t("no")],
  ];
  return `<dl class="details-list">${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>
    <p class="muted small">${t("ensembleNote")}</p>`;
}

// ── SHAP ─────────────────────────────────────────────────────────────────────
let shapSummary = null;

function renderShapBlock() {
  if (shapSummary) return shapChart(shapSummary);
  if (shapPlot) return `<h4>${t("shapTitle")}</h4><img class="shap-img" src="data:image/png;base64,${shapPlot}" alt="shap_summary">`;
  return `<h4>${t("shapTitle")}</h4>
    <p class="muted">${t("shapIntro")}</p>
    <div class="row-gap">
      <button id="generate-shap-button" class="btn btn-secondary" type="button" onclick="generateShapPlot()">${t("generateShap")}</button>
      <div id="training-spinner-shap" class="spinner" hidden></div>
    </div>`;
}

function shapStep(f) { return f == null ? "na" : String(Math.max(0, Math.min(4, Math.round(f * 4)))); }

function shapSentence(f, s) {
  const d = f.direction;
  const cls = s.explained_class;
  const up = s.task === "regression" ? t("shapUpReg") : t("shapUpCls", { cls });
  const down = s.task === "regression" ? t("shapDownReg") : t("shapDownCls", { cls });
  if (f.levels && d != null && Math.abs(d) >= 0.3) {
    return t("shapLevel", { col: f.name, level: f.levels[d > 0 ? 1 : 0], effect: up });
  }
  if (f.numeric && d != null && Math.abs(d) >= 0.3) {
    return t(d > 0 ? "shapHigh" : "shapLow", { col: f.name, effect: up });
  }
  if (f.categories?.length && !f.levels) {
    const top = [...f.categories].sort((a, b) => b.mean - a.mean)[0];
    if (top && top.mean > 0) return t("shapLevel", { col: f.name, level: top.value, effect: up });
  }
  return t("shapMixed", { col: f.name, down });
}

function shapChart(s) {
  const feats = s.features.slice(0, 10);
  const w = 560, rowH = 34, m = { l: 150, r: 16, t: 8, b: 44 };
  const h = m.t + m.b + rowH * feats.length;
  const all = feats.flatMap(f => (f.points || []).map(p => p[0]));
  const [lo, hi] = niceDomain(Math.min(0, ...all), Math.max(0, ...all), 5);
  const x = v => m.l + (v - lo) / ((hi - lo) || 1) * (w - m.l - m.r);
  const rows = feats.map((f, r) => {
    const cy = m.t + rowH * r + rowH / 2;
    const pts = f.points || [];
    // Beeswarm: spread each point vertically according to how crowded its x position is
    const xs = pts.map(p => x(p[0]));
    const counts = new Map();
    xs.forEach(px => { const b = Math.round(px / 3); counts.set(b, (counts.get(b) || 0) + 1); });
    const maxOff = rowH / 2 - 3;
    const dots = pts.map((p, i) => {
      const crowd = counts.get(Math.round(xs[i] / 3)) || 1;
      const spread = Math.min(maxOff, 1.6 * Math.sqrt(crowd - 1) + (crowd > 1 ? 1.5 : 0));
      const jitter = (((i * 2654435761) >>> 0) % 1000) / 1000 * 2 - 1;  // deterministic in [-1, 1]
      const off = spread * jitter;
      const frac = f.levels ? (p[1] == null ? null : p[1] / Math.max(1, f.levels.length - 1)) : (f.numeric ? p[1] : null);
      const tip = `<strong>${esc(f.name)} = ${esc(p[2] == null ? "∅" : p[2])}</strong><br>SHAP ${p[0] >= 0 ? "+" : ""}${fmt(p[0], 3)}`;
      return `<circle class="sv sv-${shapStep(frac)}" r="3" cx="${xs[i].toFixed(1)}" cy="${(cy + off).toFixed(1)}" data-tip="${esc(tip)}"/>`;
    }).join("");
    return `<text class="tick shap-name" x="${m.l - 10}" y="${cy + 4}" text-anchor="end">${esc(f.name.length > 20 ? f.name.slice(0, 19) + "…" : f.name)}<title>${esc(f.name)}</title></text>
      <line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${cy}" y2="${cy}"/>${dots}`;
  }).join("");
  const ticks = niceTicks(lo, hi, 5).map(v => `<text class="tick" x="${x(v)}" y="${h - m.b + 16}" text-anchor="middle">${fmt(v, 2)}</text>`).join("");
  const cls = s.explained_class;
  const leftLabel = s.task === "regression" ? t("shapAxisLowReg") : t("shapAxisLow", { cls });
  const rightLabel = s.task === "regression" ? t("shapAxisHighReg") : t("shapAxisHigh", { cls });
  const sentences = feats.slice(0, 5).map(f => `<li>${esc(shapSentence(f, s))}</li>`).join("");
  return `<h4>${t("shapTitle")}</h4>
    <p class="muted">${t("shapRead")}</p>
    <div class="shap-legend"><span>${t("shapValueLow")}</span>${[0, 1, 2, 3, 4].map(i => `<i class="sv-key sv-${i}"></i>`).join("")}<span>${t("shapValueHigh")}</span><i class="sv-key sv-na"></i><span>${t("shapNoValue")}</span></div>
    <div class="viz-scroll"><svg class="viz shap-viz" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(t("shapTitle"))}"
         onmousemove="const c=event.target.closest('circle'); chartTip(event, c ? c.dataset.tip : '')" onmouseleave="chartTip(event, '')">
      ${rows}
      <line class="ref" x1="${x(0)}" x2="${x(0)}" y1="${m.t}" y2="${h - m.b}"/>
      <line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}"/>
      ${ticks}
      <text class="axis-label" x="${x(0) - 8}" y="${h - 8}" text-anchor="end">← ${esc(leftLabel)}</text>
      <text class="axis-label" x="${x(0) + 8}" y="${h - 8}" text-anchor="start">${esc(rightLabel)} →</text>
    </svg></div>
    <h4 class="shap-summary-title">${t("shapInWords")}</h4>
    <ul class="shap-sentences">${sentences}</ul>
    <p class="muted small expert-only">${t("shapExpert", { model: s.explained_model || "—", n: fmt(s.n, 0) })}</p>`;
}

function renderImportance(importance) {
  const all = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const entries = all.slice(0, 12);
  const max = Math.max(...entries.map(([, v]) => Math.abs(v))) || 1;
  const rows = entries.map(([name, v]) => `
    <div class="imp-row"><span class="imp-name" title="${esc(name)}">${esc(name)}</span>
      <div class="imp-bar"><span class="${v <= 0 ? "neg" : ""}" style="width:${Math.max(1, Math.abs(v) / max * 100).toFixed(1)}%"></span></div>
      <span class="imp-val num">${fmt(v, 3)}</span></div>`).join("");
  const useless = all.filter(([, v]) => v <= 0).length;
  return `<div class="card"><h4>${t("importanceTitle")}</h4><p class="muted">${t("importanceIntro")}</p><div class="imp-list">${rows}</div>
    ${useless ? `<p class="muted small">${t("importanceUseless", { n: useless })}</p>` : ""}</div>`;
}

function renderLeaderboard(rows, best) {
  const scores = rows.map(r => r.score_val).filter(v => typeof v === "number");
  const min = Math.min(...scores), max = Math.max(...scores), span = (max - min) || 1;
  const body = rows.map(r => `<tr class="${r.model === best ? "is-best" : ""}">
      <td>${esc(r.model)}${r.model === best ? ` <span class="badge badge-excellent">${t("best")}</span>` : ""}</td>
      <td class="num"><div class="score-cell"><div class="meter meter-primary"><span style="width:${typeof r.score_val === "number" ? (15 + 85 * (r.score_val - min) / span).toFixed(1) : 0}%"></span></div>${typeof r.score_val === "number" ? fmt(r.score_val, 4) : "—"}</div></td>
      <td class="num">${typeof r.fit_time === "number" ? `${fmt(r.fit_time, 1)} s` : "—"}</td>
      <td class="num">${typeof r.pred_time_val === "number" ? `${fmt(r.pred_time_val, 2)} s` : "—"}</td></tr>`).join("");
  return `<h4>${t("tabModels")}</h4><div class="table-wrap"><table class="table">
    <thead><tr><th>${t("model")}</th><th class="num">${t("scoreVal")}</th><th class="num">${t("fitTime")}</th><th class="num">${t("predictTime")}</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

async function generateShapPlot() {
  if (!lastTrainedModelPath || !lastCleanedCsvBlob || !lastUsedTargetColumn) {
    showAlert(t("missingSHAPInfo"), "error");
    return;
  }
  if (shapSummary || shapPlot) { showResultsSection("shap"); return; }
  const requestId = uid();
  currentShapRequestId = requestId;
  const button = $("generate-shap-button");
  const spinner = $("training-spinner-shap");
  if (button) button.disabled = true;
  if (spinner) spinner.hidden = false;
  const formData = new FormData();
  formData.append("training_id", trainingId || "");
  formData.append("model_path", lastTrainedModelPath || "");
  formData.append("target_column", lastUsedTargetColumn);
  formData.append("lang", currentLang());
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
          shapSummary = data.result.shap_summary || null;
          const box = $("res-shap");
          if (box) box.innerHTML = renderShapBlock();
          resolve();
        } catch (e) { clearInterval(interval); reject(e); }
      }, 700);
    });
  } catch (err) {
    showAlert(`${t("shapGenerationError")}: ${t(err.message) !== err.message ? t(err.message) : err.message}`, "error");
    const b = $("generate-shap-button");
    if (b) b.disabled = false;
  } finally {
    const sp = $("training-spinner-shap");
    if (sp) sp.hidden = true;
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
    const r = lastResult;
    const payload = {
      lang: currentLang(),
      dataset_name: lastDatasetName,
      target_column: lastUsedTargetColumn,
      n_rows: datasetRows ? datasetRows.length - 1 : null,
      n_columns: datasetRows ? datasetRows[0].length : null,
      excluded: ((dataPreprocessing.data_preprocessing || {}).columns_drop || []).filter(c => c !== lastUsedTargetColumn),
      imputation: (dataPreprocessing.data_preprocessing || {}).imputation_missing_value || {},
      task_type: r.task_type, best_model: r.best_model, train_time: r.train_time, used_gpu: !!r.used_gpu,
      metrics: Object.fromEntries(metricEntries(r).map(([k, m]) => [k, m.value])),
      plot_data: r.plot_data || null,
      feature_importance: r.feature_importance || null,
      feature_importance_plot: r.feature_importance ? null : (r.feature_importance_plot || null),
      forecast_plot: r.forecast_plot || null,
      leaderboard: (r.leaderboard || []).map(row => ({ model: row.model, score_val: row.score_val, fit_time: row.fit_time })),
      stats: dataSetStats.map(({ hist, ...rest }) => rest),
    };
    if (shapPlot) payload.shap_plot = shapPlot;
    if (shapSummary) payload.shap_sentences = shapSummary.features.slice(0, 5).map(f => shapSentence(f, shapSummary));
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
    const formData = new FormData();
    formData.append("dataset", dataset);
    formData.append("zip_model", model);
    formData.append("lang", currentLang());
    const data = await (await fetch("/predict", { method: "POST", body: formData })).json();
    if (data.error) { showAlert(t(data.error), "error"); return; }
    if (!data.preview?.length) { showAlert(t("noPrediction"), "warning"); return; }
    PlotsPredictionResults = data.plots;
    lastPredictionSummary = data.summary || null;
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
    const groqKey = getGroqKey();
    if (groqKey) payload.groq_api_key = groqKey;
    if (appMode === 1) {
      if (dataSetStats) payload.stats = dataSetStats.map(({ hist, ...rest }) => rest);
      if (lastResult) {
        const baseline = ["binary", "multiclass"].includes(lastResult.task_type) ? majorityShare(lastUsedTargetColumn) : null;
        payload.result = {
          task_type: lastResult.task_type, target: lastUsedTargetColumn, best_model: lastResult.best_model,
          train_time: lastResult.train_time, majority_share: baseline,
          evaluation: lastTrainConfig && lastTrainConfig.cv === false ? `holdout ${lastTrainConfig.valPct}%` : "out-of-fold (cross-validation)",
          metrics: Object.fromEntries(metricEntries(lastResult).map(([k, m]) => [k, m.value])),
          leaderboard: (lastResult.leaderboard || []).slice(0, 10).map(r => ({ model: r.model, score_val: r.score_val })),
        };
        if (lastResult.feature_importance) payload.feature_importance = lastResult.feature_importance;
        if (lastResult.plot_data) payload.plot_data = lastResult.plot_data;
        if (shapSummary) payload.shap_summary = shapSummary;
      }
    } else if (lastPredictionSummary) {
      payload.prediction_summary = lastPredictionSummary;
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

function appState() {
  const state = {
    mode: appMode === 1 ? "train" : "predict",
    step: ["", "data", "train", "results"][wizardCurrentStep] || "data",
    actions_enabled: true,
    dataset: null,
    // Labels exactly as the user sees them, so the assistant names buttons correctly
    ui_labels: {
      steps: [t("stepData"), t("stepTrain"), t("stepResults")],
      continue_button: t("continue"), train_button: t("startTraining"), stop_button: t("stopTraining"),
      advanced_options: t("advancedOptions"), metric_field: t("evalMetric"), time_field: t("timeTitle"),
      variables_table: t("variablesTitle"), use_column: t("use"), results_sections: [t("secPerformance"), t("secVariables"), t("shapTitle"), t("secDetails")],
      results_buttons: [t("predictWithModel"), t("generateShap"), t("askButton")],
      settings: [t("language"), t("darkMode"), t("expertMode")],
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
      tabs: ["performance", "importance", "shap", "models"].filter(k => $({ performance: "res-performance", importance: "res-importance", shap: "res-shap", models: "res-models" }[k])),
      shap_computed: !!(shapSummary || shapPlot),
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
    show_results_tab: () => t("act_tab", { tab: t({ performance: "secPerformance", importance: "secVariables", shap: "shapTitle", models: "secDetails" }[args.tab]) }),
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
  show_results_tab: ({ tab }) => showResultsSection(tab),
  generate_shap: () => { showResultsSection("shap"); generateShapPlot(); },
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
