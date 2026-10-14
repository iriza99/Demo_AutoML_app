// User-study mode: telemetry, task brief, guardrail guidance and task submission.
// Only active when the page is opened with ?study=1, e.g.
//   /?study=1&pid=P07&cond=adaptive&ds=A&lang=es
// Without that parameter this file does nothing, so the regular app is unchanged.

const STUDY = (() => {
  const params = new URLSearchParams(window.location.search);
  if (params.get("study") !== "1") return null;
  return {
    participantId: params.get("pid") || "anon",
    condition: params.get("cond") || "baseline",
    datasetKey: params.get("ds") || null,
    lang: params.get("lang") || "es",
    sessionId: null,
    dictionary: null,
    queue: [],
    trainOverride: false,
    hasResults: false,
    submitted: false,
  };
})();

const STUDY_TEXT = {
  es: {
    task: "Tarea",
    moment: "Momento de la predicción",
    dictionary: "Diccionario de datos",
    showDictionary: "Ver diccionario de datos",
    hideDictionary: "Ocultar diccionario",
    column: "Columna",
    description: "Descripción",
    finish: "Finalizar tarea",
    guidance: "Avisos del asistente",
    ok: "Entendido",
    more: "Ver detalle",
    review: "Revisar antes de entrenar",
    trainAnyway: "Entrenar igualmente",
    beforeTraining: "Antes de entrenar",
    thanks: "Gracias por tu respuesta.",
    submitTitle: "Finalizar la tarea",
    deployQ: "¿Desplegarías este modelo en tu servicio?",
    yes: "Sí",
    no: "No",
    confidenceQ: "¿Cuánta confianza tienes en esa decisión? (1 = ninguna, 7 = total)",
    issuesQ: "¿Has detectado algún problema en los datos o en el modelo? Descríbelo.",
    cancel: "Cancelar",
    submit: "Enviar",
    submitted: "Tarea enviada. Avisa al investigador.",
    needDeploy: "Indica si desplegarías el modelo.",
    needIssues: "Describe los problemas que hayas visto, o escribe «ninguno».",
    issuesPlaceholder: "Escribe «ninguno» si no has visto problemas",
    needTraining: "Entrena al menos un modelo antes de finalizar la tarea.",
  },
  en: {
    task: "Task",
    moment: "Prediction moment",
    dictionary: "Data dictionary",
    showDictionary: "Show data dictionary",
    hideDictionary: "Hide dictionary",
    column: "Column",
    description: "Description",
    finish: "Finish task",
    guidance: "Assistant notices",
    ok: "Got it",
    more: "Details",
    review: "Review before training",
    trainAnyway: "Train anyway",
    beforeTraining: "Before training",
    thanks: "Thanks for your answer.",
    submitTitle: "Finish the task",
    deployQ: "Would you deploy this model in your department?",
    yes: "Yes",
    no: "No",
    confidenceQ: "How confident are you in that decision? (1 = not at all, 7 = completely)",
    issuesQ: "Did you notice any problem with the data or the model? Describe it.",
    cancel: "Cancel",
    submit: "Submit",
    submitted: "Task submitted. Please tell the researcher.",
    needDeploy: "Please say whether you would deploy the model.",
    needIssues: "Describe the problems you noticed, or write \"none\".",
    issuesPlaceholder: "Write \"none\" if you did not notice any problem",
    needTraining: "Train at least one model before finishing the task.",
  },
};

function st(key) {
  const lang = STUDY && STUDY_TEXT[STUDY.lang] ? STUDY.lang : "en";
  return STUDY_TEXT[lang][key] || key;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  });
  children.forEach(c => c && node.appendChild(c));
  return node;
}

// ── Telemetry ────────────────────────────────────────────────────────────────

function studyLog(type, data = {}) {
  if (!STUDY) return;
  STUDY.queue.push({ type, ts: Date.now(), ...data });
}

async function studyFlush(useBeacon = false) {
  if (!STUDY || !STUDY.sessionId || STUDY.queue.length === 0) return;
  const events = STUDY.queue.splice(0, STUDY.queue.length);
  const body = JSON.stringify({ session_id: STUDY.sessionId, events });
  if (useBeacon && navigator.sendBeacon) {
    navigator.sendBeacon("/study/telemetry", new Blob([body], { type: "application/json" }));
    return;
  }
  try {
    await fetch("/study/telemetry", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  } catch (_) {
    STUDY.queue.unshift(...events);  // retry on next flush
  }
}

function currentUiState() {
  const dropped = Array.from(document.querySelectorAll('#drop-columns-list input[type="checkbox"]:checked'))
    .map(cb => cb.value);
  return {
    dropped,
    target: document.getElementById("target-column")?.value || null,
    metric: document.getElementById("adv-eval-metric")?.value || null,
    use_cv: document.getElementById("adv-use-cv")?.checked ?? null,
  };
}

// ── Guidance rendering ───────────────────────────────────────────────────────

async function requestGuidance(stage, extraState = {}) {
  if (!STUDY || !STUDY.sessionId) return [];
  await studyFlush();
  try {
    const res = await fetch("/study/guidance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: STUDY.sessionId, stage, state: { ...currentUiState(), ...extraState } }),
    });
    const data = await res.json();
    return data.interventions || [];
  } catch (_) {
    return [];
  }
}

function renderIntervention(iv, container, onAnswered) {
  const card = el("div", { class: `study-card study-${iv.mode}`, "data-key": iv.key });
  card.appendChild(el("div", { class: "study-card-title", text: iv.title }));

  if (iv.mode === "ask") {
    card.appendChild(el("p", { text: iv.probe.question }));
    const options = el("div", { class: "study-options" });
    iv.probe.options.forEach(opt => {
      options.appendChild(el("button", {
        class: "neutral-button", type: "button", text: opt.label,
        onclick: async () => {
          options.querySelectorAll("button").forEach(b => (b.disabled = true));
          studyLog("guidance_action", { key: iv.key, mode: iv.mode, action: "answer", answer: opt.key });
          await studyFlush();
          await fetch("/study/probe", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: STUDY.sessionId, key: iv.key, answer: opt.key }),
          }).catch(() => {});
          card.appendChild(el("p", { class: "study-muted", text: st("thanks") }));
          if (onAnswered) onAnswered();
        },
      }));
    });
    card.appendChild(options);
  } else if (iv.mode === "subtle") {
    const detail = el("p", { text: iv.text, class: "study-hidden" });
    card.appendChild(el("button", {
      class: "study-link", type: "button", text: st("more"),
      onclick: () => {
        detail.classList.toggle("study-hidden");
        studyLog("guidance_action", { key: iv.key, mode: iv.mode, action: "expand" });
      },
    }));
    card.appendChild(detail);
  } else {
    card.appendChild(el("p", { text: iv.text }));
    card.appendChild(el("button", {
      class: "neutral-button", type: "button", text: st("ok"),
      onclick: () => {
        studyLog("guidance_action", { key: iv.key, mode: iv.mode, action: "dismiss" });
        card.remove();
      },
    }));
  }
  container.appendChild(card);
}

function showGuidance(interventions, stage) {
  const panel = document.getElementById("study-guidance");
  if (!panel) return;
  interventions.forEach(iv => {
    studyLog("guidance_shown", { key: iv.key, mode: iv.mode, level: iv.level, stage, blocking: iv.blocking, new: iv.new });
    if (!iv.new) return;
    const old = panel.querySelector(`[data-key="${CSS.escape(iv.key)}"]`);
    if (old) old.remove();
    renderIntervention(iv, panel.querySelector(".study-guidance-list"), () => refreshGuidance(stage));
  });
  panel.style.display = panel.querySelector(".study-card") ? "block" : "none";
}

async function refreshGuidance(stage, extraState = {}) {
  const interventions = await requestGuidance(stage, extraState);
  showGuidance(interventions, stage);
  return interventions;
}

function showBlockingModal(interventions, proceed) {
  const overlay = el("div", { class: "study-overlay" });
  const box = el("div", { class: "study-modal" });
  box.appendChild(el("h3", { text: st("beforeTraining") }));
  const list = el("div");
  interventions.forEach(iv => {
    studyLog("guidance_shown", { key: iv.key, mode: iv.mode, level: iv.level, stage: "pre_train", blocking: true, new: iv.new });
    renderIntervention(iv, list);
  });
  box.appendChild(list);
  const actions = el("div", { class: "study-actions" });
  actions.appendChild(el("button", {
    class: "neutral-button", type: "button", text: st("review"),
    onclick: () => {
      studyLog("train_blocked", { keys: interventions.map(i => i.key) });
      overlay.remove();
      if (typeof setWizardStep === "function") setWizardStep(2);
    },
  }));
  actions.appendChild(el("button", {
    class: "primary-action-button", type: "button", text: st("trainAnyway"),
    onclick: () => {
      studyLog("train_override", { keys: interventions.map(i => i.key) });
      overlay.remove();
      proceed();
    },
  }));
  box.appendChild(actions);
  overlay.appendChild(box);
  document.body.appendChild(overlay);
}

// ── Task brief and submission ────────────────────────────────────────────────

function buildTaskBrief() {
  const dict = STUDY.dictionary;
  const brief = el("div", { id: "study-brief", class: "card" });
  if (dict) {
    const lang = STUDY.lang in (dict.task || {}) ? STUDY.lang : "en";
    brief.appendChild(el("div", { class: "study-card-title", text: st("task") }));
    brief.appendChild(el("p", { text: dict.task[lang] }));
    brief.appendChild(el("p", { text: `${st("moment")}: ${dict.prediction_moment[lang]}` }));
    const table = el("table", { class: "stat-table study-hidden" });
    table.appendChild(el("tr", {}, [el("th", { text: st("column") }), el("th", { text: st("description") })]));
    Object.entries(dict.columns).forEach(([col, desc]) => {
      table.appendChild(el("tr", {}, [el("td", { text: col }), el("td", { text: desc[lang] || desc.en })]));
    });
    const toggle = el("button", {
      class: "study-link", type: "button", text: st("showDictionary"),
      onclick: () => {
        const hidden = table.classList.toggle("study-hidden");
        toggle.textContent = hidden ? st("showDictionary") : st("hideDictionary");
        studyLog("dictionary_toggle", { open: !hidden });
      },
    });
    brief.appendChild(toggle);
    brief.appendChild(table);
  }
  brief.appendChild(el("button", { class: "primary-action-button study-finish", type: "button", text: st("finish"), onclick: openSubmitModal }));
  const main = document.getElementById("training-section");
  if (main) main.prepend(brief);
}

function openSubmitModal() {
  if (STUDY.submitted) return;
  studyLog("submit_open", { has_results: STUDY.hasResults });
  if (!STUDY.hasResults) { showAlert(st("needTraining"), "warning"); return; }
  const overlay = el("div", { class: "study-overlay" });
  const box = el("div", { class: "study-modal" });
  box.appendChild(el("h3", { text: st("submitTitle") }));
  box.appendChild(el("p", { text: st("deployQ") }));
  const deploy = el("div", { class: "study-options" });
  ["yes", "no"].forEach(v => {
    const id = `study-deploy-${v}`;
    deploy.appendChild(el("label", { for: id }, [el("input", { type: "radio", name: "study-deploy", id, value: v }), document.createTextNode(` ${st(v)}`)]));
  });
  box.appendChild(deploy);
  box.appendChild(el("p", { text: st("confidenceQ") }));
  const conf = el("input", { type: "range", min: "1", max: "7", step: "1", value: "4" });
  const confValue = el("span", { text: "4" });
  conf.addEventListener("input", () => (confValue.textContent = conf.value));
  box.appendChild(el("div", { class: "study-options" }, [conf, confValue]));
  box.appendChild(el("p", { text: st("issuesQ") }));
  const issues = el("textarea", { rows: "5", class: "study-textarea", placeholder: st("issuesPlaceholder") });
  box.appendChild(issues);
  const actions = el("div", { class: "study-actions" });
  actions.appendChild(el("button", { class: "neutral-button", type: "button", text: st("cancel"), onclick: () => overlay.remove() }));
  actions.appendChild(el("button", {
    class: "primary-action-button", type: "button", text: st("submit"),
    onclick: async () => {
      const choice = box.querySelector('input[name="study-deploy"]:checked');
      if (!choice) { showAlert(st("needDeploy"), "warning"); return; }
      if (issues.value.trim().length < 2) { showAlert(st("needIssues"), "warning"); return; }
      studyLog("task_submit", {
        deploy: choice.value === "yes", confidence: parseInt(conf.value, 10),
        issues_text: issues.value.trim(), final_state: currentUiState(),
      });
      STUDY.submitted = true;
      await studyFlush();
      overlay.remove();
      // Nothing after submission should be mixed into the task data
      document.getElementById("study-guidance")?.remove();
      document.querySelector("#study-brief .study-finish")?.setAttribute("disabled", "disabled");
      showAlert(st("submitted"), "success");
    },
  }));
  box.appendChild(actions);
  overlay.appendChild(box);
  document.body.appendChild(overlay);
}

// ── Instrumentation of the existing app ──────────────────────────────────────

function wrapGlobal(name, makeWrapper) {
  const original = window[name];
  if (typeof original !== "function") return;
  window[name] = makeWrapper(original);
}

function instrumentApp() {
  wrapGlobal("setWizardStep", orig => function (step) {
    const before = wizardCurrentStep;
    const result = orig.apply(this, arguments);
    if (wizardCurrentStep !== before) studyLog("step_view", { step: wizardCurrentStep });
    return result;
  });
  wrapGlobal("toggleAdvancedOptions", orig => function () {
    const result = orig.apply(this, arguments);
    studyLog("advanced_toggle", { open: document.getElementById("advanced-options-body").style.display !== "none" });
    return result;
  });
  wrapGlobal("generateShapPlot", orig => function () {
    studyLog("shap_request");
    return orig.apply(this, arguments);
  });
  wrapGlobal("downloadPDF", orig => function () {
    studyLog("download", { what: "pdf" });
    return orig.apply(this, arguments);
  });
  wrapGlobal("downloadAllPlots", orig => function () {
    studyLog("download", { what: "plots" });
    return orig.apply(this, arguments);
  });
  wrapGlobal("sendChat", orig => async function () {
    const text = document.getElementById("chat-input")?.value.trim();
    if (text) {
      studyLog("chat_send", { text });
      // The adaptive assistant conditions on the profile, so send this message's evidence first
      await studyFlush();
    }
    return orig.apply(this, arguments);
  });
  wrapGlobal("toggleChatSidebar", orig => function () {
    const result = orig.apply(this, arguments);
    studyLog("chat_toggle", { open: !document.getElementById("chat-sidebar").classList.contains("collapsed") });
    return result;
  });

  wrapGlobal("startTraining", orig => async function () {
    const args = arguments;
    const fileReady = document.getElementById("upload-csv")?.files?.length > 0;
    const state = currentUiState();
    if (!fileReady || !state.target) return orig.apply(this, args);
    const proceed = () => {
      studyLog("train_start", { config: state });
      studyFlush();
      orig.apply(this, args);
    };
    if (STUDY.condition === "baseline") { proceed(); return; }
    const interventions = await requestGuidance("pre_train");
    const blocking = interventions.filter(iv => iv.blocking);
    // Blocking items live only in the modal, so the panel never duplicates them
    blocking.forEach(iv => document.querySelector(`#study-guidance [data-key="${CSS.escape(iv.key)}"]`)?.remove());
    showGuidance(interventions.filter(iv => !iv.blocking), "pre_train");
    if (blocking.length) showBlockingModal(blocking, proceed);
    else proceed();
  });

  wrapGlobal("renderTrainingResults", orig => function (data) {
    const result = orig.apply(this, arguments);
    STUDY.hasResults = true;
    const metrics = {};
    Object.entries(data.metrics || {}).forEach(([k, v]) => {
      if (v && typeof v.value === "number") metrics[k] = v.value;
    });
    studyLog("results_view", { metrics, best_model: data.best_model, feature_importance: data.feature_importance || {} });
    if (STUDY.condition !== "baseline") {
      refreshGuidance("post_train", { feature_importance: data.feature_importance || {}, metrics });
    }
    return result;
  });

  // Delegated listeners for controls that script.js rebuilds dynamically
  document.addEventListener("change", (e) => {
    const t = e.target;
    if (t.closest("#drop-columns-list")) studyLog("column_drop_toggle", { column: t.value, dropped: t.checked });
    else if (t.id === "target-column") studyLog("target_select", { column: t.value });
    else if (t.id === "adv-eval-metric") studyLog("metric_change", { metric: t.value });
    else if (t.id === "adv-use-cv") studyLog("cv_toggle", { use_cv: t.checked });
    else if (t.id === "adv-val-size") studyLog("val_size_change", { value: t.value });
    else if (t.closest("#adv-models-container")) studyLog("model_toggle", { model: t.value, on: t.checked });
    else if (t.id === "training-time-limit") studyLog("time_limit_change", { minutes: t.value });
    else if (t.id === "enable-imputation") studyLog("imputation_toggle", { enabled: t.checked });
  }, true);

  let lastHover = { key: null, ts: 0 };
  document.addEventListener("mouseover", (e) => {
    const icon = e.target.closest?.(".help-icon");
    if (!icon) return;
    const key = icon.getAttribute("data-i18n") || "help";
    const now = Date.now();
    if (key === lastHover.key && now - lastHover.ts < 2000) return;
    lastHover = { key, ts: now };
    studyLog("tooltip_hover", { key });
  });

  // Dwell time on rows of the descriptive statistics table
  let rowEnter = null;
  document.addEventListener("mouseover", (e) => {
    const row = e.target.closest?.("#stats-table tr");
    if (row && (!rowEnter || rowEnter.row !== row)) rowEnter = { row, ts: Date.now() };
  });
  document.addEventListener("mouseout", (e) => {
    if (!rowEnter || e.target.closest?.("#stats-table tr") !== rowEnter.row) return;
    if (rowEnter.row.contains(e.relatedTarget)) return;
    const dwell = Date.now() - rowEnter.ts;
    const column = rowEnter.row.querySelector("td")?.textContent?.trim();
    if (column && dwell > 800) studyLog("inspect_column", { column, dwell_ms: dwell });
    rowEnter = null;
  });

  document.getElementById("upload-csv")?.addEventListener("change", async function () {
    const file = this.files[0];
    if (!file || !STUDY.sessionId) return;
    studyLog("dataset_loaded", { name: file.name, size: file.size });
    const form = new FormData();
    form.append("session_id", STUDY.sessionId);
    form.append("file", file);
    try {
      await fetch("/study/dataset", { method: "POST", body: form });
    } catch (_) { /* profiling is best effort */ }
    // Let script.js finish building the target selector before reading it
    setTimeout(() => {
      const target = document.getElementById("target-column")?.value;
      if (target) studyLog("target_select", { column: target, initial: true });
      if (STUDY.condition !== "baseline") refreshGuidance("data");
    }, 500);
  });
}

async function initStudy() {
  if (!STUDY) return;
  if (typeof changeLanguage === "function") changeLanguage(STUDY.lang);
  try {
    const res = await fetch("/study/session", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        participant_id: STUDY.participantId, condition: STUDY.condition,
        dataset_key: STUDY.datasetKey, lang: STUDY.lang,
      }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    STUDY.sessionId = data.session_id;
    STUDY.dictionary = data.dictionary;
  } catch (err) {
    console.error("Study session could not be created", err);
    return;
  }
  document.body.classList.add("study-mode");
  const panel = el("div", { id: "study-guidance" }, [
    el("div", { class: "study-card-title", text: st("guidance") }),
    el("div", { class: "study-guidance-list" }),
  ]);
  panel.style.display = "none";
  document.body.appendChild(panel);
  buildTaskBrief();
  instrumentApp();
  studyLog("session_start", { condition: STUDY.condition, dataset_key: STUDY.datasetKey, viewport: [window.innerWidth, window.innerHeight] });
  studyLog("step_view", { step: wizardCurrentStep });
  setInterval(studyFlush, 3000);
  window.addEventListener("beforeunload", () => studyFlush(true));
}

if (STUDY) window.addEventListener("DOMContentLoaded", initStudy);
