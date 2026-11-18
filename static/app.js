"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_UPLOAD_BYTES = 150 * 1024 * 1024;
const POLL_MS = 500;
const POLL_RETRY_MS = 2000;
const TERMINAL = ["done", "failed"];
const PRESETS = {
  soft: { w: 400, l: 40 },
  liver: { w: 150, l: 30 },
  lung: { w: 1500, l: -600 },
  bone: { w: 1800, l: 400 },
};
// Model structures get filled colours (assigned per name, in order of first sight).
const FILL_COLOURS = [[230, 57, 70], [58, 134, 255], [255, 183, 3], [199, 125, 255], [251, 133, 0], [0, 187, 249]];
const REFERENCE_COLOUR = [46, 204, 113]; // green outline

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  series: null,          // metadata from POST /api/series
  volume: null,          // Int16Array of HU, laid out (slice, row, column)
  slice: 0,
  window: { ...PRESETS.soft },
  jobs: [],              // latest GET /api/jobs for the current series
  defaultJobId: null,    // most recently started finished job (decided by the server)
  pickedJobId: null,     // set when the user clicks a job; overrides the default
  masks: new Map(),      // "jobId/name" -> Uint8Array packed bits | "loading" | Error. Keyed by job id.
  diceOverrides: new Map(), // jobId -> dice result for a structure the user picked
  hidden: new Set(),     // structure names the user switched off
  colours: new Map(),    // structure name -> colour
  reference: null,       // Uint8Array packed bits on the current series grid
  showReference: true,
  opacity: 0.45,
};
let seriesLoadId = 0;    // ignore responses for a file the user has since replaced
let pollTimer = null;
let lastJobsSnapshot = "";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
const sliceCanvas = document.createElement("canvas"); // one slice at native resolution
const sliceCtx = sliceCanvas.getContext("2d");

// ---------------------------------------------------------------------------
// Fetch helpers: every request has a timeout and friendly errors
// ---------------------------------------------------------------------------

class UserError extends Error {
  constructor(message, detail = "") {
    super(message);
    this.detail = detail;
  }
}

async function api(url, { method = "GET", body = null, headers = {}, timeoutMs = 30000, as = "json", onProgress = null } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { method, body, headers, signal: controller.signal });
    if (!resp.ok) throw await errorFromResponse(resp);
    if (as === "bytes") return await readBytes(resp, onProgress);
    return await resp.json();
  } catch (err) {
    if (err instanceof UserError) throw err;
    if (err.name === "AbortError") throw new UserError("The local server took too long to respond. Please try again.");
    if (err instanceof SyntaxError) throw new UserError("The local server sent a response the page couldn't read.", String(err));
    throw new UserError("Couldn't reach the local server. Is ./run.sh still running?", String(err));
  } finally {
    clearTimeout(timer);
  }
}

async function errorFromResponse(resp) {
  let message = `The local server returned an error (HTTP ${resp.status}).`;
  let detail = "";
  try {
    const data = await resp.json();
    if (data && data.error && typeof data.error.message === "string") {
      message = data.error.message;
      detail = String(data.error.detail || "");
    }
  } catch (_) { /* not JSON: keep the generic message */ }
  return new UserError(message, detail);
}

async function readBytes(resp, onProgress) {
  const total = Number(resp.headers.get("Content-Length")) || 0;
  if (!resp.body || !onProgress) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(received / total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Errors and status text (always textContent, never innerHTML)
// ---------------------------------------------------------------------------

function showError(err) {
  const message = err instanceof Error ? err.message : String(err);
  const detail = err instanceof UserError ? err.detail : "";
  $("error-message").textContent = message;
  $("error-details").textContent = detail;
  $("error-details-wrap").hidden = !detail;
  $("error-area").hidden = false;
}

function clearError() {
  $("error-area").hidden = true;
}

function el(tag, text = "", className = "") {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// ---------------------------------------------------------------------------
// Loading a scan
// ---------------------------------------------------------------------------

async function looksLikeZip(file) {
  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  return head[0] === 0x50 && head[1] === 0x4b && head[2] === 0x03 && head[3] === 0x04; // "PK\x03\x04"
}

async function onFileChosen(file) {
  clearError();
  if (!file) return;
  if (file.size === 0) return showError("The file is empty.");
  if (file.size > MAX_UPLOAD_BYTES) return showError("The file is larger than the 150 MB limit.");
  if (!(await looksLikeZip(file))) return showError("This doesn't look like a zip file. Please select the DICOM zip.");

  const loadId = ++seriesLoadId;
  const progress = $("volume-progress");
  $("series-status").textContent = `Reading ${file.name}…`;
  try {
    const meta = await api("/api/series", {
      method: "POST", body: file, headers: { "Content-Type": "application/zip" }, timeoutMs: 180000,
    });
    if (loadId !== seriesLoadId) return;
    $("series-status").textContent = "Loading image data…";
    progress.value = 0;
    progress.hidden = false;
    const bytes = await api(`/api/series/${meta.id}/volume`, {
      as: "bytes", timeoutMs: 180000, onProgress: (p) => { progress.value = p; },
    });
    if (loadId !== seriesLoadId) return;
    const { slices, rows, columns } = meta.shape;
    const count = slices * rows * columns;
    if (bytes.byteLength !== count * 2) throw new UserError("The image data arrived incomplete. Please try again.");
    // Int16Array uses the platform byte order; the server sends little-endian, as all browsers' platforms are.
    setSeries(meta, new Int16Array(bytes.buffer, 0, count));
    $("series-status").textContent =
      `${file.name}: ${slices} slices of ${columns} × ${rows}` + (meta.warnings.length ? ` (${meta.warnings.join(" ")})` : "");
    await loadReference(meta);
  } catch (err) {
    if (loadId !== seriesLoadId) return;
    $("series-status").textContent = state.series ? "Still showing the previous scan." : "";
    showError(err);
  } finally {
    if (loadId === seriesLoadId) progress.hidden = true;
  }
}

function setSeries(meta, volume) {
  state.series = meta;
  state.volume = volume;
  state.slice = Math.floor(meta.shape.slices / 2);
  state.jobs = [];
  lastJobsSnapshot = "";
  state.defaultJobId = null;
  state.pickedJobId = null;
  state.masks.clear();
  state.diceOverrides.clear();
  state.reference = null;
  const slider = $("slice-slider");
  slider.max = String(meta.shape.slices - 1);
  slider.disabled = false;
  $("run-button").disabled = false;
  $("placeholder").hidden = true;
  for (const side of ["top", "bottom", "left", "right"]) $(`orient-${side}`).textContent = meta.labels[side];
  renderAll();
}

async function loadReference(meta) {
  if (!meta.reference.available) {
    $("reference-status").textContent = meta.reference.message;
    return;
  }
  $("reference-status").textContent = "Loading reference…";
  try {
    const bits = await api(`/api/series/${meta.id}/reference`, { as: "bytes", timeoutMs: 60000 });
    if (state.series !== meta) return;
    state.reference = bits;
    $("reference-status").textContent = "";
  } catch (err) {
    $("reference-status").textContent = err.message;
  }
  render();
}

// ---------------------------------------------------------------------------
// Jobs: start, poll, select
// ---------------------------------------------------------------------------

async function onRun() {
  if (!state.series) return;
  clearError();
  try {
    await api(`/api/series/${state.series.id}/jobs`, { method: "POST", timeoutMs: 15000 });
  } catch (err) {
    showError(err);
  }
  schedulePoll(0);
}

function schedulePoll(delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, delay);
}

async function poll() {
  if (!state.series) return;
  const seriesId = state.series.id;
  try {
    const data = await api(`/api/jobs?series_id=${encodeURIComponent(seriesId)}`, { timeoutMs: 10000 });
    if (!state.series || state.series.id !== seriesId) return;
    $("poll-status").textContent = "";
    const snapshot = JSON.stringify(data);
    if (snapshot !== lastJobsSnapshot) { // re-render only on change, so clicks aren't lost mid-rebuild
      lastJobsSnapshot = snapshot;
      state.jobs = data.jobs;
      state.defaultJobId = data.default_job_id;
      forgetMissingJobs();
      renderAll();
      const job = selectedJob();
      if (job && job.status === "done") loadMasks(job);
    }
  } catch (err) {
    $("poll-status").textContent = `Lost contact with the local server, retrying… (${err.message})`;
    schedulePoll(POLL_RETRY_MS);
    return;
  }
  if (state.jobs.some((j) => !TERMINAL.includes(j.status))) schedulePoll(POLL_MS);
}

function forgetMissingJobs() {
  const ids = new Set(state.jobs.map((j) => j.id));
  for (const key of state.masks.keys()) if (!ids.has(key.split("/")[0])) state.masks.delete(key);
  if (state.pickedJobId && !ids.has(state.pickedJobId)) state.pickedJobId = null;
}

function selectedJob() {
  const id = state.pickedJobId || state.defaultJobId;
  return state.jobs.find((j) => j.id === id) || null;
}

function pickJob(jobId) {
  state.pickedJobId = jobId;
  renderAll();
  const job = selectedJob();
  if (job && job.status === "done") loadMasks(job);
}

async function loadMasks(job) {
  for (const structure of job.structures) {
    if (!structure.has_overlay) continue;
    const key = `${job.id}/${structure.name}`;
    if (state.masks.has(key)) continue;
    state.masks.set(key, "loading");
    try {
      const url = `/api/jobs/${encodeURIComponent(job.id)}/masks/${encodeURIComponent(structure.name)}`;
      state.masks.set(key, await api(url, { as: "bytes", timeoutMs: 60000 }));
    } catch (err) {
      state.masks.set(key, err);
    }
    renderAll();
  }
}

async function onDiceStructureChosen(job, name) {
  try {
    const url = `/api/jobs/${encodeURIComponent(job.id)}/dice?structure=${encodeURIComponent(name)}`;
    state.diceOverrides.set(job.id, await api(url, { timeoutMs: 30000 }));
  } catch (err) {
    showError(err);
  }
  renderDice();
}

// ---------------------------------------------------------------------------
// Side panel rendering
// ---------------------------------------------------------------------------

function renderAll() {
  renderJobs();
  renderStructureToggles();
  renderDice();
  render();
}

function renderJobs() {
  const list = $("job-list");
  list.replaceChildren();
  const shown = selectedJob();
  for (const job of [...state.jobs].reverse()) {
    const item = el("li", "", job.status);
    if (shown && job.id === shown.id) item.classList.add("selected");
    const title = el("div", `Job #${job.number} · `, "job-title");
    title.append(el("span", job.status, "job-status"));
    if (job.id === state.defaultJobId && !state.pickedJobId) title.append(el("span", " (shown)"));
    item.append(title, el("div", job.message, "job-message"));
    if (job.error && job.error.detail) {
      const details = el("details");
      details.append(el("summary", "Technical details"), el("pre", job.error.detail));
      item.append(details);
    }
    item.addEventListener("click", () => pickJob(job.id));
    list.append(item);
  }
}

function colourFor(name) {
  if (!state.colours.has(name)) state.colours.set(name, FILL_COLOURS[state.colours.size % FILL_COLOURS.length]);
  return state.colours.get(name);
}

function renderStructureToggles() {
  const box = $("structure-toggles");
  const legend = $("legend");
  box.replaceChildren();
  legend.replaceChildren();
  const job = selectedJob();
  if (!job) {
    box.append(el("div", "No job selected.", "status"));
  } else if (job.status !== "done") {
    box.append(el("div", `Job #${job.number}: ${job.message}`, "status"));
  } else {
    for (const structure of job.structures) {
      if (!structure.has_overlay) {
        box.append(el("div", `${structure.name}: ${structure.message}`, "structure-failed"));
        continue;
      }
      const label = el("label", "", "toggle");
      const input = el("input");
      input.type = "checkbox";
      input.checked = !state.hidden.has(structure.name);
      input.addEventListener("change", () => {
        if (input.checked) state.hidden.delete(structure.name); else state.hidden.add(structure.name);
        renderStructureToggles();
        render();
      });
      const mask = state.masks.get(`${job.id}/${structure.name}`);
      const suffix = mask === "loading" ? " (loading…)" : mask instanceof Error ? ` (${mask.message})` : "";
      label.append(input, swatch(colourFor(structure.name)), el("span", structure.name + suffix));
      box.append(label);
      for (const warning of structure.warnings) box.append(el("div", warning, "status"));
      if (input.checked) legend.append(legendEntry(swatch(colourFor(structure.name)), `${structure.name} (model)`));
    }
  }
  if (state.reference && state.showReference) {
    legend.append(legendEntry(el("span", "", "swatch reference"), "reference (outline)"));
  }
}

function swatch(rgb) {
  const node = el("span", "", "swatch");
  node.style.background = `rgb(${rgb.join(",")})`;
  return node;
}

function legendEntry(icon, text) {
  const node = el("span");
  node.append(icon, document.createTextNode(text));
  return node;
}

function renderDice() {
  const box = $("dice-box");
  box.replaceChildren();
  const job = selectedJob();
  if (!job) return box.append(el("div", "Run segmentation to compute Dice.", "status"));
  if (job.status === "failed") return box.append(el("div", "This job failed, so there is no Dice.", "status"));
  if (job.status !== "done" || !job.dice) return box.append(el("div", "Waiting for the job to finish…", "status"));

  const dice = state.diceOverrides.get(job.id) || job.dice;
  if (dice.dice !== null && dice.dice !== undefined) {
    box.append(el("div", dice.dice.toFixed(3), "dice-value"));
    const table = el("table");
    addRow(table, `Model (${dice.structure})`, `${dice.model_voxels.toLocaleString()} vox · ${dice.model_ml.toFixed(1)} mL`);
    addRow(table, "Reference", `${dice.reference_voxels.toLocaleString()} vox · ${dice.reference_ml.toFixed(1)} mL`);
    addRow(table, "Grid check", dice.grid_reason);
    box.append(table);
  } else {
    box.append(el("div", dice.message || "Dice isn't available.", "status"));
    if (dice.model_voxels !== undefined) {
      const table = el("table");
      addRow(table, "Model voxels", dice.model_voxels.toLocaleString());
      addRow(table, "Reference voxels", dice.reference_voxels.toLocaleString());
      box.append(table);
    }
  }
  const choices = dice.choices || [];
  if (choices.length > 1 || (choices.length === 1 && choices[0] !== dice.structure)) {
    const select = el("select");
    for (const name of choices) {
      const option = el("option", `Compare: ${name}`);
      option.value = name;
      option.selected = name === dice.structure;
      select.append(option);
    }
    if (!choices.includes(dice.structure)) {
      const option = el("option", "Choose a structure…");
      option.value = "";
      option.selected = true;
      option.disabled = true;
      select.prepend(option);
    }
    select.addEventListener("change", () => onDiceStructureChosen(job, select.value));
    box.append(select);
  }
}

function addRow(table, label, value) {
  const row = el("tr");
  row.append(el("td", label), el("td", value));
  table.append(row);
}

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------

// np.packbits order: voxel n is bit (7 - n % 8) of byte n / 8.
function bitAt(bits, index) {
  return (bits[index >> 3] >> (7 - (index & 7))) & 1;
}

function render() {
  resizeCanvas();
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.volume) return;

  const { slices, rows, columns } = state.series.shape;
  const k = state.slice;
  const offset = k * rows * columns; // first voxel of slice k in (slice, row, column) order
  if (sliceCanvas.width !== columns || sliceCanvas.height !== rows) {
    sliceCanvas.width = columns;
    sliceCanvas.height = rows;
  }
  const image = sliceCtx.createImageData(columns, rows);
  drawGreyscale(image.data, offset, rows * columns);
  drawModelFills(image.data, offset, rows * columns);
  if (state.showReference && state.reference) drawOutline(image.data, state.reference, offset, rows, columns, REFERENCE_COLOUR);
  sliceCtx.putImageData(image, 0, 0);

  // Fit to the viewport while keeping the physical aspect ratio from the pixel spacing.
  const widthMm = columns * state.series.spacing_mm.column;
  const heightMm = rows * state.series.spacing_mm.row;
  const scale = Math.min(canvas.width / widthMm, canvas.height / heightMm);
  const w = widthMm * scale;
  const h = heightMm * scale;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(sliceCanvas, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);

  const z = state.series.slice_positions_mm[k];
  $("slice-label").textContent = `Slice ${k + 1} / ${slices} · z = ${z.toFixed(1)} mm`;
  $("window-label").textContent = `W ${Math.round(state.window.w)} / L ${Math.round(state.window.l)}`;
  $("slice-slider").value = String(k);
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawGreyscale(pixels, offset, count) {
  const low = state.window.l - state.window.w / 2;
  const scale = 255 / state.window.w;
  for (let p = 0; p < count; p++) {
    const grey = Math.min(255, Math.max(0, (state.volume[offset + p] - low) * scale));
    pixels[4 * p] = pixels[4 * p + 1] = pixels[4 * p + 2] = grey;
    pixels[4 * p + 3] = 255;
  }
}

function drawModelFills(pixels, offset, count) {
  const job = selectedJob();
  if (!job || job.status !== "done") return;
  const alpha = state.opacity;
  for (const structure of job.structures) {
    const bits = state.masks.get(`${job.id}/${structure.name}`);
    if (!(bits instanceof Uint8Array) || state.hidden.has(structure.name)) continue;
    const colour = colourFor(structure.name);
    for (let p = 0; p < count; p++) {
      if (!bitAt(bits, offset + p)) continue;
      for (let c = 0; c < 3; c++) pixels[4 * p + c] = pixels[4 * p + c] * (1 - alpha) + colour[c] * alpha;
    }
  }
}

// A voxel is on the outline if it is inside and any 4-neighbour is outside (or off the image).
function drawOutline(pixels, bits, offset, rows, columns, colour) {
  const inside = (r, c) => r >= 0 && r < rows && c >= 0 && c < columns && bitAt(bits, offset + r * columns + c);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      if (!inside(r, c)) continue;
      if (inside(r - 1, c) && inside(r + 1, c) && inside(r, c - 1) && inside(r, c + 1)) continue;
      const p = r * columns + c;
      pixels[4 * p] = colour[0];
      pixels[4 * p + 1] = colour[1];
      pixels[4 * p + 2] = colour[2];
    }
  }
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

function setSlice(k) {
  if (!state.series) return;
  state.slice = Math.max(0, Math.min(state.series.shape.slices - 1, k));
  render();
}

function setupInteraction() {
  $("file-input").addEventListener("change", (event) => {
    const file = event.target.files[0];
    event.target.value = ""; // so choosing the same file again still triggers "change"
    onFileChosen(file).catch(showError);
  });
  $("run-button").addEventListener("click", onRun);
  $("error-dismiss").addEventListener("click", clearError);
  $("slice-slider").addEventListener("input", (e) => setSlice(Number(e.target.value)));
  $("opacity").addEventListener("input", (e) => { state.opacity = Number(e.target.value) / 100; render(); });
  $("reference-toggle").addEventListener("change", (e) => {
    state.showReference = e.target.checked;
    renderStructureToggles();
    render();
  });
  $("preset").addEventListener("change", (e) => {
    if (PRESETS[e.target.value]) state.window = { ...PRESETS[e.target.value] };
    render();
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    setSlice(state.slice + (e.deltaY > 0 ? 1 : -1));
  }, { passive: false });

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    const steps = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 10, PageDown: -10 };
    if (e.key in steps) {
      e.preventDefault();
      setSlice(state.slice + steps[e.key]);
    }
  });

  // Drag: horizontal changes window width, vertical changes level.
  let drag = null;
  canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, ...state.window }; });
  window.addEventListener("mouseup", () => { drag = null; });
  window.addEventListener("mousemove", (e) => {
    if (!drag || !state.volume) return;
    state.window = { w: Math.max(1, drag.w + 2 * (e.clientX - drag.x)), l: drag.l + 2 * (e.clientY - drag.y) };
    $("preset").value = "custom";
    render();
  });

  window.addEventListener("resize", render);
}

async function checkHealth() {
  const banner = $("config-banner");
  try {
    const health = await api("/api/health", { timeoutMs: 5000 });
    const notes = [];
    if (!health.token_configured) notes.push("API token not configured: set NUCLEO_API_TOKEN in .env and restart ./run.sh. You can still view scans.");
    if (!health.reference_available) notes.push(health.reference_message);
    banner.textContent = notes.join(" ");
    banner.hidden = notes.length === 0;
    if (!health.reference_available) $("reference-status").textContent = health.reference_message;
  } catch (err) {
    banner.textContent = err.message;
    banner.hidden = false;
  }
}

setupInteraction();
render();
checkHealth();
