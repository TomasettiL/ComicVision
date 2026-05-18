// ── State ─────────────────────────────────────────────────────────────────────
let filters = {};       // { name: metadata } from /filters
let videoEnabled = false;
let uploadedFile = null;
let currentJobId = null;
let pollTimer = null;
let layerCounter = 0;   // ever-increasing id for unique param namespacing

// Accent colours cycled per layer so cards are visually distinct
const LAYER_COLORS = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b'];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const uploadArea      = $('uploadArea');
const fileInput       = $('fileInput');
const fileInfo        = $('fileInfo');
const fileNameEl      = $('fileName');
const clearFileBtn    = $('clearFile');
const layerListEl     = $('layerList');
const addLayerBtn     = $('addLayerBtn');
const processBtn      = $('processBtn');
const processBtnLabel = $('processBtnLabel');
const processBtnIcon  = $('processBtnIcon');
const uploadView      = $('uploadView');
const resultView      = $('resultView');
const uploadNewBtn    = $('uploadNewBtn');
const previewArea     = $('previewArea');
const imageResult     = $('imageResult');
const processedImg    = $('processedImg');
const videoCompare    = $('videoCompare');
const originalVideo   = $('originalVideo');
const processedVideo  = $('processedVideo');
const videoPlaceholder = $('videoPlaceholder');
const progressWrap    = $('progressWrap');
const progressFill    = $('progressFill');
const progressText    = $('progressText');
const actionBar       = $('actionBar');
const actionNote      = $('actionNote');
const downloadBtn     = $('downloadBtn');
const convertFpsEl    = $('convertFps');

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function init() {
  try {
    const [filtersRes, configRes] = await Promise.all([fetch('/filters'), fetch('/config')]);
    filters = await filtersRes.json();
    const config = await configRes.json();
    videoEnabled = !!config.video_enabled;
  } catch (e) {
    console.error('Could not load config:', e);
    return;
  }

  // Restrict file picker and label when video is disabled
  if (!videoEnabled) {
    fileInput.accept = 'image/*';
    document.querySelector('.upload-sub').textContent = 'PNG, JPG, WEBP & more';
  }

  setupUpload();
  setupTooltip();

  addLayerBtn.addEventListener('click', () => addLayer());

  // Start with one Canny layer pre-loaded
  addLayer(Object.keys(filters)[0]);
}

// ── Tooltip engine ────────────────────────────────────────────────────────────
function setupTooltip() {
  const tip = document.createElement('div');
  tip.id = 'cv-tooltip';
  document.body.appendChild(tip);
  let anchor = null;

  function show(el) {
    tip.textContent = el.dataset.tip;
    tip.style.opacity = '1';
    anchor = el;
    reposition();
  }
  function hide() { tip.style.opacity = '0'; anchor = null; }
  function reposition() {
    if (!anchor) return;
    const r = anchor.getBoundingClientRect();
    const w = 230, m = 10;
    let left = r.right + m;
    if (left + w > window.innerWidth - m) left = r.left - w - m;
    let top = Math.max(m, Math.min(r.top + r.height / 2 - 24, window.innerHeight - 120));
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top  = `${Math.round(top)}px`;
  }

  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (el) show(el); else hide();
  });
  document.addEventListener('mouseout', e => {
    if (!e.relatedTarget?.closest('[data-tip]')) hide();
  });
  document.addEventListener('scroll', reposition, true);
}

// ── Layer management ──────────────────────────────────────────────────────────
function addLayer(filterName = null) {
  const id = layerCounter++;
  const name = filterName ?? Object.keys(filters)[0];
  const color = LAYER_COLORS[id % LAYER_COLORS.length];
  const cardIndex = layerListEl.children.length + 1;

  const card = document.createElement('div');
  card.className = 'layer-card';
  card.dataset.layerId = id;
  card.style.setProperty('--layer-color', color);

  // ── Header ────────────────────────────────────────────────────────────────
  const header = document.createElement('div');
  header.className = 'layer-header';

  const badge = document.createElement('span');
  badge.className = 'layer-badge';
  badge.textContent = cardIndex;

  const select = document.createElement('select');
  select.className = 'layer-filter-select';
  for (const [fname, meta] of Object.entries(filters)) {
    const opt = document.createElement('option');
    opt.value = fname;
    opt.textContent = meta.label;
    if (fname === name) opt.selected = true;
    if (meta.tab_tip) opt.title = meta.tab_tip;
    select.appendChild(opt);
  }

  const typeTag = document.createElement('span');
  function updateTypeTag(filterName) {
    const t = filters[filterName]?.output_type ?? 'edge';
    typeTag.className = `layer-type-tag ${t}`;
    typeTag.textContent = t;
  }
  updateTypeTag(name);

  const removeBtn = document.createElement('button');
  removeBtn.className = 'btn-remove-layer';
  removeBtn.title = 'Remove layer';
  removeBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>`;
  removeBtn.addEventListener('click', () => {
    card.remove();
    renumberLayers();
    updateProcessBtn();
  });

  header.append(badge, select, typeTag, removeBtn);

  // ── Params ────────────────────────────────────────────────────────────────
  const paramsDiv = document.createElement('div');
  paramsDiv.className = 'layer-params';

  card.append(header, paramsDiv);
  layerListEl.appendChild(card);

  renderLayerParams(paramsDiv, name, id);

  select.addEventListener('change', () => {
    updateTypeTag(select.value);
    renderLayerParams(paramsDiv, select.value, id);
  });

  updateProcessBtn();
}

function renumberLayers() {
  layerListEl.querySelectorAll('.layer-badge').forEach((badge, i) => {
    badge.textContent = i + 1;
  });
}

// ── Parameter rendering ───────────────────────────────────────────────────────
function renderLayerParams(container, filterName, layerId) {
  container.innerHTML = '';
  const params = filters[filterName]?.params ?? [];
  const ns = `L${layerId}`; // namespace prefix so IDs never collide across cards

  if (!params.length) {
    container.innerHTML = '<p style="font-size:11px;color:var(--muted)">No parameters</p>';
    return;
  }

  for (const p of params) {
    const group = document.createElement('div');
    group.className = 'param-group';
    const tipAttr = p.tip ? `data-tip="${p.tip.replace(/"/g, '&quot;')}"` : '';
    const tipIcon = p.tip ? `<span class="info-tip" ${tipAttr}>i</span>` : '';

    if (p.type === 'range') {
      group.innerHTML = `
        <div class="param-row">
          <span style="display:flex;align-items:center;gap:5px;">
            <span class="param-label-text">${p.label}</span>${tipIcon}
          </span>
          <span class="param-value" data-val="${ns}-${p.name}">${p.default}</span>
        </div>
        <input type="range" data-param="${ns}-${p.name}"
          min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">`;
      container.appendChild(group);
      const input = group.querySelector('input');
      const valEl = group.querySelector(`[data-val="${ns}-${p.name}"]`);
      input.addEventListener('input', () => { valEl.textContent = Number(input.value); });

    } else if (p.type === 'select') {
      const opts = p.options
        .map(o => `<option value="${o}" ${o === p.default ? 'selected' : ''}>${o}</option>`)
        .join('');
      group.innerHTML = `
        <div class="param-row">
          <span style="display:flex;align-items:center;gap:5px;">
            <span class="param-label-text">${p.label}</span>${tipIcon}
          </span>
        </div>
        <select class="param-select" data-param="${ns}-${p.name}">${opts}</select>`;
      container.appendChild(group);

    } else if (p.type === 'checkbox') {
      group.innerHTML = `
        <div style="display:flex;align-items:center;gap:7px;">
          <label class="param-checkbox" style="flex:1;">
            <input type="checkbox" data-param="${ns}-${p.name}" ${p.default ? 'checked' : ''}>
            <span>${p.label}</span>
          </label>${tipIcon}
        </div>`;
      container.appendChild(group);
    }
  }
}

// ── Collect all layers into the JSON the server expects ───────────────────────
function collectLayers() {
  const layers = [];
  layerListEl.querySelectorAll('.layer-card').forEach(card => {
    const layerId = card.dataset.layerId;
    const filterName = card.querySelector('.layer-filter-select').value;
    const ns = `L${layerId}`;
    const params = {};
    for (const p of (filters[filterName]?.params ?? [])) {
      const el = card.querySelector(`[data-param="${ns}-${p.name}"]`);
      if (!el) continue;
      params[p.name] = (p.type === 'checkbox') ? (el.checked ? 'true' : 'false') : el.value;
    }
    layers.push({ filter: filterName, params });
  });
  return layers;
}

// ── Upload ────────────────────────────────────────────────────────────────────
function setupUpload() {
  uploadArea.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  clearFileBtn.addEventListener('click', clearUpload);
  uploadNewBtn.addEventListener('click', clearUpload);
  processBtn.addEventListener('click', processFile);
}

function handleFile(file) {
  if (!videoEnabled && file.type.startsWith('video/')) {
    alert('Video processing is not available. Please upload an image.');
    return;
  }
  uploadedFile = file;
  fileNameEl.textContent = file.name;
  fileInfo.hidden = false;
  uploadArea.hidden = true;
  updateProcessBtn();
}

function clearUpload() {
  stopPolling();
  uploadedFile = null;
  fileInput.value = '';
  fileInfo.hidden = true;
  uploadArea.hidden = false;
  // Switch back to upload view
  resultView.hidden = true;
  uploadView.hidden = false;
  // Reset result view internals
  actionBar.hidden = true;
  progressWrap.hidden = true;
  imageResult.hidden = true;
  videoCompare.hidden = true;
  setBusy(false);
  updateProcessBtn();
}

function updateProcessBtn() {
  const hasFile = !!uploadedFile;
  const hasLayers = layerListEl.querySelectorAll('.layer-card').length > 0;
  processBtn.disabled = !hasFile || !hasLayers;
}

// ── Processing ────────────────────────────────────────────────────────────────
async function processFile() {
  if (!uploadedFile) return;
  const layers = collectLayers();
  if (!layers.length) return;
  stopPolling();

  // Switch to result view
  uploadView.hidden = true;
  resultView.hidden = false;

  const isVideo = uploadedFile.type.startsWith('video/');
  imageResult.hidden = true;
  videoCompare.hidden = true;
  actionBar.hidden = true;

  if (isVideo) {
    videoCompare.hidden = false;
    videoPlaceholder.hidden = false;
    processedVideo.hidden = true;
    originalVideo.src = URL.createObjectURL(uploadedFile);
  }

  const formData = new FormData();
  formData.append('file', uploadedFile);
  formData.append('layers', JSON.stringify(layers));
  formData.append('convert_fps', convertFpsEl.checked ? 'true' : 'false');

  setBusy(true);
  progressWrap.hidden = false;
  progressText.className = 'progress-text';
  progressText.textContent = 'Uploading…';
  progressFill.style.width = '0%';
  actionBar.hidden = true;

  try {
    const res = await fetch('/process', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    currentJobId = data.job_id;
    progressText.textContent = 'Processing…';
    pollTimer = setInterval(pollStatus, 400);
  } catch (err) {
    showError(err.message);
    setBusy(false);
  }
}

async function pollStatus() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/status/${currentJobId}`);
    const job = await res.json();
    progressFill.style.width = `${job.progress}%`;
    progressText.textContent = `Processing… ${job.progress}%`;
    if (job.status === 'done')  { stopPolling(); showResult(currentJobId, job.is_video); }
    if (job.status === 'error') { stopPolling(); showError(job.error || 'Processing failed'); setBusy(false); }
  } catch (_) {}
}

function showResult(jobId, isVideo) {
  const url = `/result/${jobId}`;
  progressWrap.hidden = true;
  actionBar.hidden = false;
  downloadBtn.href = url;
  downloadBtn.download = isVideo ? 'comic_processed.mp4' : 'comic_processed.png';
  setBusy(false);

  if (isVideo) {
    videoPlaceholder.hidden = true;
    processedVideo.hidden = false;
    processedVideo.src = url;
    processedVideo.load();
    actionNote.textContent = 'If the video does not play, use the download button.';
  } else {
    imageResult.hidden = false;
    processedImg.src = url;
    actionNote.textContent = '';
  }
}

function showError(msg) {
  progressText.className = 'progress-text error';
  progressText.textContent = `Error: ${msg}`;
  progressWrap.hidden = false;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setBusy(busy) {
  processBtn.classList.toggle('busy', busy);
  processBtn.disabled = busy;
  processBtnLabel.textContent = busy ? 'Processing…' : 'Process';
  processBtnIcon.innerHTML = busy
    ? '<circle cx="12" cy="12" r="9" stroke-dasharray="56" stroke-dashoffset="14" stroke-linecap="round" fill="none"/>'
    : '<polygon points="5 3 19 12 5 21 5 3"/>';
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

// ── Go ────────────────────────────────────────────────────────────────────────
init();
