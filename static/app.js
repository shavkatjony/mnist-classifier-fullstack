/* ═══════════════════════════════════════════════════════════════════════════
   MNIST Neural Network Explorer — Frontend
   Handles: drawing, prediction, layer visualization, training, metrics
═══════════════════════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  drawing: false,
  hasPixels: false,
  modelTrained: false,
  isTraining: false,
  predictTimer: null,
  lastPixels: null,           // Float32Array, 784 values [0-1]
};

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const drawCanvas  = document.getElementById('draw-canvas');
const pixelCanvas = document.getElementById('pixel-canvas');
const dctx        = drawCanvas.getContext('2d');
const pctx        = pixelCanvas.getContext('2d');

const canvasHint   = document.getElementById('canvas-hint');
const predDigit    = document.getElementById('pred-digit');
const predConf     = document.getElementById('pred-conf');
const predTime     = document.getElementById('pred-time');
const logScroll    = document.getElementById('log-scroll');
const epochTbody   = document.getElementById('epoch-tbody');
const progressBar  = document.getElementById('progress-bar');
const progLabel    = document.getElementById('prog-label');
const progStats    = document.getElementById('prog-stats');
const progDetails  = document.getElementById('progress-details');
const btnTrain     = document.getElementById('btn-train');
const btnStop      = document.getElementById('btn-stop');
const btnClear     = document.getElementById('btn-clear');
const btnSample    = document.getElementById('btn-sample');
const pillStatus   = document.getElementById('pill-status');
const pillParams   = document.getElementById('pill-params');
const pillDevice   = document.getElementById('pill-device');

// ─── Logging ──────────────────────────────────────────────────────────────────
function log(msg, type = '') {
  const now   = new Date().toLocaleTimeString('en', { hour12: false });
  const entry = document.createElement('div');
  entry.className = `log-entry ${type ? 'log-' + type : ''}`;
  entry.innerHTML = `<span class="log-time">[${now}]</span> ${msg}`;
  logScroll.appendChild(entry);
  logScroll.scrollTop = logScroll.scrollHeight;
  if (logScroll.children.length > 120) logScroll.children[0].remove();
}

// ─── Drawing Canvas ───────────────────────────────────────────────────────────
dctx.fillStyle = '#000';
dctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

function getPos(e) {
  const r = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / r.width;
  const scaleY = drawCanvas.height / r.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - r.left) * scaleX,
    y: (src.clientY - r.top)  * scaleY,
  };
}

function draw(e) {
  if (!state.drawing) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  dctx.globalCompositeOperation = 'source-over';
  dctx.fillStyle = '#fff';
  dctx.beginPath();
  dctx.arc(x, y, 9, 0, Math.PI * 2);
  dctx.fill();
  state.hasPixels = true;
  canvasHint.classList.add('hidden');
  schedulePrediction();
}

drawCanvas.addEventListener('mousedown',  e => { state.drawing = true; draw(e); });
drawCanvas.addEventListener('mousemove',  draw);
drawCanvas.addEventListener('mouseup',    () => state.drawing = false);
drawCanvas.addEventListener('mouseleave', () => state.drawing = false);
drawCanvas.addEventListener('touchstart', e => { state.drawing = true; draw(e); }, { passive: false });
drawCanvas.addEventListener('touchmove',  draw, { passive: false });
drawCanvas.addEventListener('touchend',   () => state.drawing = false);

btnClear.addEventListener('click', () => {
  dctx.fillStyle = '#000';
  dctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  state.hasPixels = false;
  state.lastPixels = null;
  canvasHint.classList.remove('hidden');
  clearPredictionUI();
  clearFlowUI();
  pctx.fillStyle = '#000';
  pctx.fillRect(0, 0, pixelCanvas.width, pixelCanvas.height);
  log('Canvas cleared.');
});

// ─── Extract 28×28 pixels from drawing canvas ─────────────────────────────────
function get28x28() {
  const tmp  = document.createElement('canvas');
  tmp.width  = 28;
  tmp.height = 28;
  const ctx  = tmp.getContext('2d');
  ctx.drawImage(drawCanvas, 0, 0, 28, 28);
  const imgData = ctx.getImageData(0, 0, 28, 28).data;
  const pixels  = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = imgData[i * 4] / 255.0;  // red channel (grayscale)
  }
  return pixels;
}

function renderPixelGrid(pixels) {
  const sz = pixelCanvas.width / 28;   // cell size in display pixels
  pctx.clearRect(0, 0, pixelCanvas.width, pixelCanvas.height);
  for (let row = 0; row < 28; row++) {
    for (let col = 0; col < 28; col++) {
      const v = pixels[row * 28 + col];
      const c = Math.round(v * 255);
      pctx.fillStyle = `rgb(${c},${c},${c})`;
      pctx.fillRect(col * sz, row * sz, sz, sz);
    }
  }
}

// ─── Prediction ───────────────────────────────────────────────────────────────
function schedulePrediction() {
  clearTimeout(state.predictTimer);
  state.predictTimer = setTimeout(runPrediction, 200);
}

async function runPrediction() {
  if (!state.hasPixels) return;
  const pixels = get28x28();
  state.lastPixels = pixels;
  renderPixelGrid(pixels);

  try {
    const res  = await fetch(`${API}/api/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ pixels: Array.from(pixels) }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      log(err.detail || 'Prediction failed.', 'warn');
      return;
    }
    const data = await res.json();
    updatePredictionUI(data);
    updateFlowUI(data);
  } catch (err) {
    log(`Prediction error: ${err.message}`, 'err');
  }
}

function updatePredictionUI(data) {
  const { prediction, confidence, probabilities, inference_ms } = data;
  predDigit.textContent = prediction;
  predConf.textContent  = (confidence * 100).toFixed(1) + '%';
  predTime.textContent  = `${inference_ms} ms inference`;

  // Update header stats
  document.getElementById('fstat-pred').textContent    = prediction;
  document.getElementById('fstat-conf').textContent    = (confidence * 100).toFixed(1) + '%';
  document.getElementById('fstat-inference').textContent = inference_ms + ' ms';

  // Confidence bars
  let maxIdx = probabilities.indexOf(Math.max(...probabilities));
  for (let d = 0; d < 10; d++) {
    const pct  = (probabilities[d] * 100).toFixed(1);
    const row  = document.querySelector(`.conf-row[data-digit="${d}"]`);
    const bar  = document.getElementById(`conf-${d}`);
    const label= document.getElementById(`conf-pct-${d}`);
    bar.style.width   = pct + '%';
    label.textContent = pct + '%';
    row.classList.toggle('winner', d === maxIdx);
  }
}

function clearPredictionUI() {
  predDigit.textContent = '?';
  predConf.textContent  = '—';
  predTime.textContent  = '—';
  for (let d = 0; d < 10; d++) {
    document.getElementById(`conf-${d}`).style.width   = '0%';
    document.getElementById(`conf-pct-${d}`).textContent = '0%';
    document.querySelector(`.conf-row[data-digit="${d}"]`).classList.remove('winner');
  }
  ['fstat-pred','fstat-conf','fstat-inference'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
}

// ─── Layer Activation Heatmaps ────────────────────────────────────────────────
const ACT_CANVASES = {
  input:  { canvas: document.getElementById('act-input'),  size: 28, count: 784 },
  layer1: { canvas: document.getElementById('act-layer1'), cols: 16, rows: 16, count: 256 },
  layer2: { canvas: document.getElementById('act-layer2'), cols: 16, rows: 8,  count: 128 },
  layer3: { canvas: document.getElementById('act-layer3'), cols: 8,  rows: 8,  count: 64  },
};

function activationColor(v) {
  // Map 0→dark blue, 0.5→cyan, 1→white
  v = Math.max(0, Math.min(1, v));
  if (v < 0.5) {
    const t = v * 2;
    const r = Math.round(6  + t * (59 - 6));
    const g = Math.round(11 + t * (130- 11));
    const b = Math.round(20 + t * (246- 20));
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (v - 0.5) * 2;
    const r = Math.round(59  + t * (255 - 59));
    const g = Math.round(130 + t * (255 - 130));
    const b = Math.round(246 + t * (255 - 246));
    return `rgb(${r},${g},${b})`;
  }
}

function drawHeatmap(canvas, values, cols, rows) {
  const W   = canvas.width;
  const H   = canvas.height;
  const ctx = canvas.getContext('2d');
  const cw  = W / cols;
  const ch  = H / rows;
  ctx.clearRect(0, 0, W, H);
  for (let i = 0; i < Math.min(values.length, cols * rows); i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    ctx.fillStyle = activationColor(values[i]);
    ctx.fillRect(col * cw, row * ch, cw, ch);
  }
}

function updateFlowUI(data) {
  const { activations, prediction, probabilities } = data;
  if (!activations) return;

  // Input: 28×28 grid
  drawHeatmap(ACT_CANVASES.input.canvas,  activations.input,  28, 28);
  drawHeatmap(ACT_CANVASES.layer1.canvas, activations.layer1, 16, 16);
  drawHeatmap(ACT_CANVASES.layer2.canvas, activations.layer2, 16, 8);
  drawHeatmap(ACT_CANVASES.layer3.canvas, activations.layer3, 8,  8);

  // Output neurons
  const maxP = Math.max(...probabilities);
  for (let d = 0; d < 10; d++) {
    const fill = document.getElementById(`out-${d}`);
    const wrap = fill.parentElement;
    fill.style.width = (probabilities[d] * 100) + '%';
    wrap.classList.toggle('winner', d === prediction);
  }
}

function clearFlowUI() {
  Object.values(ACT_CANVASES).forEach(({ canvas }) => {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });
  for (let d = 0; d < 10; d++) {
    document.getElementById(`out-${d}`).style.width = '0%';
    document.getElementById(`out-${d}`).parentElement.classList.remove('winner');
  }
}

// ─── Sample button ────────────────────────────────────────────────────────────
btnSample.addEventListener('click', async () => {
  try {
    const res  = await fetch(`${API}/api/sample`);
    if (!res.ok) { log('Dataset not available yet. Train first to download it.', 'warn'); return; }
    const data = await res.json();
    const pixels = new Float32Array(data.pixels);

    // Draw sample onto the drawing canvas (scale 28×28 → 252×252)
    const tmp  = document.createElement('canvas');
    tmp.width  = 28; tmp.height = 28;
    const tctx = tmp.getContext('2d');
    const imgData = tctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(pixels[i] * 255);
      imgData.data[i*4]   = v;
      imgData.data[i*4+1] = v;
      imgData.data[i*4+2] = v;
      imgData.data[i*4+3] = 255;
    }
    tctx.putImageData(imgData, 0, 0);
    dctx.fillStyle = '#000';
    dctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    dctx.imageSmoothingEnabled = false;
    dctx.drawImage(tmp, 0, 0, drawCanvas.width, drawCanvas.height);

    state.hasPixels = true;
    state.lastPixels = pixels;
    canvasHint.classList.add('hidden');
    renderPixelGrid(pixels);
    log(`Loaded sample digit: ${data.label}`, 'ok');
    runPrediction();
  } catch (err) {
    log(`Sample load error: ${err.message}`, 'err');
  }
});

// ─── Training WebSocket ───────────────────────────────────────────────────────
let trainWs = null;

function connectTrainWs() {
  if (trainWs && trainWs.readyState < 2) return;
  trainWs = new WebSocket(`ws://localhost:8000/ws/train`);

  trainWs.onopen  = () => log('Training channel connected.', 'ok');
  trainWs.onclose = () => { log('Training channel closed.'); setTimeout(connectTrainWs, 3000); };
  trainWs.onerror = () => log('Training channel error.', 'err');

  trainWs.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    handleTrainMsg(msg);
  };
}

function handleTrainMsg(msg) {
  switch (msg.type) {

    case 'state':
      state.isTraining   = msg.is_training;
      state.modelTrained = msg.model_trained;
      updateTrainButtons();
      if (msg.history && msg.history.epochs.length > 0) {
        rebuildCharts(msg.history);
        rebuildEpochTable(msg.history);
      }
      break;

    case 'status':
      log(msg.message);
      progLabel.textContent = msg.message;
      break;

    case 'batch': {
      state.isTraining = true;
      updateTrainButtons();
      const pct = ((msg.epoch - 1) / msg.total_epochs + msg.batch / (msg.total_batches * msg.total_epochs)) * 100;
      progressBar.style.width = pct.toFixed(1) + '%';
      progLabel.textContent   = `Epoch ${msg.epoch} / ${msg.total_epochs}  —  Batch ${msg.batch} / ${msg.total_batches}`;
      progStats.textContent   = `Loss ${msg.loss.toFixed(4)}  |  Acc ${(msg.train_acc*100).toFixed(1)}%  |  LR ${msg.lr}`;
      progDetails.textContent = `Speed: ${msg.speed.toLocaleString()} samples/s  |  Elapsed: ${fmt(msg.elapsed)}  |  ETA: ${fmt(msg.eta)}`;
      break;
    }

    case 'epoch': {
      const { epoch, total_epochs, loss, train_acc, val_acc, epoch_time, history } = msg;
      log(`Epoch ${epoch}/${total_epochs} — Loss: ${loss} | Train: ${(train_acc*100).toFixed(1)}% | Val: ${(val_acc*100).toFixed(1)}%`, 'ok');
      progressBar.style.width = ((epoch / total_epochs) * 100).toFixed(1) + '%';
      rebuildCharts(history);
      addEpochRow({ epoch, total_epochs, loss, train_acc, val_acc, epoch_time });
      break;
    }

    case 'complete':
      state.isTraining   = false;
      state.modelTrained = true;
      updateTrainButtons();
      log(msg.message, 'ok');
      progLabel.textContent = msg.message;
      progressBar.style.width = '100%';
      pillStatus.textContent  = 'Model Ready';
      pillStatus.classList.add('pill-green');
      if (msg.history) { rebuildCharts(msg.history); rebuildEpochTable(msg.history); }
      break;

    case 'error':
      state.isTraining = false;
      updateTrainButtons();
      log(`Training error: ${msg.message}`, 'err');
      break;
  }
}

function fmt(secs) {
  if (!secs || secs < 0) return '—';
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

btnTrain.addEventListener('click', async () => {
  const config = {
    epochs:      parseInt(document.getElementById('inp-epochs').value) || 5,
    lr:          parseFloat(document.getElementById('inp-lr').value)    || 0.001,
    batch_size:  parseInt(document.getElementById('inp-batch').value)   || 64,
    reset_model: document.getElementById('inp-reset').checked,
  };
  try {
    const res = await fetch(`${API}/api/train/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) { const e = await res.json(); log(e.detail, 'warn'); return; }
    log(`Training started: ${config.epochs} epochs, LR=${config.lr}, batch=${config.batch_size}`, 'ok');
    epochTbody.innerHTML = '';
    progressBar.style.width = '0%';
    clearCharts();
  } catch (err) {
    log(`Failed to start training: ${err.message}`, 'err');
  }
});

btnStop.addEventListener('click', async () => {
  await fetch(`${API}/api/train/stop`, { method: 'POST' }).catch(() => {});
  log('Stop signal sent.', 'warn');
});

function updateTrainButtons() {
  btnTrain.disabled = state.isTraining;
  btnStop.disabled  = !state.isTraining;
}

// ─── Charts ───────────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true,
  animation: { duration: 300 },
  plugins: { legend: { labels: { color: '#7a9bbf', font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#3a5570' }, grid: { color: '#1a2e48' } },
    y: { ticks: { color: '#3a5570' }, grid: { color: '#1a2e48' } },
  },
};

const lossChart = new Chart(document.getElementById('chart-loss'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Loss',
      data: [],
      borderColor: '#ef4444',
      backgroundColor: 'rgba(239,68,68,0.1)',
      fill: true,
      tension: 0.4,
      pointRadius: 4,
    }],
  },
  options: { ...CHART_DEFAULTS },
});

const accChart = new Chart(document.getElementById('chart-acc'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Train', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: false, tension: 0.4, pointRadius: 4 },
      { label: 'Val',   data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: false, tension: 0.4, pointRadius: 4 },
    ],
  },
  options: { ...CHART_DEFAULTS },
});

function rebuildCharts(history) {
  const epochs = history.epochs.map(String);
  lossChart.data.labels            = epochs;
  lossChart.data.datasets[0].data  = history.loss;
  accChart.data.labels             = epochs;
  accChart.data.datasets[0].data   = history.train_acc.map(v => +(v * 100).toFixed(2));
  accChart.data.datasets[1].data   = history.val_acc.map(v   => +(v * 100).toFixed(2));
  lossChart.update('none');
  accChart.update('none');
}

function clearCharts() {
  [lossChart, accChart].forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(d => d.data = []);
    c.update('none');
  });
}

// ─── Epoch Table ──────────────────────────────────────────────────────────────
let bestValAcc = 0;

function addEpochRow({ epoch, total_epochs, loss, train_acc, val_acc, epoch_time }) {
  if (val_acc > bestValAcc) bestValAcc = val_acc;
  const isBest = val_acc === bestValAcc;
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${epoch} / ${total_epochs}</td>
    <td>${loss.toFixed(4)}</td>
    <td>${(train_acc * 100).toFixed(2)}%</td>
    <td class="${isBest ? 'best' : ''}">${(val_acc * 100).toFixed(2)}% ${isBest ? '★' : ''}</td>
    <td>${epoch_time}s</td>
  `;
  epochTbody.appendChild(tr);
  epochTbody.scrollTop = epochTbody.scrollHeight;
}

function rebuildEpochTable(history) {
  epochTbody.innerHTML = '';
  bestValAcc = 0;
  history.epochs.forEach((ep, i) => addEpochRow({
    epoch:      ep,
    total_epochs: history.epochs[history.epochs.length - 1],
    loss:       history.loss[i],
    train_acc:  history.train_acc[i],
    val_acc:    history.val_acc[i],
    epoch_time: '—',
  }));
}

// ─── Metrics WebSocket ────────────────────────────────────────────────────────
let metricsWs = null;

function connectMetricsWs() {
  if (metricsWs && metricsWs.readyState < 2) return;
  metricsWs = new WebSocket(`ws://localhost:8000/ws/metrics`);
  metricsWs.onclose = () => setTimeout(connectMetricsWs, 3000);
  metricsWs.onmessage = ({ data }) => updateMetrics(JSON.parse(data));
}

function updateMetrics(m) {
  document.getElementById('cpu-pct').textContent    = m.cpu_percent.toFixed(0) + '%';
  document.getElementById('gauge-cpu').style.width  = m.cpu_percent + '%';
  document.getElementById('cpu-cores').textContent  = `${m.cpu_count} logical cores`;

  document.getElementById('ram-pct').textContent    = m.ram_percent.toFixed(0) + '%';
  document.getElementById('gauge-ram').style.width  = m.ram_percent + '%';
  document.getElementById('ram-detail').textContent = `${m.ram_used_gb} GB used / ${m.ram_total_gb} GB total`;

  document.getElementById('proc-ram').textContent   = m.process_ram_mb.toFixed(0) + ' MB';

  const spd = m.training_speed;
  document.getElementById('train-speed').textContent = spd > 0
    ? (spd >= 1000 ? (spd / 1000).toFixed(1) + 'K' : spd) + ' /s'
    : m.is_training ? '…' : '—';

  document.getElementById('infer-time').textContent = m.last_inference_ms > 0
    ? m.last_inference_ms + ' ms' : '—';

  document.getElementById('device-val').textContent = m.device.toUpperCase();
  document.getElementById('device-sub').textContent = m.device === 'cpu'
    ? 'Running on CPU (no GPU detected)' : 'Running on GPU — fast!';
}

// ─── Initial Status Fetch ─────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const res  = await fetch(`${API}/api/status`);
    if (!res.ok) { pillStatus.textContent = 'Server Error'; return; }
    const data = await res.json();

    state.modelTrained = data.model_trained;

    pillDevice.textContent  = data.device.toUpperCase();
    pillParams.textContent  = data.model_params.toLocaleString() + ' params';
    pillStatus.textContent  = data.model_trained ? 'Model Ready' : 'Not Trained';
    if (data.model_trained) pillStatus.classList.add('pill-green');
    else                    pillStatus.classList.add('pill-amber');

    document.getElementById('fstat-params').textContent = data.model_params.toLocaleString();

    log(`Server connected. Device: ${data.device.toUpperCase()}, Params: ${data.model_params.toLocaleString()}`, 'ok');
    if (!data.model_trained) log('Model not trained yet — click ▶ Train to begin!', 'warn');
    else                     log('Saved model loaded. Draw a digit to predict!', 'ok');

  } catch {
    pillStatus.textContent = 'Offline';
    log('Cannot reach server at localhost:8000. Is the backend running?', 'err');
  }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
(async () => {
  updateTrainButtons();
  clearFlowUI();

  await fetchStatus();

  // WebSockets
  connectTrainWs();
  connectMetricsWs();

  // Heartbeat keep-alives
  setInterval(() => {
    if (trainWs?.readyState === 1)   trainWs.send('ping');
    if (metricsWs?.readyState === 1) metricsWs.send('ping');
  }, 20000);

  log('MNIST Neural Network Explorer ready.');
})();
