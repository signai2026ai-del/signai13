/* ═══════════════════════════════════════════════════════════════════
   SignAI — Canvas2D Chart Components (charts.js)
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

/* ── Color helper ─────────────────────────────────────────────────── */
function scoreColor(score) {
  if (score >= 50) return '#00d4aa';
  if (score >= 20) return '#ffa502';
  return '#2a3550';
}

/* ── Confidence Bar Chart (real-time, all 26 letters) ────────────── */
class ConfidenceChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.scores = new Array(26).fill(0);
    this._raf   = null;
    this._dirty = true;
    this._animVals = new Array(26).fill(0);
    this._render();
  }

  update(letterScores) {
    if (!Array.isArray(letterScores)) return;
    letterScores.forEach(item => {
      const idx = LETTERS.indexOf(item.letter);
      if (idx !== -1) this.scores[idx] = item.score;
    });
    this._dirty = true;
  }

  _render() {
    if (this._dirty) {
      this._draw();
      this._dirty = false;
    }
    this._raf = requestAnimationFrame(() => this._render());
  }

  _draw() {
    const { canvas, ctx, scores, _animVals } = this;
    const dpr = window.devicePixelRatio || 1;
    const W   = canvas.clientWidth;
    const H   = canvas.clientHeight || 200;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, W, H);

    const n      = 26;
    const gap    = 3;
    const barW   = (W - gap * (n - 1)) / n;
    const maxH   = H - 20;

    LETTERS.forEach((letter, i) => {
      const target = scores[i] || 0;
      _animVals[i] += (target - _animVals[i]) * 0.25;  // Smooth lerp
      const val    = _animVals[i];
      const barH   = Math.max(2, (val / 100) * maxH);
      const x      = i * (barW + gap);
      const y      = H - barH - 16;
      const color  = scoreColor(val);

      // Bar fill
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, 3);
      ctx.fill();

      // Letter label
      ctx.fillStyle = val >= 20 ? '#f0f4ff' : '#8892b0';
      ctx.font = `${Math.min(10, barW - 1)}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(letter, x + barW / 2, H - 2);
    });
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
  }
}

/* ── Horizontal Bar Chart (dashboard / statistics) ───────────────── */
function drawHorizontalBar(canvas, data, opts = {}) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.clientWidth;
  const n   = data.length;
  const ROW = opts.rowHeight || 36;
  const H   = ROW * n + 16;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  canvas.style.height = H + 'px';

  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(...data.map(d => d.value), 1);
  const labelW = opts.labelWidth || 28;
  const barArea = W - labelW - 60;

  data.forEach((item, i) => {
    const y      = i * ROW + 8;
    const barH   = ROW - 12;
    const barLen = (item.value / maxVal) * barArea;
    const color  = item.color || '#6c63ff';

    // Letter label
    ctx.fillStyle = '#8892b0';
    ctx.font      = '700 12px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(item.label, labelW - 4, y + barH / 2 + 5);

    // Track
    ctx.fillStyle = '#2a3550';
    ctx.beginPath();
    ctx.roundRect(labelW, y, barArea, barH, 4);
    ctx.fill();

    // Fill (animated via CSS transition analog)
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(labelW, y, Math.max(4, barLen), barH, 4);
    ctx.fill();

    // Value
    ctx.fillStyle = '#f0f4ff';
    ctx.textAlign = 'left';
    ctx.font      = '600 11px Inter, sans-serif';
    ctx.fillText(item.value, labelW + barLen + 6, y + barH / 2 + 4);
  });
}

/* ── Line Chart (confidence over time) ───────────────────────────── */
function drawLineChart(canvas, points, opts = {}) {
  if (!points || points.length < 2) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.clientWidth;
  const H   = canvas.clientHeight || 200;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, W, H);

  const pad   = { top: 16, right: 16, bottom: 32, left: 48 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top  - pad.bottom;

  const vals = points.map(p => p.y);
  const minY = Math.min(...vals, 0);
  const maxY = Math.max(...vals, 100);
  const rangeY = maxY - minY || 1;

  function toX(i) { return pad.left + (i / (points.length - 1)) * chartW; }
  function toY(v) { return pad.top + chartH - ((v - minY) / rangeY) * chartH; }

  // Grid lines
  ctx.strokeStyle = '#2a3550';
  ctx.lineWidth   = 1;
  [0, 25, 50, 75, 100].forEach(v => {
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();

    ctx.fillStyle = '#8892b0';
    ctx.font      = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(v + '%', pad.left - 6, y + 4);
  });

  // Gradient fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  grad.addColorStop(0, 'rgba(108,99,255,0.3)');
  grad.addColorStop(1, 'rgba(108,99,255,0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(points[0].y));
  points.forEach((p, i) => ctx.lineTo(toX(i), toY(p.y)));
  ctx.lineTo(toX(points.length - 1), H - pad.bottom);
  ctx.lineTo(toX(0), H - pad.bottom);
  ctx.closePath();
  ctx.fill();

  // Line (draw with dash-offset animation)
  const totalLen = points.reduce((sum, p, i) => {
    if (i === 0) return 0;
    const dx = toX(i) - toX(i - 1);
    const dy = toY(p.y) - toY(points[i - 1].y);
    return sum + Math.sqrt(dx * dx + dy * dy);
  }, 0);

  ctx.strokeStyle = '#6c63ff';
  ctx.lineWidth   = 2.5;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(points[0].y));
  points.forEach((p, i) => { if (i > 0) ctx.lineTo(toX(i), toY(p.y)); });
  ctx.stroke();

  // X-axis labels (every N points)
  ctx.fillStyle = '#8892b0';
  ctx.font      = '10px Inter, sans-serif';
  ctx.textAlign = 'center';
  const step = Math.ceil(points.length / 6);
  points.forEach((p, i) => {
    if (i % step === 0) {
      ctx.fillText(p.label || '', toX(i), H - pad.bottom + 16);
    }
  });
}

/* ── Donut Chart (top letters) ───────────────────────────────────── */
function drawDonut(canvas, slices, opts = {}) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.clientWidth;
  const H   = canvas.clientHeight || W;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, W, H);

  const cx   = W / 2;
  const cy   = H / 2;
  const outerR = Math.min(W, H) / 2 - 20;
  const innerR = outerR * 0.55;
  const total  = slices.reduce((s, sl) => s + sl.value, 0) || 1;
  const COLORS = ['#6c63ff','#00d4aa','#ffa502','#ff4757','#a29bfe','#55efc4','#fdcb6e','#e17055'];

  let startAngle = -Math.PI / 2;
  slices.forEach((sl, i) => {
    const angle = (sl.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outerR, startAngle, startAngle + angle);
    ctx.arc(cx, cy, innerR, startAngle + angle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fill();

    // Label on outer edge
    const midAngle = startAngle + angle / 2;
    const lx = cx + (outerR + 14) * Math.cos(midAngle);
    const ly = cy + (outerR + 14) * Math.sin(midAngle);
    ctx.fillStyle = '#f0f4ff';
    ctx.font = '700 12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (angle > 0.2) ctx.fillText(sl.label, lx, ly);

    startAngle += angle;
  });

  // Center text
  ctx.fillStyle = '#f0f4ff';
  ctx.font = '700 14px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(opts.centerLabel || 'Letters', cx, cy);
}

/* ── Stability Ring ──────────────────────────────────────────────── */
function updateRing(svgEl, fill, total = 7) {
  const circle = svgEl.querySelector('.ring-fill');
  if (!circle) return;
  const r   = parseFloat(circle.getAttribute('r'));
  const circ = 2 * Math.PI * r;
  circle.style.strokeDasharray  = circ;
  const pct = Math.max(0, Math.min(1, fill / total));
  circle.style.strokeDashoffset = circ * (1 - pct);
}

/* ── Heatmap letter grid ─────────────────────────────────────────── */
function renderHeatmap(container, freqMap) {
  if (!container) return;
  const max = Math.max(...Object.values(freqMap), 1);
  container.innerHTML = '';
  LETTERS.forEach(l => {
    const cnt = freqMap[l] || 0;
    const intensity = cnt / max;
    const cell = document.createElement('div');
    cell.className = 'heatmap-cell';
    cell.setAttribute('data-count', cnt);
    const h = Math.round(240 - intensity * 240);  // Blue (cold) to Red (hot)
    cell.style.background = `hsl(${h}, 70%, ${20 + intensity * 30}%)`;
    cell.style.color = intensity > 0.5 ? '#fff' : '#f0f4ff';
    cell.textContent = l;
    container.appendChild(cell);
  });
}

/* ── Contribution calendar (last 90 days) ─────────────────────────── */
function renderCalendar(container, dailyData) {
  if (!container) return;
  container.innerHTML = '';
  const dayMap = {};
  dailyData.forEach(d => { dayMap[d.day] = d.cnt; });
  const max = Math.max(...Object.values(dayMap), 1);

  for (let i = 89; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    const cnt = dayMap[key] || 0;
    const level = cnt === 0 ? 0 : Math.ceil((cnt / max) * 4);
    const cell = document.createElement('div');
    cell.className = 'cal-cell';
    cell.setAttribute('data-level', level);
    cell.title = `${key}: ${cnt} predictions`;
    container.appendChild(cell);
  }
}

window.SignAICharts = {
  ConfidenceChart, drawHorizontalBar, drawLineChart,
  drawDonut, updateRing, renderHeatmap, renderCalendar, scoreColor
};
