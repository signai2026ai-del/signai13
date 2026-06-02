/* ═══════════════════════════════════════════════════════════════════
   SignAI — Recognition Page Logic
   ───────────────────────────────────────────────────────────────────
   Camera: Server-side OpenCV → MJPEG stream (no getUserMedia / HTTPS needed)
   Predictions: Sequential async loop → /api/predict/frame
                Waits for each response before starting next request.
                This prevents flooding the server with parallel requests.
   UI: Updates only when the displayed value changes meaningfully
       to eliminate visual flickering.
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

document.addEventListener('DOMContentLoaded', () => {
  const { Toast, apiFetch, setButtonLoading } = window.SignAI;
  const { ConfidenceChart, updateRing } = window.SignAICharts;

  /* ── State ────────────────────────────────────────────────────── */
  let _isRunning        = false;
  let _autoSave         = true;
  let _soundEnabled     = true;
  let _showLandmarks    = true;
  let _confChart        = null;
  let _lastStableLetter = null;

  /* Anti-flicker: track what is currently displayed */
  let _displayedLetter    = null;
  let _displayedConf      = -1;
  let _displayedHandState = null;   // 'detected' | 'none'

  /* ── DOM refs ─────────────────────────────────────────────────── */
  const videoFeed         = document.getElementById('video-feed');
  const bigLetter         = document.getElementById('big-letter');
  const confPercent       = document.getElementById('conf-percent');
  const confBar           = document.getElementById('conf-bar');
  const top5Panel         = document.getElementById('top5-panel');
  const wordDisplay       = document.getElementById('word-display');
  const noHandOverlay     = document.getElementById('no-hand-overlay');
  const disconnectOverlay = document.getElementById('disconnect-overlay');
  const liveBadge         = document.getElementById('live-badge');
  const uncertaintyWarn   = document.getElementById('uncertainty-warning');
  const chartCanvas       = document.getElementById('conf-chart');
  const ringSvg           = document.getElementById('stability-ring');
  const bufferVotes       = document.getElementById('buffer-votes');
  const videoPredLetter   = document.getElementById('video-pred-letter');
  const videoPredConf     = document.getElementById('video-pred-conf');
  const cameraError       = document.getElementById('camera-error');
  const latencyDisplay    = document.getElementById('latency-display');

  /* Settings panel */
  const settingsBtn      = document.getElementById('settings-btn');
  const settingsSidebar  = document.getElementById('settings-sidebar');
  const sidebarOverlay   = document.getElementById('sidebar-overlay');
  const thresholdSlider  = document.getElementById('threshold-slider');
  const thresholdDisplay = document.getElementById('threshold-display');
  const toggleAutoSave   = document.getElementById('toggle-autosave');
  const toggleSound      = document.getElementById('toggle-sound');
  const toggleLandmarks  = document.getElementById('toggle-landmarks');
  const fpsSelect        = document.getElementById('fps-select');

  /* ── Init Charts ──────────────────────────────────────────────── */
  if (chartCanvas) _confChart = new ConfidenceChart(chartCanvas);

  /* ── Camera Controls ──────────────────────────────────────────── */
  async function startCamera() {
    const btn = document.getElementById('btn-start-camera');
    setButtonLoading(btn, true);
    hideCameraError();
    try {
      const res = await apiFetch('/api/camera/open', { method: 'POST' });
      if (!res.data.success) throw new Error(res.data.error || 'Failed to open camera');
      _isRunning = true;
      videoFeed.src = '/video_feed';
      videoFeed.style.display = 'block';
      liveBadge && (liveBadge.style.display = 'flex');
      if (btn) { btn.textContent = 'Camera On'; btn.disabled = true; }
      disconnectOverlay && disconnectOverlay.classList.remove('show');
      runPredictionLoop();
    } catch (err) {
      showCameraError('Could not open camera: ' + err.message);
    } finally {
      setButtonLoading(btn, false);
    }
  }

  async function stopCamera() {
    _isRunning = false;
    videoFeed.src = '';
    videoFeed.style.display = 'none';
    liveBadge && (liveBadge.style.display = 'none');
    const btn = document.getElementById('btn-start-camera');
    if (btn) { btn.textContent = 'Start Camera'; btn.disabled = false; }
    await apiFetch('/api/camera/close', { method: 'POST' }).catch(() => {});
  }

  /* ── Camera feed disconnect detection ────────────────────────── */
  if (videoFeed) {
    videoFeed.addEventListener('error', () => {
      if (_isRunning) {
        disconnectOverlay && disconnectOverlay.classList.add('show');
        stopCamera();
      }
    });
    videoFeed.addEventListener('load', () => {
      disconnectOverlay && disconnectOverlay.classList.remove('show');
    });
  }
  if (disconnectOverlay) {
    disconnectOverlay.addEventListener('click', startCamera);
  }

  /* ── Camera error helpers ─────────────────────────────────────── */
  function showCameraError(msg) {
    if (cameraError) { cameraError.textContent = msg; cameraError.style.display = 'block'; }
    Toast.error(msg, 6000);
  }
  function hideCameraError() {
    if (cameraError) cameraError.style.display = 'none';
  }

  /* ══════════════════════════════════════════════════════════════
     PREDICTION LOOP — Sequential, not setInterval
     ──────────────────────────────────────────────────────────────
     Why sequential?
       setInterval fires even if the previous request hasn't
       finished, flooding the server with parallel requests.
       This loop instead:
         1. Sends request
         2. Waits for response
         3. Sleeps for (targetInterval - elapsed) ms
         4. Repeats
     This gives a clean, predictable FPS with zero request piling.
   ══════════════════════════════════════════════════════════════ */

  async function runPredictionLoop() {
    const getFps = () => parseInt(fpsSelect?.value || '10');

    while (_isRunning) {
      const fps          = getFps();
      const targetMs     = 1000 / fps;
      const t0           = performance.now();

      try {
        const res = await apiFetch('/api/predict/frame', {
          method: 'POST',
          body:   JSON.stringify({ auto_save: _autoSave }),
        });

        if (_isRunning && res.data.success) {
          renderPrediction(res.data.data);
        }
      } catch (err) {
        /* Silent — network hiccups shouldn't crash the loop */
        console.warn('[SignAI] Prediction error:', err.message);
      }

      /* Sleep for remaining time in this frame window */
      const elapsed = performance.now() - t0;
      const sleep   = Math.max(0, targetMs - elapsed);
      if (sleep > 0) {
        await new Promise(r => setTimeout(r, sleep));
      }
    }
  }

  /* ══════════════════════════════════════════════════════════════
     RENDER — Anti-flicker: only update DOM when value changes
   ══════════════════════════════════════════════════════════════ */

  function renderPrediction(d) {
    /* ── No hand detected ───────────────────────────────────────── */
    if (!d.hand_detected) {
      if (_displayedHandState !== 'none') {
        _displayedHandState = 'none';
        _displayedLetter    = null;
        _displayedConf      = -1;
        noHandOverlay && noHandOverlay.classList.add('show');
        if (bigLetter) { bigLetter.textContent = '—'; bigLetter.className = 'big-letter'; }
        if (videoPredLetter) videoPredLetter.textContent = '—';
        if (videoPredConf)   videoPredConf.textContent   = '';
        top5Panel && (top5Panel.innerHTML = '');
        if (uncertaintyWarn) uncertaintyWarn.classList.remove('show');
      }
      /* Still update ring/votes even if no hand, for buffer drain */
      if (ringSvg) updateRing(ringSvg, d.buffer_fill || 0, 10);
      return;
    }

    /* ── Hand detected ──────────────────────────────────────────── */
    noHandOverlay && noHandOverlay.classList.remove('show');
    _displayedHandState = 'detected';

    const conf   = d.confidence;
    const letter = d.letter;

    /* Big letter — update only when letter changes */
    if (letter !== _displayedLetter) {
      _displayedLetter = letter;
      if (bigLetter) {
        bigLetter.textContent = letter;
        bigLetter.className   = 'big-letter '
          + (conf >= 70 ? 'high-conf' : conf >= 40 ? 'medium-conf' : 'low-conf');
      }
      if (videoPredLetter) videoPredLetter.textContent = letter;

      /* Rebuild top5 only when letter changes */
      if (top5Panel && d.top5) {
        top5Panel.innerHTML = d.top5.map(t =>
          `<div class="top5-pill">
             <span class="letter">${t.letter}</span>
             <span class="pct">${t.confidence.toFixed(1)}%</span>
           </div>`
        ).join('');
      }
    }

    /* Confidence — update only when delta > 1 % (reduces bar jitter) */
    if (Math.abs(conf - _displayedConf) >= 1) {
      _displayedConf = conf;
      if (confPercent) confPercent.textContent = conf.toFixed(1) + '%';
      if (confBar)     confBar.style.width      = conf + '%';
      if (videoPredConf) videoPredConf.textContent = conf.toFixed(1) + '%';

      /* Update letter colour class when confidence bracket changes */
      if (bigLetter) {
        const cls = conf >= 70 ? 'high-conf' : conf >= 40 ? 'medium-conf' : 'low-conf';
        if (!bigLetter.className.includes(cls)) {
          bigLetter.className = 'big-letter ' + cls;
        }
      }
    }

    /* Latency display */
    if (latencyDisplay && d.latency_ms != null) {
      latencyDisplay.textContent = d.latency_ms.toFixed(0) + ' ms';
    }

    /* Low-confidence warning */
    if (uncertaintyWarn) {
      uncertaintyWarn.classList.toggle('show', !!d.low_confidence_warning);
    }

    /* Confidence chart — every frame (it handles its own smoothing) */
    if (_confChart && d.letter_scores) _confChart.update(d.letter_scores);

    /* Stability ring */
    if (ringSvg) updateRing(ringSvg, d.buffer_fill || 0, 10);

    /* Buffer votes */
    if (bufferVotes && d.buffer_votes) {
      bufferVotes.innerHTML = Object.entries(d.buffer_votes)
        .sort(([, a], [, b]) => b - a)
        .map(([l, v]) => `<span class="badge badge-violet">${l}:${v}</span>`)
        .join(' ');
    }

    /* Auto-append stable letter to word (only once per stable event) */
    if (d.stable && d.stable_letter && d.stable_letter !== _lastStableLetter) {
      _lastStableLetter = d.stable_letter;
      appendLetter(d.stable_letter);
    }
  }

  /* ── Word Builder ──────────────────────────────────────────────── */
  async function loadCurrentWord() {
    const res = await apiFetch('/api/word/current').catch(() => null);
    if (res?.data?.success) updateWordDisplay(res.data.data.sentence);
  }

  function updateWordDisplay(sentence) {
    if (!wordDisplay) return;
    wordDisplay.innerHTML = '';
    for (const ch of sentence) {
      const span = document.createElement('span');
      span.className   = 'word-char';
      span.textContent = ch === ' ' ? '\u00A0' : ch;
      wordDisplay.appendChild(span);
    }
  }

  async function appendLetter(letter) {
    const res = await apiFetch('/api/word/append', {
      method: 'POST',
      body:   JSON.stringify({ letter }),
    }).catch(() => null);
    if (res?.data?.success) updateWordDisplay(res.data.data.sentence);
  }

  async function wordAction(endpoint) {
    const res = await apiFetch(endpoint, { method: 'POST' }).catch(() => null);
    if (res?.data?.success) {
      updateWordDisplay(res.data.data.sentence);
      if (endpoint === '/api/word/space' && _soundEnabled) {
        const s = res.data.data.sentence.trim();
        if (s) speakText(s);
      }
    }
  }

  /* ── TTS ──────────────────────────────────────────────────────── */
  async function speakText(text) {
    if (!_soundEnabled || !text.trim()) return;
    try {
      const res = await apiFetch('/api/tts/speak', {
        method: 'POST',
        body:   JSON.stringify({ text }),
      });
      if (!res.data.success) return;
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const raw = atob(res.data.data.audio_base64);
      const buf = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
      const decoded = await audioCtx.decodeAudioData(buf.buffer);
      const source  = audioCtx.createBufferSource();
      source.buffer = decoded;
      source.connect(audioCtx.destination);
      source.start();
    } catch (err) {
      console.warn('TTS error:', err.message);
    }
  }

  /* ── Settings Panel ────────────────────────────────────────────── */
  if (settingsBtn) {
    settingsBtn.addEventListener('click', () => {
      settingsSidebar && settingsSidebar.classList.toggle('open');
      sidebarOverlay  && sidebarOverlay.classList.toggle('open');
    });
  }
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', () => {
      settingsSidebar && settingsSidebar.classList.remove('open');
      sidebarOverlay.classList.remove('open');
    });
  }

  if (thresholdSlider) {
    thresholdSlider.addEventListener('input', () => {
      if (thresholdDisplay)
        thresholdDisplay.textContent = (thresholdSlider.value * 100).toFixed(0) + '%';
    });
    thresholdSlider.addEventListener('change', async () => {
      const val = parseFloat(thresholdSlider.value);
      const res = await apiFetch('/api/profile/threshold', {
        method: 'POST',
        body:   JSON.stringify({ confidence_threshold: val }),
      }).catch(() => null);
      if (res?.data?.success) Toast.success('Threshold updated.');
      else Toast.error('Failed to update threshold.');
    });
  }

  if (toggleAutoSave) toggleAutoSave.addEventListener('change', () => {
    _autoSave = toggleAutoSave.checked;
    Toast.info('Auto-save ' + (_autoSave ? 'on' : 'off'));
  });
  if (toggleSound) toggleSound.addEventListener('change', () => {
    _soundEnabled = toggleSound.checked;
    Toast.info('Sound ' + (_soundEnabled ? 'on' : 'off'));
  });
  if (toggleLandmarks) toggleLandmarks.addEventListener('change', () => {
    _showLandmarks = toggleLandmarks.checked;
    Toast.info('Landmarks ' + (_showLandmarks ? 'on' : 'off'));
  });

  /* ── Keyboard Shortcuts ────────────────────────────────────────── */
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
      case ' ':         e.preventDefault(); wordAction('/api/word/space');     break;
      case 'Backspace': wordAction('/api/word/backspace');                      break;
      case 'Delete':    wordAction('/api/word/clear');                          break;
      case 'Enter': {
        const s = wordDisplay?.textContent.trim();
        if (s) speakText(s);
        break;
      }
    }
  });

  /* ── Button Bindings ───────────────────────────────────────────── */
  document.getElementById('btn-start-camera')?.addEventListener('click', startCamera);
  document.getElementById('btn-stop-camera')?.addEventListener('click',  stopCamera);

  document.getElementById('btn-space')    ?.addEventListener('click', () => wordAction('/api/word/space'));
  document.getElementById('btn-backspace')?.addEventListener('click', () => wordAction('/api/word/backspace'));
  document.getElementById('btn-clear')    ?.addEventListener('click', () => wordAction('/api/word/clear'));

  document.getElementById('btn-speak')?.addEventListener('click', () => {
    const s = wordDisplay?.textContent.trim();
    if (s) speakText(s);
    else Toast.info('Nothing to speak yet.');
  });

  document.getElementById('btn-reset-session')?.addEventListener('click', async () => {
    await apiFetch('/api/session/reset', { method: 'POST' }).catch(() => null);
    _lastStableLetter = null;
    Toast.info('Session reset.');
  });

  /* ── Load initial state ────────────────────────────────────────── */
  loadCurrentWord();

  /* ── Cleanup on page leave ─────────────────────────────────────── */
  window.addEventListener('beforeunload', () => {
    stopCamera();
    if (_confChart) _confChart.destroy();
  });
});
