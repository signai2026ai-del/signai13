/* ═══════════════════════════════════════════════════════════════════
   SignAI — Shared Utilities & Global UI (main.js)
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

/* ── Theme Manager ───────────────────────────────────────────────── */
const ThemeManager = (() => {
  const STORAGE_KEY = 'signai-theme';
  let _current = localStorage.getItem(STORAGE_KEY) || 'dark';

  function apply(theme) {
    document.documentElement.setAttribute('data-theme', theme === 'light' ? 'light' : '');
    _current = theme;
    localStorage.setItem(STORAGE_KEY, theme);
    document.querySelectorAll('[data-theme-icon]').forEach(el => {
      el.querySelector('.icon-sun').style.display  = theme === 'dark'  ? 'block' : 'none';
      el.querySelector('.icon-moon').style.display = theme === 'light' ? 'block' : 'none';
    });
  }

  function toggle() { apply(_current === 'dark' ? 'light' : 'dark'); }

  function init() {
    apply(_current);
    document.querySelectorAll('.theme-btn').forEach(btn => {
      btn.addEventListener('click', toggle);
    });
  }

  return { init, toggle, apply, get: () => _current };
})();

/* ── Toast Notifications ─────────────────────────────────────────── */
const Toast = (() => {
  function _getContainer() {
    let c = document.getElementById('toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'toast-container';
      document.body.appendChild(c);
    }
    return c;
  }

  function _iconSvg(type) {
    const icons = {
      success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
      error:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
      info:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
      warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    };
    return icons[type] || icons.info;
  }

  function show(message, type = 'info', duration = 3500) {
    const container = _getContainer();
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.innerHTML = `${_iconSvg(type)}<span>${message}</span>`;
    container.appendChild(el);

    setTimeout(() => {
      el.classList.add('out');
      el.addEventListener('animationend', () => el.remove(), { once: true });
    }, duration);

    el.addEventListener('click', () => {
      el.classList.add('out');
      el.addEventListener('animationend', () => el.remove(), { once: true });
    });
  }

  return {
    success: (msg, dur) => show(msg, 'success', dur),
    error:   (msg, dur) => show(msg, 'error',   dur || 5000),
    info:    (msg, dur) => show(msg, 'info',    dur),
    warning: (msg, dur) => show(msg, 'warning', dur),
  };
})();

/* ── API Fetch Helper ────────────────────────────────────────────── */
async function apiFetch(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);

  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      signal: controller.signal,
      ...options,
    });
    clearTimeout(timeout);
    const json = await res.json();
    return { ok: res.ok, status: res.status, data: json };
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') throw new Error('Request timed out (10s).');
    throw err;
  }
}

/* ── Animated Counter ────────────────────────────────────────────── */
function animateCounter(el, target, duration = 1200, suffix = '') {
  const start = performance.now();
  const initial = 0;
  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(initial + (target - initial) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── IntersectionObserver Reveal ─────────────────────────────────── */
function initReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        // Trigger counters
        const counters = entry.target.querySelectorAll('[data-counter]');
        counters.forEach(el => {
          const target = parseFloat(el.dataset.counter);
          const suffix = el.dataset.suffix || '';
          animateCounter(el, target, 1400, suffix);
        });
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
  document.querySelectorAll('[data-counter]:not(.reveal *)').forEach(el => {
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        animateCounter(el, parseFloat(el.dataset.counter), 1400, el.dataset.suffix || '');
        obs.unobserve(el);
      }
    }, { threshold: 0.2 });
    obs.observe(el);
  });
}

/* ── Modal Manager ───────────────────────────────────────────────── */
const Modal = (() => {
  function open(id) {
    const overlay = document.getElementById(id);
    if (!overlay) return;
    overlay.classList.add('open');
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close(id);
    }, { once: true });
    document.addEventListener('keydown', onEscape);
    function onEscape(e) {
      if (e.key === 'Escape') { close(id); document.removeEventListener('keydown', onEscape); }
    }
  }

  function close(id) {
    const overlay = document.getElementById(id);
    if (overlay) overlay.classList.remove('open');
  }

  function shake(id) {
    const modal = document.querySelector(`#${id} .modal`);
    if (!modal) return;
    modal.classList.remove('shake');
    void modal.offsetWidth;
    modal.classList.add('shake');
    modal.addEventListener('animationend', () => modal.classList.remove('shake'), { once: true });
  }

  return { open, close, shake };
})();

/* ── Tab Manager ─────────────────────────────────────────────────── */
function initTabs(container) {
  const tabs = container.querySelectorAll('.tab-btn');
  const panels = container.querySelectorAll('.tab-panel');
  const indicator = container.querySelector('.tab-indicator');

  function activate(btn) {
    tabs.forEach(t => t.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const target = container.querySelector(`#${btn.dataset.tab}`);
    if (target) target.classList.add('active');
    if (indicator) {
      indicator.style.left  = btn.offsetLeft + 'px';
      indicator.style.width = btn.offsetWidth + 'px';
    }
  }

  tabs.forEach(btn => {
    btn.addEventListener('click', () => activate(btn));
  });

  if (tabs[0]) {
    requestAnimationFrame(() => activate(tabs[0]));
  }
}

/* ── Navbar Mobile Toggle ────────────────────────────────────────── */
function initNavbar() {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.navbar-nav');
  if (!toggle || !nav) return;

  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('mobile-open');
    toggle.setAttribute('aria-expanded', open);
  });

  document.addEventListener('click', (e) => {
    if (!toggle.contains(e.target) && !nav.contains(e.target)) {
      nav.classList.remove('mobile-open');
      toggle.setAttribute('aria-expanded', false);
    }
  });
}

/* ── Password Strength Meter ─────────────────────────────────────── */
function initPasswordStrength(inputEl, barEl, labelEl) {
  if (!inputEl || !barEl) return;

  function measure(pwd) {
    let score = 0;
    if (pwd.length >= 8)  score++;
    if (pwd.length >= 12) score++;
    if (/[A-Z]/.test(pwd) && /[a-z]/.test(pwd)) score++;
    if (/\d/.test(pwd)) score++;
    if (/[^A-Za-z0-9]/.test(pwd)) score++;
    return Math.min(4, score);
  }

  const labels = ['', 'Weak', 'Fair', 'Strong', 'Very Strong'];
  const colors = ['', '#ff4757', '#ffa502', '#6c63ff', '#00d4aa'];
  const widths = ['0%', '25%', '50%', '75%', '100%'];

  inputEl.addEventListener('input', () => {
    const s = measure(inputEl.value);
    barEl.style.width    = widths[s];
    barEl.style.background = colors[s];
    if (labelEl) {
      labelEl.textContent = labels[s];
      labelEl.style.color = colors[s];
    }
  });
}

/* ── Email Validation Icon ───────────────────────────────────────── */
function initEmailValidation(inputEl) {
  if (!inputEl) return;
  const wrapper = inputEl.parentElement;
  let validIcon   = wrapper.querySelector('.email-icon.valid');
  let invalidIcon = wrapper.querySelector('.email-icon.invalid');
  if (!validIcon || !invalidIcon) return;

  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  inputEl.addEventListener('input', () => {
    const valid = re.test(inputEl.value);
    validIcon.style.display   = inputEl.value && valid  ? 'block' : 'none';
    invalidIcon.style.display = inputEl.value && !valid ? 'block' : 'none';
  });
}

/* ── Submit Button Loading State ─────────────────────────────────── */
function setButtonLoading(btn, loading) {
  if (loading) {
    btn.classList.add('loading');
    btn.disabled = true;
  } else {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

/* ── Form double-submit protection ──────────────────────────────── */
function initFormProtection(form) {
  form.addEventListener('submit', function handler() {
    const btn = form.querySelector('[type="submit"]');
    if (btn) setButtonLoading(btn, true);
    form.removeEventListener('submit', handler);
  });
}

/* ── Particle Canvas Background ──────────────────────────────────── */
function initParticles(canvasEl) {
  if (!canvasEl) return;
  const ctx = canvasEl.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvasEl.width  = canvasEl.offsetWidth;
    H = canvasEl.height = canvasEl.offsetHeight;
  }

  function makeParticle() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.5 + 0.5,
      alpha: Math.random() * 0.5 + 0.15,
    };
  }

  function init() {
    resize();
    particles = Array.from({ length: 120 }, makeParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(108,99,255,${p.alpha})`;
      ctx.fill();
    });

    // Draw connecting lines between nearby particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(108,99,255,${0.08 * (1 - dist / 100)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init();
  draw();
}

/* ── Global Init ─────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  ThemeManager.init();
  initReveal();
  initNavbar();

  // Init all tab groups
  document.querySelectorAll('[data-tabs]').forEach(initTabs);

  // Password fields
  document.querySelectorAll('.password-wrapper').forEach(wrapper => {
    const input = wrapper.querySelector('input[type="password"]');
    const toggle = wrapper.querySelector('.password-toggle');
    if (!input || !toggle) return;
    toggle.addEventListener('click', () => {
      input.type = input.type === 'password' ? 'text' : 'password';
      toggle.setAttribute('aria-label', input.type === 'password' ? 'Show password' : 'Hide password');
    });
  });

  // Auth forms
  const pwInput  = document.getElementById('password');
  const pwBar    = document.getElementById('strength-bar');
  const pwLabel  = document.getElementById('strength-label');
  initPasswordStrength(pwInput, pwBar, pwLabel);
  initEmailValidation(document.getElementById('email'));

  // Form protections
  document.querySelectorAll('form[data-protect]').forEach(initFormProtection);

  // Modal close buttons
  document.querySelectorAll('[data-modal-close]').forEach(btn => {
    btn.addEventListener('click', () => Modal.close(btn.dataset.modalClose));
  });
  document.querySelectorAll('[data-modal-open]').forEach(btn => {
    btn.addEventListener('click', () => Modal.open(btn.dataset.modalOpen));
  });

  // Particle canvases
  document.querySelectorAll('.hero-canvas, .auth-canvas').forEach(initParticles);
});

/* Export to global scope for template scripts */
window.SignAI = { Toast, Modal, apiFetch, setButtonLoading, initTabs, animateCounter };
