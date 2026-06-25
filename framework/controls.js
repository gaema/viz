// controls.js -- interaction widgets + the step-mode transport for
// viz concept pages. Design: plan/framework.md.
//
// Two roles:
//   1. Widgets (slider / select / toggle / stepper / dimKnobs / button) that
//      write into a shared `state` object and trigger a redraw on change.
//   2. The Transport: play / pause / step+- / scrub over a tensor.js *Steps
//      sequence (collectSteps output), emitting the current step to the page.
//
// Wiring: layout.js constructs `new Controls(panelEl, { onChange })` where
// onChange = page.redraw. Widgets and the transport call onChange to repaint.
// A widget flagged {rebuild:true} marks the transport dirty so layout rebuilds
// the step list before the next draw (inputs changed -> recompute the steps).
//
// ES module; also exposed as window.VizControls for module contexts.

function el(tag, props = {}, kids = []) {
  const e = document.createElement(tag);
  for (const k in props) {
    if (k === 'class') e.className = props[k];
    else if (k === 'text') e.textContent = props[k];
    else if (k.startsWith('on') && typeof props[k] === 'function') e.addEventListener(k.slice(2).toLowerCase(), props[k]);
    else if (props[k] != null) e.setAttribute(k, props[k]);
  }
  for (const c of [].concat(kids)) if (c) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  return e;
}
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export class Controls {
  constructor(root, opts = {}) {
    this.root = root;
    this.onChange = opts.onChange || (() => {});
    this.state = {};
    this._transport = null;
    this._setters = {};   // key -> fn(val) that re-syncs the widget DOM (no event)
  }

  // Programmatic update: set state[key] AND re-sync the widget DOM, so a canvas
  // drag or a URL hook moves the slider thumb / checkbox / select / stepper too
  // (writing this.state[key] directly does NOT -- the widget would go stale).
  // opts: {rebuild} marks the transport dirty; {silent} skips the redraw.
  set(key, val, opts = {}) {
    this.state[key] = val;
    const sync = this._setters[key];
    if (sync) sync(val);
    if (opts.rebuild && this._transport) this._transport._dirty = true;
    if (!opts.silent) this.onChange();
    return this;
  }

  _row(label, control) {
    this.root.appendChild(el('div', { class: 'vz-ctl' }, [
      label ? el('label', { class: 'vz-ctl-label', text: label }) : null, control,
    ]));
    return control;
  }
  _changed(key, val, opts) {
    this.state[key] = val;
    if (opts && opts.rebuild && this._transport) this._transport._dirty = true;
    if (opts && typeof opts.onInput === 'function') opts.onInput(val, this.state);
    this.onChange();
  }

  // A numeric readout that doubles as a click-to-type field: click it, type an
  // exact value, press Enter (or blur) to commit -- clamped to [min,max] and
  // snapped to step; Escape reverts. Shared by slider() + stepper().
  _numField(value, fmt, getRange, onCommit) {
    const inp = el('input', { type: 'text', class: 'vz-val vz-num', value: fmt(value), inputmode: 'decimal', spellcheck: 'false', title: 'click to type an exact value' });
    let editing = false, preEdit = inp.value;
    const commit = () => {
      const raw = parseFloat(inp.value), { min, max, step } = getRange();
      if (!Number.isNaN(raw)) { let v = clamp(raw, min, max); if (step) v = Math.round((v - min) / step) * step + min; v = +(+v).toFixed(6); onCommit(v); inp.value = fmt(v); } else inp.value = preEdit;
      editing = false;
    };
    inp.addEventListener('focus', () => { editing = true; preEdit = inp.value; inp.select(); });
    inp.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') { e.preventDefault(); commit(); inp.blur(); } else if (e.key === 'Escape') { inp.value = preEdit; editing = false; inp.blur(); } });
    inp.addEventListener('blur', () => { if (editing) commit(); });
    return inp;
  }

  // Continuous slider with a click-to-type value field.
  // opts: {label,min,max,step,value,format,rebuild,onInput}.
  slider(key, opts = {}) {
    const { min = 0, max = 1, step = 0.01, value = min } = opts;
    this.state[key] = value;
    const fmt = opts.format || ((v) => (Number.isInteger(step) ? String(v) : (+v).toFixed(2)));
    const input = el('input', { type: 'range', class: 'vz-range', min, max, step, value });
    const num = this._numField(value, fmt, () => ({ min, max, step }), (v) => { input.value = v; this._changed(key, v, opts); });
    num.classList.add('vz-num-r');
    input.addEventListener('input', () => { const v = +input.value; num.value = fmt(v); this._changed(key, v, opts); });
    this._setters[key] = (v) => { input.value = v; if (document.activeElement !== num) num.value = fmt(v); };
    this._row(opts.label || key, el('div', { class: 'vz-slider' }, [input, num]));
    return this;
  }

  // Dropdown. opts: {label, options:[{value,label}]|string[], value, rebuild,onInput}.
  select(key, opts = {}) {
    const options = (opts.options || []).map((o) => (typeof o === 'string' ? { value: o, label: o } : o));
    const value = opts.value != null ? opts.value : (options[0] && options[0].value);
    this.state[key] = value;
    const sel = el('select', { class: 'vz-select' }, options.map((o) => el('option', { value: o.value, ...(o.value === value ? { selected: '' } : {}), text: o.label })));
    sel.addEventListener('change', () => this._changed(key, sel.value, opts));
    this._setters[key] = (v) => { sel.value = v; };
    this._row(opts.label || key, sel);
    return this;
  }

  // Checkbox. opts: {label, value, rebuild, onInput}.
  toggle(key, opts = {}) {
    const value = !!opts.value; this.state[key] = value;
    const box = el('input', { type: 'checkbox', class: 'vz-check', ...(value ? { checked: '' } : {}) });
    box.addEventListener('change', () => this._changed(key, box.checked, opts));
    this._setters[key] = (v) => { box.checked = !!v; };
    this._row(opts.label || key, box);
    return this;
  }

  // Integer -/+ stepper with a click-to-type value field. Defaults rebuild:true.
  stepper(key, opts = {}) {
    const { min = 1, max = 64, step = 1, value = min } = opts;
    const rebuild = opts.rebuild !== false;
    this.state[key] = value;
    const num = this._numField(value, (v) => String(v), () => ({ min, max, step }), (v) => this._changed(key, v, { ...opts, rebuild }));
    num.classList.add('vz-num-c');
    const set = (v) => { v = clamp(v, min, max); num.value = String(v); this._changed(key, v, { ...opts, rebuild }); };
    this._setters[key] = (v) => { if (document.activeElement !== num) num.value = String(clamp(v, min, max)); };
    const dec = el('button', { class: 'vz-btn', text: '−', onclick: () => set(this.state[key] - step) });
    const inc = el('button', { class: 'vz-btn', text: '+', onclick: () => set(this.state[key] + step) });
    this._row(opts.label || key, el('div', { class: 'vz-stepper' }, [dec, num, inc]));
    return this;
  }

  // Free-text input (typed prompts / custom strings). opts: {label, value,
  // placeholder, maxlength, rebuild, onInput}. Fires onInput per keystroke; the
  // value is mirrored into state[key] (and the URL via the deep-link sync).
  text(key, opts = {}) {
    const value = opts.value != null ? String(opts.value) : '';
    this.state[key] = value;
    const input = el('input', { type: 'text', class: 'vz-text', value, ...(opts.placeholder ? { placeholder: opts.placeholder } : {}), ...(opts.maxlength ? { maxlength: opts.maxlength } : {}) });
    input.addEventListener('input', () => this._changed(key, input.value, opts));
    this._setters[key] = (v) => { input.value = v == null ? '' : String(v); };
    this._row(opts.label || key, input);
    return this;
  }

  // Convenience: several integer steppers at once. spec: { key: {label,min,max,step,value} }.
  dimKnobs(spec) { for (const key in spec) this.stepper(key, spec[key]); return this; }

  // Plain action button.
  button(label, onClick) { this._row(null, el('button', { class: 'vz-btn vz-btn-wide', text: label, onclick: onClick })); return this; }

  // Step transport over a tensor.js *Steps sequence.
  // opts.compute(): () -> Array (e.g. collectSteps(matmulSteps(A,B))).
  // opts.onStep(record, index): optional. opts.speed: steps/sec (default 6).
  transport(opts = {}) {
    const t = (this._transport = new Transport(this, opts));
    this.root.appendChild(t.el);
    return t;
  }
}

export class Transport {
  constructor(controls, opts) {
    this.controls = controls;
    this.compute = opts.compute || (() => []);
    this.onStep = opts.onStep || null;
    this.speed = opts.speed || 6;
    this.loop = !!opts.loop;
    this.steps = [];
    this.index = -1;
    this.playing = false;
    this._dirty = true;
    this._last = null;
    this._tick = this._tick.bind(this);

    this.scrub = el('input', { type: 'range', class: 'vz-range vz-scrub', min: -1, max: 0, step: 1, value: -1 });
    this.scrub.addEventListener('input', () => this.seek(+this.scrub.value));
    this.btnPlay = el('button', { class: 'vz-btn', text: '▶', onclick: () => (this.playing ? this.pause() : this.play()) });
    const mk = (txt, fn) => el('button', { class: 'vz-btn', text: txt, onclick: fn });
    this.lbl = el('div', { class: 'vz-step-label', text: '—' });
    this.el = el('div', { class: 'vz-transport' }, [
      el('div', { class: 'vz-transport-row' }, [
        mk('⏮', () => this.seek(-1)),
        mk('◀', () => this.step(-1)),
        this.btnPlay,
        mk('▶▎', () => this.step(1)),
        mk('⏭', () => this.seek(this.steps.length - 1)),
      ]),
      this.scrub, this.lbl,
    ]);
  }

  rebuild() {
    this.steps = this.compute() || [];
    this.scrub.max = Math.max(0, this.steps.length - 1);
    this.index = clamp(this.index, -1, this.steps.length - 1);
    this._dirty = false;
    this._sync();
  }
  rebuildIfDirty() { if (this._dirty) this.rebuild(); }
  current() { return this.index >= 0 && this.index < this.steps.length ? this.steps[this.index] : null; }

  seek(i) {
    this.index = clamp(i | 0, -1, this.steps.length - 1);
    this._sync();
    if (this.onStep) this.onStep(this.current(), this.index);
    this.controls.onChange();
  }
  step(d) { this.seek(this.index + d); }

  play() {
    if (this.playing || !this.steps.length) return;
    if (this.index >= this.steps.length - 1) this.index = -1;   // restart from the top
    this.playing = true; this._last = null; this.btnPlay.textContent = '⏸';
    requestAnimationFrame(this._tick);
  }
  pause() { this.playing = false; this.btnPlay.textContent = '▶'; }

  _tick(now) {
    if (!this.playing) return;
    if (this._last == null) this._last = now;
    if ((now - this._last) / 1000 >= 1 / this.speed) {
      this._last = now;
      if (this.index >= this.steps.length - 1) {
        if (this.loop) this.seek(-1);            // restart from the top, keep playing
        else { this.pause(); return; }
      } else {
        this.seek(this.index + 1);
      }
    }
    requestAnimationFrame(this._tick);
  }

  _sync() {
    this.scrub.value = String(this.index);
    const s = this.current();
    const n = this.steps.length;
    this.lbl.textContent = s ? `${this.index + 1} / ${n}  —  ${s.label || s.op || ''}` : (n ? `0 / ${n}  —  (start)` : '—');
  }
}

if (typeof window !== 'undefined') window.VizControls = { Controls, Transport };
