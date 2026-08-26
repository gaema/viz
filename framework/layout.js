// layout.js -- page chrome + orchestrator for ai/models/viz concept pages.
// Design: the shared render framework's contract.
//
// Builds the standard shell:
//   +----------------------------------------------+
//   | Title                        [tier badge]     |
//   | one-paragraph "what you're seeing"            |
//   +----------------+-----------------------------+
//   | controls panel |   main canvas               |
//   +----------------+-----------------------------+
//   | math readout strip (live values)             |
//   +----------------------------------------------+
//
// and wires together render.js (Renderer), controls.js (Controls + step
// Transport), and the page's draw() callback.
//
// Usage (a page module):
//   import { mount } from '../framework/layout.js';
//   const page = await mount({
//     title, blurb, prefer: 'webgl2',
//     controls: (c, page) => { c.stepper('n', {...}); c.transport({ compute: () => ... }); },
//     draw: (page) => { page.renderer.clear(T.n0); ...; page.setReadout('...'); },
//   });
//
// The draw callback receives `page`:
//   page.renderer  Renderer (render.js)        page.ctx      its 2D context
//   page.state     live control values         page.controls Controls
//   page.step()    current transport record (with .partial) or null
//   page.W/page.H  logical canvas size         page.setReadout(text)
//   page.redraw()  request a repaint (rAF-coalesced)

import { Renderer } from './render.js';
import { Controls } from './controls.js';
import { currentSlug, neighbours, ORDER, FAMILIES } from './order.js';
import { T, themeSwitch, onThemeChange, alphaOf } from './theme.js';

const STYLE_ID = 'vz-style';
const CSS = `
:root{color-scheme:light}
html,body{background:var(--vz-n0);margin:0}
.vz-page{font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--vz-n14);max-width:1100px;margin:0 auto;padding:16px}
.vz-theme-btn{font:12px ui-monospace,monospace;line-height:1.5;border:1px solid var(--vz-n6);border-radius:6px;background:var(--vz-n0);color:var(--vz-n11);padding:2px 8px;cursor:pointer}
.vz-theme-btn:hover{border-color:var(--vz-accent);color:var(--vz-accent)}
.vz-nav{display:flex;align-items:center;gap:10px;font:12px ui-monospace,monospace;margin-bottom:10px;flex-wrap:wrap}
.vz-nav a{color:var(--vz-accent);text-decoration:none;border:1px solid var(--vz-accentBg);background:var(--vz-accentBg);border-radius:6px;padding:2px 9px;white-space:nowrap}
.vz-nav a:hover{background:var(--vz-accentBg);border-color:var(--vz-accentLine)}
.vz-nav .vz-nav-home{font-weight:600}
/* Wordmark + page picker: the same header contract every gaema.ai site
   carries. The site name goes to this site's root, "· gaema" to the apex, and
   the <select> jumps to any demo -- the control arch uses for its part picker,
   so the two sites navigate the same way. */
.vz-nav .vz-brand{font-weight:650;color:var(--vz-n14);background:none;border:1px solid transparent;padding:2px 4px}
.vz-nav .vz-brand:hover{background:none;border-color:transparent;color:var(--vz-accent)}
.vz-nav .vz-brand-home{color:var(--vz-n11);font-weight:400;background:none;border:1px solid transparent;padding:2px 4px 2px 0}
.vz-nav .vz-brand-home:hover{background:none;border-color:transparent;color:var(--vz-accent)}
.vz-pick{font:12px ui-monospace,monospace;color:var(--vz-n14);background:var(--vz-n0);border:1px solid var(--vz-n6);border-radius:6px;padding:2px 8px;cursor:pointer;max-width:15rem}
.vz-pick:hover{border-color:var(--vz-accent)}
.vz-pick option,.vz-pick optgroup{background:var(--vz-n0);color:var(--vz-n14)}
.vz-nav .vz-nav-pos{color:var(--vz-n10)}
.vz-nav .vz-nav-fam{color:var(--vz-n11);background:var(--vz-n2);border:1px solid var(--vz-n4);border-radius:6px;padding:1px 7px}
.vz-nav .vz-nav-sp{flex:1 1 auto}
.vz-nav .vz-nav-copy{cursor:pointer;color:var(--vz-n13);background:var(--vz-n0);border:1px solid var(--vz-n6)}
.vz-nav .vz-nav-copy:hover{background:var(--vz-n2)}
.vz-nav .vz-nav-dim{color:var(--vz-n8);border-color:var(--vz-n3);background:var(--vz-n1);cursor:default}
.vz-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;border-bottom:1px solid var(--vz-n4);padding-bottom:8px}
.vz-head h1{font-size:20px;margin:0;font-weight:650}
.vz-badge{font:12px ui-monospace,monospace;color:var(--vz-n11);background:var(--vz-n2);border:1px solid var(--vz-n4);border-radius:6px;padding:2px 8px;white-space:nowrap}
.vz-blurb{color:var(--vz-n12);margin:10px 0 14px}
.vz-body{display:flex;gap:16px;align-items:flex-start}
.vz-controls{flex:0 0 230px;display:flex;flex-direction:column;gap:10px}
.vz-stage{flex:1 1 auto;min-width:0;background:var(--vz-n1);border:1px solid var(--vz-n4);border-radius:8px;padding:8px}
.vz-stage canvas{width:100%;height:auto;display:block;border-radius:4px;touch-action:none}
.vz-cmp-row{display:flex;flex-direction:column;gap:14px;width:100%}
.vz-cmp-pane{width:100%}
.vz-cmp-row:not(.on) .vz-cmp-b{display:none}
.vz-cmp-cap{display:none;font:12px ui-monospace,monospace;font-weight:600;margin-bottom:5px;padding:2px 9px;border-radius:5px;text-align:center}
.vz-cmp-row.on .vz-cmp-cap{display:block}
.vz-cmp-cap-a{color:var(--vz-accent);background:var(--vz-accentBg);border:1px solid var(--vz-accentBg)}
.vz-cmp-cap-b{color:var(--vz-warn);background:var(--vz-warnBg);border:1px solid var(--vz-warnLine)}
.vz-readout{margin-top:12px;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--vz-n13);background:var(--vz-n1);border:1px solid var(--vz-n4);border-radius:8px;padding:10px 12px;white-space:pre-wrap;min-height:1.45em}
.vz-challenge{margin-top:10px;border:1px solid var(--vz-goldLine);border-radius:8px;overflow:hidden;font-size:13px}
.vz-ch-head{display:flex;align-items:center;gap:8px;padding:7px 11px;cursor:pointer;background:var(--vz-goldBg);color:var(--vz-warnDeep);font-weight:600;user-select:none}
.vz-ch-head .vz-ch-ar{margin-left:auto;color:var(--vz-warn);font-size:11px;font-weight:400}
.vz-ch-body{padding:9px 12px;display:none;background:var(--vz-n0);border-top:1px solid var(--vz-warnBg)}
.vz-challenge.open .vz-ch-body{display:block}
.vz-ch-goal{color:var(--vz-n13);margin-bottom:6px}
.vz-ch-status{font:12px ui-monospace,monospace;font-weight:600}
.vz-ch-ok{color:var(--vz-ok)}
.vz-ch-no{color:var(--vz-bad)}
.vz-ch-hint{color:var(--vz-gold);font-size:12px;margin-top:5px}
.vz-ch-next{margin-top:8px;font:12px system-ui;cursor:pointer;background:var(--vz-n0);border:1px solid var(--vz-goldLine);border-radius:6px;padding:3px 10px;color:var(--vz-warnDeep)}
.vz-ch-next:hover{background:var(--vz-goldBg)}
.vz-ctl{display:flex;flex-direction:column;gap:3px}
.vz-ctl-label{font-size:12px;color:var(--vz-n11);font-weight:550}
.vz-slider{display:flex;align-items:center;gap:8px}
.vz-range{flex:1 1 auto;width:100%}
.vz-val{font:12px ui-monospace,monospace;color:var(--vz-n13);min-width:34px;text-align:right}
.vz-num{border:1px solid transparent;background:transparent;border-radius:4px;padding:0 3px;cursor:text;font:12px ui-monospace,monospace;color:var(--vz-n13);box-sizing:border-box}
.vz-num:hover{border-color:var(--vz-n5);background:var(--vz-n0)}
.vz-num:focus{outline:none;border-color:var(--vz-accent);background:var(--vz-n0);box-shadow:0 0 0 2px rgba(31,111,235,.15)}
.vz-num-r{width:48px;text-align:right}
.vz-num-c{width:42px;text-align:center}
.vz-select,.vz-check{font:13px system-ui,sans-serif}
.vz-text{width:100%;box-sizing:border-box;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;padding:5px 8px;border:1px solid var(--vz-n6);border-radius:6px;color:var(--vz-n13)}
.vz-text:focus{outline:none;border-color:var(--vz-accent);box-shadow:0 0 0 2px rgba(31,111,235,0.18)}
.vz-stepper{display:flex;align-items:center;gap:6px}
.vz-stepper .vz-val{min-width:28px;text-align:center}
.vz-btn{font:13px system-ui,sans-serif;cursor:pointer;background:var(--vz-n0);border:1px solid var(--vz-n6);border-radius:6px;padding:2px 9px;color:var(--vz-n13);line-height:1.6}
.vz-btn:hover{background:var(--vz-n2)}
.vz-btn:active{background:var(--vz-n3)}
.vz-btn-wide{width:100%;padding:5px}
.vz-transport{display:flex;flex-direction:column;gap:6px;margin-top:4px;padding-top:10px;border-top:1px solid var(--vz-n4)}
.vz-transport-row{display:flex;gap:4px}
.vz-transport-row .vz-btn{flex:1 1 auto;padding:3px 0}
.vz-step-label{font:11.5px ui-monospace,monospace;color:var(--vz-n11);min-height:1.4em;word-break:break-word}
@media(max-width:760px){.vz-body{flex-direction:column}.vz-controls{flex-basis:auto;width:100%}}
`;

function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style'); s.id = STYLE_ID; s.textContent = CSS;
  document.head.appendChild(s);
}

function resolveContainer(target) {
  if (!target) return document.body;
  if (typeof target === 'string') return document.querySelector(target) || document.body;
  return target;
}

export async function mount(opts = {}) {
  injectStyles();
  const root = resolveContainer(opts.mount);

  const badge = document.createElement('span'); badge.className = 'vz-badge'; badge.textContent = '…';
  const h1 = document.createElement('h1'); h1.textContent = opts.title || 'visualization';
  const head = document.createElement('div'); head.className = 'vz-head'; head.append(h1, badge);

  const blurb = document.createElement('p'); blurb.className = 'vz-blurb'; blurb.textContent = opts.blurb || '';

  const panel = document.createElement('div'); panel.className = 'vz-controls';
  const canvas = document.createElement('canvas');
  const stage = document.createElement('div'); stage.className = 'vz-stage'; stage.appendChild(canvas);
  const body = document.createElement('div'); body.className = 'vz-body'; body.append(panel, stage);

  const readout = document.createElement('div'); readout.className = 'vz-readout';

  // In-demo navigation strip: back to the catalogue + prev/next across the
  // curriculum order (order.js), a position indicator, and a copy-link button.
  //
  // `opts.curriculum: false` drops the catalogue / prev / next / position part
  // and keeps only the theme switch + copy link. That is for a page that REUSES
  // this framework from OUTSIDE this tree (ai/models/shard-planner does) and is
  // therefore not in order.js: without it the strip renders a dead
  // `../index.html` link and a permanently dimmed "start / end" pair.
  const slug = currentSlug(opts.slug);
  const nb = neighbours(slug);
  const inCurriculum = opts.curriculum !== false;
  const nav = document.createElement('div'); nav.className = 'vz-nav';
  const mkLink = (href, text, cls) => { const a = document.createElement('a'); a.href = href; a.textContent = text; if (cls) a.className = cls; return a; };
  const mkDim = (text) => { const s = document.createElement('span'); s.className = 'vz-nav-copy vz-nav-dim'; s.textContent = text; return s; };
  if (inCurriculum) {
    // Wordmark first, as on every gaema.ai page: site name -> this site's
    // root, "· gaema" -> the apex. Before this a demo page had no route up to
    // gaema.ai at all -- only the catalogue carried one, so 90-odd pages were
    // dead ends as far as the rest of the family was concerned.
    nav.append(mkLink('../index.html', 'viz', 'vz-brand'));
    const home = mkLink('https://gaema.ai', '· gaema', 'vz-brand-home');
    nav.append(home);

    // Page picker, grouped by family -- the same control arch uses, so the two
    // sites are navigated the same way. It also replaces the old "← all demos"
    // link (its first option) and the "Family X" badge (the optgroup names it).
    const pick = document.createElement('select');
    pick.className = 'vz-pick';
    pick.setAttribute('aria-label', 'Choose a demo');
    const all = document.createElement('option');
    all.value = '../index.html'; all.textContent = 'All demos';
    pick.append(all);
    const groups = new Map();
    for (const item of ORDER) {
      if (!groups.has(item.family)) {
        const g = document.createElement('optgroup');
        g.label = item.family + ' — ' + (FAMILIES[item.family] || item.family);
        groups.set(item.family, g);
        pick.append(g);
      }
      const o = document.createElement('option');
      o.value = '../' + item.slug + '/index.html';
      o.textContent = item.slug;
      if (item.slug === slug) o.selected = true;
      groups.get(item.family).append(o);
    }
    pick.addEventListener('change', () => { if (pick.value) location.href = pick.value; });
    nav.append(pick);

    nav.append(nb.prev ? mkLink('../' + nb.prev.slug + '/index.html', '‹ ' + nb.prev.slug) : mkDim('‹ start'));
    nav.append(nb.next ? mkLink('../' + nb.next.slug + '/index.html', nb.next.slug + ' ›') : mkDim('end ›'));
    if (nb.index >= 0) { const pos = document.createElement('span'); pos.className = 'vz-nav-pos'; pos.textContent = `${nb.index + 1} / ${nb.total}`; nav.append(pos); }
  }
  const spacer = document.createElement('span'); spacer.className = 'vz-nav-sp'; nav.append(spacer);
  const copyBtn = document.createElement('span'); copyBtn.className = 'vz-nav-copy'; copyBtn.style.cursor = 'pointer'; copyBtn.style.borderRadius = '6px'; copyBtn.style.padding = '2px 9px'; copyBtn.textContent = '\u{1F517} copy link';
  copyBtn.addEventListener('click', () => { try { navigator.clipboard.writeText(location.href); const o = copyBtn.textContent; copyBtn.textContent = '✓ copied'; setTimeout(() => { copyBtn.textContent = o; }, 1200); } catch (_) {} });
  nav.append(themeSwitch(document));
  nav.append(copyBtn);

  const page = document.createElement('div'); page.className = 'vz-page';
  page.append(nav, head, blurb, body, readout);
  root.appendChild(page);

  // Global keyboard nav: ←/→ move between demos, '/' jumps to the catalogue.
  // Ignored while a form control (slider/select) is focused so it doesn't fight
  // the widgets.
  if (inCurriculum && typeof document !== 'undefined') document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = document.activeElement, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft' && nb.prev) location.href = '../' + nb.prev.slug + '/index.html';
    else if (e.key === 'ArrowRight' && nb.next) location.href = '../' + nb.next.slug + '/index.html';
    else if (e.key === '/') { e.preventDefault(); location.href = '../index.html'; }
  });

  // Give the canvas a CSS size before the renderer reads its box (DPR sizing).
  if (!canvas.style.height) canvas.style.aspectRatio = opts.aspect || '16 / 10';

  const renderer = await new Renderer().init(canvas, { prefer: opts.prefer || 'webgl2' });
  badge.textContent = renderer.name;

  let curR = renderer, _override = null;   // A/B compare swaps the render target + overrides one state key
  const api = {
    el: page, canvas,
    get renderer() { return curR; }, get ctx() { return curR.ctx; },
    get state() { return _override ? Object.assign({}, controls.state, _override) : controls.state; },
    get W() { return curR.W; }, get H() { return curR.H; },
    step() { return controls._transport ? controls._transport.current() : null; },
    setReadout(text) { readout.textContent = text == null ? '' : String(text); return api; },
    setTip(text) { _tip = text == null ? null : String(text); return api; },
    redraw() { schedule(); return api; },
    pointer: { x: 0, y: 0, over: false, down: false, dragging: false },
    t: 0,
    probe: {},   // a page sets this in draw() with quantities its challenge check() reads
  };

  // Deep links: mirror the control state into the query string on every change
  // so the URL (and the copy-link button) always reproduce the current view.
  // Skipped on file:// (replaceState with a query is unreliable / unneeded there).
  const syncURL = () => {
    if (typeof history === 'undefined' || (typeof location !== 'undefined' && location.protocol === 'file:')) return;
    try {
      const p = new URLSearchParams(), st = controls.state || {};
      for (const k of Object.keys(st)) { const v = st[k]; if (v == null) continue; p.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : v); }
      history.replaceState(null, '', location.pathname + (p.toString() ? '?' + p.toString() : ''));
    } catch (_) {}
  };
  const controls = new Controls(panel, { onChange: () => { api.redraw(); syncURL(); } });
  api.controls = controls;

  // A/B compare: render the page twice, side by side, with ONE
  // parameter set to two values. opts.compare = { key, a, b } overrides a state
  // key; { stepA, stepB } (numbers or 'first'/'last') overrides the transport
  // index. labelA/labelB caption each pane. A "⇄ A/B compare" nav button toggles.
  const cmp = (opts.compare && typeof opts.compare === 'object') ? opts.compare : null;
  let rendererB = null, canvasB = null, compareOn = false, cmpRow = null, cmpBtn = null, pointerCanvas = canvas;
  if (cmp) {
    canvasB = document.createElement('canvas');
    if (!canvasB.style.height) canvasB.style.aspectRatio = opts.aspect || '16 / 10';
    const capA = document.createElement('div'); capA.className = 'vz-cmp-cap vz-cmp-cap-a'; capA.textContent = cmp.labelA || 'A';
    const capB = document.createElement('div'); capB.className = 'vz-cmp-cap vz-cmp-cap-b'; capB.textContent = cmp.labelB || 'B';
    const paneA = document.createElement('div'); paneA.className = 'vz-cmp-pane';
    const paneB = document.createElement('div'); paneB.className = 'vz-cmp-pane vz-cmp-b';
    cmpRow = document.createElement('div'); cmpRow.className = 'vz-cmp-row';
    stage.removeChild(canvas); paneA.append(capA, canvas); paneB.append(capB, canvasB); cmpRow.append(paneA, paneB); stage.appendChild(cmpRow);
    rendererB = await new Renderer().init(canvasB, { prefer: opts.prefer || 'webgl2' });
    // Pane B gets the same pointer pipeline as A (ptr() is hoisted), tagged with
    // its own canvas/renderer so coords + the active-pane tooltip resolve right.
    canvasB.style.touchAction = 'none';
    canvasB.addEventListener('pointermove', (e) => ptr(e, 'move', canvasB, rendererB));
    canvasB.addEventListener('pointerdown', (e) => { try { canvasB.setPointerCapture(e.pointerId); } catch (_) {} ptr(e, 'down', canvasB, rendererB); });
    canvasB.addEventListener('pointerup', (e) => ptr(e, 'up', canvasB, rendererB));
    canvasB.addEventListener('pointerleave', (e) => ptr(e, 'leave', canvasB, rendererB));
    cmpBtn = document.createElement('span'); cmpBtn.className = 'vz-nav-copy'; cmpBtn.style.cssText = 'cursor:pointer;border-radius:6px;padding:2px 9px';
    cmpBtn.textContent = '⇄ A/B compare';
    cmpBtn.addEventListener('click', () => { compareOn = !compareOn; cmpRow.classList.toggle('on', compareOn); cmpBtn.textContent = compareOn ? '⇄ comparing A·B' : '⇄ A/B compare'; cmpBtn.style.background = compareOn ? T.accentBg : T.n0; if (compareOn && controls._transport) { const tr = controls._transport; tr.pause(); if (tr.steps && tr.steps.length) tr.seek(tr.steps.length - 1); } else if (!compareOn && cmp.rebuild && controls._transport) { controls._transport.rebuild(); } api.redraw(); });
    nav.insertBefore(cmpBtn, copyBtn);
  }

  // Challenge mode: optional goals with a live pass/fail check. A page
  // declares mount({ challenges: [{goal, check, hint}] }); check(api) reads
  // api.state + api.probe (set by draw) and returns {solved, detail}. The card
  // toggles open; chEval runs after each draw.
  const challenges = Array.isArray(opts.challenges) ? opts.challenges.filter((c) => c && typeof c.check === 'function') : [];
  let chIdx = 0, chEval = null;
  if (challenges.length) {
    const mkEl = (cls, txt) => { const e = document.createElement('div'); e.className = cls; if (txt) e.textContent = txt; return e; };
    const box = mkEl('vz-challenge'), head = mkEl('vz-ch-head'), bodyc = mkEl('vz-ch-body');
    head.append(document.createTextNode('🎯 Challenge mode')); const ar = mkEl('vz-ch-ar', 'click to start'); head.append(ar);
    const goalEl = mkEl('vz-ch-goal'), statusEl = mkEl('vz-ch-status'), hintEl = mkEl('vz-ch-hint');
    const nextBtn = document.createElement('button'); nextBtn.className = 'vz-ch-next'; nextBtn.textContent = '↻ next challenge';
    bodyc.append(goalEl, statusEl, hintEl, nextBtn); box.append(head, bodyc);
    head.addEventListener('click', () => { box.classList.toggle('open'); ar.textContent = box.classList.contains('open') ? `${chIdx + 1}/${challenges.length}` : 'click to start'; api.redraw(); });
    nextBtn.addEventListener('click', () => { chIdx = (chIdx + 1) % challenges.length; ar.textContent = `${chIdx + 1}/${challenges.length}`; api.redraw(); });
    page.append(box);
    // ?ch=N (1-based) or ?challenge=1 opens the card (deep-link / screenshot hook)
    const cq = (typeof location !== 'undefined') ? new URLSearchParams(location.search) : new URLSearchParams();
    if (cq.has('ch') || cq.get('challenge') === '1') { chIdx = Math.max(0, Math.min(challenges.length - 1, (+cq.get('ch') || 1) - 1)); box.classList.add('open'); ar.textContent = `${chIdx + 1}/${challenges.length}`; }
    chEval = () => {
      if (!box.classList.contains('open')) return;
      const c = challenges[chIdx]; let res = {}; try { res = c.check(api) || {}; } catch (_) {}
      goalEl.textContent = '🎯 ' + (c.goal || 'reach the goal');
      statusEl.textContent = res.solved ? '✓ solved!' : '✗ ' + (res.detail || 'not yet'); statusEl.className = 'vz-ch-status ' + (res.solved ? 'vz-ch-ok' : 'vz-ch-no');
      hintEl.textContent = c.hint ? 'hint: ' + c.hint : ''; nextBtn.style.display = challenges.length > 1 ? 'inline-block' : 'none';
    };
  }

  const draw = typeof opts.draw === 'function' ? opts.draw : () => {};
  let pending = false, _tip = null;
  // Draw one compare pane: point the target renderer + apply the side's override.
  function drawSide(rdr, side) {
    curR = rdr;
    let restoreIdx = null;
    if (cmp.key != null) {
      _override = { [cmp.key]: side === 'a' ? cmp.a : cmp.b };
      // rebuild:true -> re-run the transport's compute() with the override active,
      // so pages that build their data in compute() (not draw) reflect this pane's
      // value. Each pane rebuilds + seeks to its last step (full reveal).
      if (cmp.rebuild && controls._transport) { const tr = controls._transport; tr.rebuild(); if (tr.steps && tr.steps.length) tr.index = tr.steps.length - 1; }
    } else if (cmp.stepA != null && controls._transport) { const tr = controls._transport, steps = tr.steps || []; restoreIdx = tr.index; const resolve = (s) => s === 'first' ? 0 : s === 'last' ? steps.length - 1 : (s | 0); tr.index = resolve(side === 'a' ? cmp.stepA : cmp.stepB); }
    rdr.resize(); _tip = null; api.probe = {};
    draw(api);
    // Tooltip only on the pane the pointer is over -- its values reflect that
    // pane's override (the draw above ran with it active).
    if (_tip && ((rdr === renderer ? canvas : canvasB) === pointerCanvas)) drawTip(api, _tip);
    _override = null;
    if (restoreIdx != null && controls._transport) controls._transport.index = restoreIdx;
  }
  function paint() {
    pending = false;
    if (controls._transport) controls._transport.rebuildIfDirty();
    if (compareOn && rendererB) {
      drawSide(renderer, 'a');
      drawSide(rendererB, 'b');
      curR = renderer;
    } else {
      curR = renderer;
      renderer.resize();
      _tip = null; api.probe = {};
      draw(api);
      if (_tip) drawTip(api, _tip);
    }
    if (chEval) chEval();
  }
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(paint); } }

  // Pointer input -> api.pointer (canvas CSS-px space, same as draw rects) +
  // the optional opts.onPointer hook for direct manipulation (drag/grab).
  function ptr(e, type, srcCanvas, srcR) {
    srcCanvas = srcCanvas || canvas; srcR = srcR || renderer;
    pointerCanvas = srcCanvas;
    const r = srcCanvas.getBoundingClientRect();
    const sx = r.width ? srcR.W / r.width : 1, sy = r.height ? srcR.H / r.height : 1;
    const x = (e.clientX - r.left) * sx, y = (e.clientY - r.top) * sy;
    const p = api.pointer, dx = x - p.x, dy = y - p.y;
    if (type === 'leave') { p.over = false; p.down = false; p.dragging = false; }
    else p.over = true;
    if (type === 'down') { p.down = true; p.dragging = true; }
    if (type === 'up') { p.down = false; p.dragging = false; }
    p.x = x; p.y = y;
    if (typeof opts.onPointer === 'function') opts.onPointer(api, { type, x, y, dx, dy });
    api.redraw();
  }
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointermove', (e) => ptr(e, 'move'));
  canvas.addEventListener('pointerdown', (e) => { try { canvas.setPointerCapture(e.pointerId); } catch (_) {} ptr(e, 'down'); });
  canvas.addEventListener('pointerup', (e) => ptr(e, 'up'));
  canvas.addEventListener('pointerleave', (e) => ptr(e, 'leave'));

  // Let the page register controls (and a transport) before the first paint.
  if (typeof opts.controls === 'function') opts.controls(controls, api);

  // Deep links are READ here, not only written. syncURL() above mirrors control
  // state OUT to the query string, and for a long time nothing read it back --
  // so every `?key=` hook a page documented was inert unless that page wrote its
  // own restore loop. Three builders found it independently and identically: six
  // different URLs producing a byte-identical readout. It also blocks anything
  // that addresses a page purely by URL, which is how the demos get captured
  // for video.
  //
  // Restored BEFORE the transport rebuild, so a restored control feeds the step
  // list about to be computed; and silently, so this costs one paint, not one
  // per parameter. A key the controls do not own (step, theme, hover, play,
  // compare, ch) is left to whoever does own it.
  if (typeof location !== 'undefined') {
    try {
      for (const [k, raw] of new URLSearchParams(location.search)) {
        if (!(k in controls.state)) continue;
        // A query string is UNTRUSTED INPUT. Only a value the widget could
        // itself have produced may enter state -- otherwise a link can push an
        // option that does not exist, or a number past a slider's end, and the
        // page carries it silently. Anything else is ignored, leaving the
        // default exactly as it was.
        const dom = (controls._domain || {})[k];
        const cur = controls.state[k];
        let v = raw;
        if (dom && dom.kind === 'enum') {
          if (!dom.values.includes(raw)) continue;
        } else if (dom && dom.kind === 'bool') {
          v = (raw === '1' || raw === 'true');
        } else if (dom && dom.kind === 'number') {
          const n = Number(raw);
          if (!Number.isFinite(n) || n < dom.min || n > dom.max) continue;
          v = n;
        } else if (typeof cur === 'boolean') {
          v = (raw === '1' || raw === 'true');
        } else if (typeof cur === 'number') {
          const n = Number(raw);
          if (!Number.isFinite(n)) continue;
          v = n;
        }
        controls.set(k, v, { rebuild: true, silent: true });
      }
    } catch (_) {}
  }

  if (controls._transport) controls._transport.rebuild();
  if (opts.autoplay && controls._transport) controls._transport.play();
  // Open compare via ?compare=1 AFTER the transport exists, so the toggle's
  // seek-to-end (full reveal in both panes) actually runs.
  if (cmpBtn && typeof location !== 'undefined' && new URLSearchParams(location.search).get('compare') === '1') cmpBtn.click();

  if (typeof window !== 'undefined') window.addEventListener('resize', () => api.redraw());
  // A theme flip changes every colour the page paints, and canvas pixels do not
  // restyle themselves the way the DOM chrome does -- so repaint on the switch
  // (and on an OS change while in auto).
  onThemeChange(() => api.redraw());

  // Ambient animation clock (dataflow motion): advance api.t + redraw each frame.
  if (opts.animate) {
    let t0 = null;
    const tick = (now) => { if (t0 == null) t0 = now; api.t = (now - t0) / 1000; paint(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  } else {
    // First paint after layout settles (so clientWidth/Height are real).
    requestAnimationFrame(paint);
  }

  return api;
}

// Tooltip for hover-to-inspect: drawn on the 2D ctx AFTER the page's draw, so
// it overlays everything. Page calls api.setTip(text) during draw; multi-line
// via '\n'. Positioned near the cursor, clamped inside the canvas.
function drawTip(api, text) {
  const p = api.pointer; if (!p.over) return;
  const ctx = api.ctx, lines = String(text).split('\n');
  ctx.save();
  ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
  let w = 0; for (const ln of lines) w = Math.max(w, ctx.measureText(ln).width);
  const padX = 7, padY = 5, lh = 14, bw = w + padX * 2, bh = lines.length * lh + padY * 2;
  let x = p.x + 14, y = p.y + 14;
  if (x + bw > api.W) x = p.x - bw - 12;
  if (y + bh > api.H) y = p.y - bh - 12;
  x = Math.max(2, x); y = Math.max(2, y);
  // The tooltip INVERTS: it is painted in the ink colour with ground-coloured
  // text, so it stays a high-contrast bubble in both themes instead of a dark
  // box sitting on an equally dark canvas.
  ctx.fillStyle = alphaOf('n14', 0.93); ctx.strokeStyle = alphaOf('n0', 0.22); ctx.lineWidth = 1;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, bw, bh, 5); ctx.fill(); ctx.stroke(); }
  else { ctx.fillRect(x, y, bw, bh); ctx.strokeRect(x, y, bw, bh); }
  ctx.fillStyle = T.n0;
  for (let i = 0; i < lines.length; i++) ctx.fillText(lines[i], x + padX, y + padY + i * lh);
  ctx.restore();
}

if (typeof window !== 'undefined') window.VizLayout = { mount };
