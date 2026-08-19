// theme.js -- light / dark / auto for every page in this tree.
//
// WHY A TOKEN TABLE AND NOT TWO STYLESHEETS: these pages paint into a CANVAS,
// so a CSS-only dark mode would flip the chrome and leave every visualization
// a white rectangle. Canvas colours have to be resolved in JS at draw time.
// So one table owns both: `T.<token>` for canvas painting, `--vz-<token>` on
// :root for the DOM chrome, both rebuilt from the same source when the theme
// changes.
//
// USING IT IN A PAGE:
//   import { T } from '../framework/theme.js';
//   r.clear(T.n0);  ctx.fillStyle = T.n11;  ctx.strokeStyle = T.accent;
//
// READ TOKENS AT DRAW TIME, NEVER CAPTURE THEM. `T` is mutated in place when
// the theme changes, so `const ink = T.n11` at module scope freezes the colour
// a page loaded with and it will stay light-mode ink forever. Inside draw() it
// is always current.
//
// THE NEUTRAL RAMP INVERTS, THE HUES DO NOT. n0..n14 run lightest->darkest in
// light mode and darkest->lightest in dark mode, so a page that used n0 as a
// background and n14 as ink keeps that relationship for free. The semantic
// hues (accent / ok / warn / bad / violet / teal / gold) keep their identity in
// both themes and only move in lightness and saturation, because a green that
// turns blue in dark mode is a different fact, not a different theme.

// token -> [light, dark]
const P = {
  // neutral ramp: n0 = page ground, n14 = strongest ink. INVERTS with theme.
  n0:  ['#ffffff', '#16191d'],
  n1:  ['#f7f8f9', '#1a1e22'],
  n2:  ['#f3f4f6', '#1f242a'],
  n3:  ['#eceef0', '#242a31'],
  n4:  ['#e3e6ea', '#2b323a'],
  n5:  ['#dfe3e6', '#333b44'],
  n6:  ['#d0d7de', '#3d4650'],
  n7:  ['#c4ccd3', '#4a545f'],
  n8:  ['#b8bec4', '#5b6672'],
  n9:  ['#9aa4ad', '#7f8b96'],
  n10: ['#8a939b', '#98a3ae'],
  n11: ['#586069', '#b3bcc5'],
  n12: ['#3a4047', '#ccd4dc'],
  n13: ['#24292e', '#e2e7ec'],
  n14: ['#1a1d21', '#f0f3f6'],

  // semantic hues -- identity preserved, lightness/saturation retuned.
  accent:     ['#1f6feb', '#58a6ff'],
  accentBg:   ['#eef3ff', '#122437'],
  accentLine: ['#9ec1ef', '#2f5d96'],
  ok:         ['#2ca02c', '#4ec94e'],
  okDeep:     ['#0a7227', '#2ea043'],
  okBg:       ['#e6ffec', '#0f2417'],
  okLine:     ['#b7f5c2', '#2b5c39'],
  warn:       ['#d2691e', '#f0883e'],
  warnBg:     ['#fff3e9', '#2b1c10'],
  warnLine:   ['#f0ddc9', '#5f4426'],
  warnDeep:   ['#9a5b1a', '#d1893f'],
  bad:        ['#d1242f', '#f85149'],
  violet:     ['#8250df', '#bc8cff'],
  violetDeep: ['#5a189a', '#a371f7'],
  teal:       ['#17a2b8', '#3fc9dd'],
  tealDeep:   ['#0a9396', '#2eb3b6'],
  gold:       ['#9a8b73', '#b5a68d'],
  goldDeep:   ['#9a6700', '#d4a72c'],
  goldBg:     ['#fff7ed', '#271f12'],
  goldLine:   ['#e9ddcb', '#4b3d25'],
};

export const TOKENS = Object.keys(P);

// Live colour table. Mutated in place on theme change -- see the warning above.
export const T = {};

const KEY = 'vz-theme';
const MODES = ['light', 'dark', 'auto'];
let mode = 'auto';
const listeners = new Set();

// ONE control that cycles, not three radio buttons: the state is a single
// setting, and three buttons spend three times the width to say so. The glyph
// shows what is in force NOW and the tooltip says what a click does -- which
// also carries the piece a glyph cannot, namely what `auto` currently resolves
// to.
const GLYPH = { light: '\u2600', dark: '\u263e', auto: 'A' };
const NEXT_MODE = { light: 'dark', dark: 'auto', auto: 'light' };
const MODE_WORD = { light: 'light', dark: 'dark', auto: 'auto' };

/** "Theme: auto (system: dark) -- click for light" */
export function themeTitle() {
  const now = mode === 'auto' ? `auto (system: ${effectiveTheme()})` : MODE_WORD[mode];
  return `Theme: ${now} \u2014 click for ${MODE_WORD[NEXT_MODE[mode]]}`;
}

const mql = typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null;

/** The mode the user asked for: 'light' | 'dark' | 'auto'. */
export const themeMode = () => mode;
/** What that resolves to right now: 'light' | 'dark'. */
export function effectiveTheme() {
  if (mode === 'light' || mode === 'dark') return mode;
  return mql && mql.matches ? 'dark' : 'light';
}

function paint() {
  const dark = effectiveTheme() === 'dark';
  const root = typeof document !== 'undefined' ? document.documentElement : null;
  for (const k of TOKENS) {
    T[k] = P[k][dark ? 1 : 0];
    if (root) root.style.setProperty('--vz-' + k, T[k]);
  }
  if (root) {
    root.dataset.vzTheme = effectiveTheme();
    root.style.colorScheme = effectiveTheme();   // native form controls + scrollbars
  }
  for (const fn of listeners) { try { fn(effectiveTheme()); } catch (_) { /* one bad listener must not stop the rest */ } }
}

/** Set the mode and persist it. Anything not in MODES is treated as 'auto'. */
export function setTheme(next) {
  mode = MODES.includes(next) ? next : 'auto';
  try { localStorage.setItem(KEY, mode); } catch (_) { /* private mode */ }
  paint();
  return mode;
}

/** Run cb(effective) whenever the resolved theme changes. Returns an unsubscribe. */
export function onThemeChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

// Resolution order: ?theme= (a deep link wins, and is NOT persisted so it can't
// silently retheme every later visit) -> stored choice -> auto.
let pinnedByQuery = false;
function initialMode() {
  try {
    const q = new URLSearchParams(location.search).get('theme');
    if (MODES.includes(q)) { pinnedByQuery = true; return q; }
  } catch (_) { /* no location (node) */ }
  try {
    const s = localStorage.getItem(KEY);
    if (MODES.includes(s)) return s;
  } catch (_) { /* no storage */ }
  return 'auto';
}

mode = initialMode();
paint();

// Follow the choice made in another document of this origin: the guided tour
// embeds each demo in an iframe, and a theme switch in the parent must repaint
// the demo inside it, not leave it on the old theme. Also keeps two open tabs
// in agreement. A page pinned by ?theme= opts out -- an explicit deep link is a
// statement about THAT view.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (e) => {
    if (pinnedByQuery || e.key !== KEY) return;
    const next = MODES.includes(e.newValue) ? e.newValue : 'auto';
    if (next === mode) return;
    mode = next;
    paint();
  });
}

// In auto, follow the OS switching live -- no reload.
if (mql) {
  const onOS = () => { if (mode === 'auto') paint(); };
  if (mql.addEventListener) mql.addEventListener('change', onOS);
  else if (mql.addListener) mql.addListener(onOS);          // older WebKit
}


/** '#rrggbb' -> [r,g,b]. */
export function rgbOf(color) {
  // Accepts a token name, '#rrggbb', an [r,g,b] triple, OR a CSS rgb()/rgba()
  // string. That last one is not a nicety: signedColor(), mixColor() and
  // alphaOf() all RETURN rgb()/rgba() strings, so without it the helpers in this
  // file could not be composed with each other -- `inkOn(signedColor(t))` sliced
  // non-hex characters, produced NaN luma, compared false in BOTH branches and
  // silently returned the same ink for every value.
  if (Array.isArray(color)) return [color[0], color[1], color[2]];
  const c = T[color] || color || '#000000';
  const m = /^rgba?\(([^)]+)\)/.exec(c);
  if (m) {
    const p = m[1].split(',').map((x) => parseFloat(x));
    return [p[0] | 0, p[1] | 0, p[2] | 0];
  }
  return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
}

/** Rec. 601 luma of a token name / hex / [r,g,b]. */
export function lumaOf(color) {
  const [r, g, b] = rgbOf(color);
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * A token as a CSS rgba() at the given alpha.
 * Use this instead of a literal `rgba(0,0,0,0.1)` wash: a black wash over a
 * light page reads as a shade, but over a dark one it is nearly invisible.
 * `rgbaToken('n14', 0.1)` is a shade of the INK in both themes, which is what
 * such a wash almost always meant.
 */
export function rgbaToken(token, alpha) {
  const [r, g, b] = rgbOf(T[token] || '#000000');
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Signed value -> colour, matching framework/render.js's diverging ramp.
 * t in [-1,1]: negative is the cool end, 0 is the PAGE GROUND, positive the warm
 * end. Several pages hand-rolled this as `rgb(255, 255-m*150, ...)`, which bakes
 * in a white zero and glares on a dark theme; this keeps zero reading as
 * "nothing here" in both.
 */
export function signedColor(t, opts) {
  const dark = effectiveTheme() === 'dark';
  const m = Math.min(1, Math.abs(t));
  // `opts.pos` / `opts.neg` override the ramp ends. Four pages carried a
  // verbatim copy of this function purely because their ends differ from the
  // default pair and are the SAME in both themes; passing them in is what lets
  // those copies collapse without restyling a single cell.
  const pos = (opts && opts.pos) || (dark ? [229, 83, 75] : [178, 24, 43]);
  const neg = (opts && opts.neg) || (dark ? [88, 150, 226] : [33, 102, 172]);
  return mixColor(T.n0, t >= 0 ? pos : neg, m);
}

/** Mix two colours (token / hex / [r,g,b]) and return a CSS rgb() string. */
export function mixColor(a, b, t) {
  const ca = rgbOf(a), cb = rgbOf(b), f = Math.max(0, Math.min(1, t));
  const c = ca.map((v, i) => Math.round(v + (cb[i] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}


/**
 * A colour at an alpha, from a hex OR a token name.
 * Prefer this over a literal `rgba(0,0,0,0.1)`: a black wash reads as a shade
 * on a light page and as nothing at all on a dark one.
 */
export function alphaOf(color, alpha) {
  const [r, g, b] = rgbOf(color);
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Ink that contrasts with `hex`: whichever END of the neutral ramp actually
 * reads on it. Needed because several ramps (viridis, the categorical set) do
 * NOT invert with the theme, so a fixed `T.n0` label lands white-on-yellow in
 * one theme and dark-on-blue in the other.
 */
export function inkOn(color) {
  // Pick by CONTRAST, not by position in the ramp. `n14` is the ink end in
  // light and the near-WHITE end in dark, so "bright fill -> n14" is right in
  // one theme and inverted in the other (it put near-white text on yellow).
  // Comparing distance in luma is theme-agnostic and needs no threshold: take
  // whichever end is further from the fill. It also answers with the LIVE ramp
  // ends rather than pure black/white, so a label never out-shouts the page.
  const l = lumaOf(color);
  return Math.abs(l - lumaOf(T.n14)) > Math.abs(l - lumaOf(T.n0)) ? T.n14 : T.n0;
}

/**
 * Build the 3-state theme switch used in the page chrome and on the catalogue.
 * Returns an element; the caller decides where it goes.
 */
export function themeSwitch(doc = document) {
  const b = doc.createElement('button');
  b.type = 'button';
  b.className = 'vz-theme-btn';
  b.onclick = () => setTheme(NEXT_MODE[mode]);
  const sync = () => {
    b.textContent = GLYPH[mode];
    b.title = themeTitle();
    b.setAttribute('aria-label', b.title);
  };
  onThemeChange(sync);
  sync();
  return b;
}
