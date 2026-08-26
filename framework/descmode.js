// descmode.js -- description language (plain / technical) for the ES-module
// pages, i.e. every demo page, through layout.js.
//
// The module twin of framework/mode-boot.js, exactly as theme.js is the module
// twin of theme-boot.js. Both halves agree by CONTRACT, and the contract is the
// whole point of having two files rather than one:
//
//   storage key   'viz-desc-mode'
//   attribute     data-vz-mode on <html>, 'plain' | 'tech'
//   URL override  ?mode=plain|tech applies to THIS VIEW and is NOT stored
//
// Only the last one is easy to get wrong, and it was: a shared ?mode=tech link
// used to overwrite the reader's saved preference permanently, so every later
// visit to the plain URL gave them technical. A link must not reconfigure the
// site for someone. Only a click on the switch is a preference.

const KEY = 'viz-desc-mode';
const MODES = ['plain', 'tech'];
const LABEL = { plain: 'plain', tech: 'technical' };

const listeners = new Set();
let mode = 'plain';
let fromUrl = false;

function stored() {
  try {
    const q = new URLSearchParams(location.search).get('mode');
    if (MODES.includes(q)) { fromUrl = true; return q; }
  } catch (e) { /* no URL API */ }
  try {
    const s = localStorage.getItem(KEY);
    if (MODES.includes(s)) return s;
  } catch (e) { /* no storage */ }
  return 'plain';
}

// Paint ONE group. Split out because a group built by modeSwitch() is not in the
// document yet -- it is returned to a caller that appends it -- so a
// document-wide query cannot reach it and the switch would render unchecked,
// with both radios tabbable, until the first click repainted it.
function paintGroup(g) {
  for (const b of g.querySelectorAll('[role="radio"]')) {
    const on = b.dataset.mode === mode;
    b.setAttribute('aria-checked', String(on));
    b.tabIndex = on ? 0 : -1;
  }
}

function apply() {
  document.documentElement.setAttribute('data-vz-mode', mode);
  for (const g of document.querySelectorAll('.vz-modes')) paintGroup(g);
  for (const fn of listeners) fn(mode);
}

/** The current mode. Read it at use time -- never capture it. */
export function descMode() { return mode; }

/** Set the mode. `persist` false applies it without making it a preference. */
export function setDescMode(next, persist = true) {
  mode = MODES.includes(next) ? next : 'plain';
  if (persist) { try { localStorage.setItem(KEY, mode); } catch (e) { /* private */ } }
  apply();
}

/** Subscribe to changes; returns an unsubscribe. */
export function onDescModeChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

/**
 * The two-radio switch. Single-select, so it is a radiogroup with aria-checked
 * rather than a pair of aria-pressed toggles -- those announce two independent
 * on/off controls and never say that choosing one un-chose the other. One tab
 * stop; the arrows move the selection inside it.
 */
export function modeSwitch(doc = document) {
  const host = doc.createElement('span');
  host.className = 'vz-modes modes';
  host.setAttribute('role', 'radiogroup');
  host.setAttribute('aria-label', 'description language');

  const mk = (m) => {
    const b = doc.createElement('button');
    b.type = 'button';
    b.className = 'vz-mode-btn';
    b.setAttribute('role', 'radio');
    b.dataset.mode = m;
    b.textContent = LABEL[m];
    b.addEventListener('click', () => setDescMode(m));
    b.addEventListener('keydown', (e) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      const next = MODES[(MODES.indexOf(m) + step + MODES.length) % MODES.length];
      setDescMode(next);
      const sel = host.querySelector(`[data-mode="${next}"]`);
      if (sel) sel.focus();
    });
    return b;
  };
  host.append(...MODES.map(mk));
  paintGroup(host);        // it is not in the document yet -- see paintGroup
  return host;
}

mode = stored();
apply();          // a URL mode is applied but deliberately NOT written back

// Same cross-document sync as theme.js: a switch in another tab lands here too.
window.addEventListener('storage', (e) => {
  if (e.key !== KEY) return;
  const next = MODES.includes(e.newValue) ? e.newValue : 'plain';
  if (next === mode) return;
  mode = next;
  apply();
});
