// mode-boot.js -- description language (plain / technical) for the pages that
// are NOT ES modules.
//
// The classic twin of framework/descmode.js, exactly as theme-boot.js is the
// classic twin of theme.js, and bound to it by the same contract: the same
// storage key, the same `data-vz-mode` attribute on <html>, the same two modes,
// and the same rule that a `?mode=` in the URL is a VIEW rather than a stored
// preference.
//
// Load it in <head>, BEFORE the body renders. The mode decides which of two
// paragraphs is display:none, so applying it later shows BOTH for a frame.
(function () {
  var KEY = 'viz-desc-mode';
  var MODES = ['plain', 'tech'];
  var LABEL = { plain: 'plain', tech: 'technical' };
  var mode = 'plain';
  var fromUrl = false;

  function stored() {
    try {
      var q = new URLSearchParams(location.search).get('mode');
      if (MODES.indexOf(q) >= 0) { fromUrl = true; return q; }
    } catch (e) { /* no URL API */ }
    try {
      var s = localStorage.getItem(KEY);
      if (MODES.indexOf(s) >= 0) return s;
    } catch (e) { /* no storage */ }
    return 'plain';
  }

  function apply() {
    document.documentElement.setAttribute('data-vz-mode', mode);
    var groups = document.querySelectorAll('.vz-modes');
    for (var g = 0; g < groups.length; g++) {
      var btns = groups[g].querySelectorAll('[role="radio"]');
      for (var i = 0; i < btns.length; i++) {
        var on = btns[i].dataset.mode === mode;
        btns[i].setAttribute('aria-checked', String(on));
        btns[i].tabIndex = on ? 0 : -1;
      }
    }
  }

  function set(next) {
    mode = MODES.indexOf(next) >= 0 ? next : 'plain';
    try { localStorage.setItem(KEY, mode); } catch (e) { /* private mode */ }
    apply();
  }

  mode = stored();
  apply();          // a URL mode is applied but deliberately NOT written back

  // Fill any <span data-mode-switch> with the two-radio group. Single-select, so
  // it is a radiogroup with aria-checked -- a pair of aria-pressed buttons would
  // announce two independent on/off controls and never say that choosing one
  // un-chose the other. One tab stop, arrows move the selection.
  function build() {
    var hosts = document.querySelectorAll('[data-mode-switch]');
    for (var h = 0; h < hosts.length; h++) {
      var host = hosts[h];
      if (host.dataset.built) continue;
      host.dataset.built = '1';
      host.className = (host.className ? host.className + ' ' : '') + 'vz-modes modes';
      host.setAttribute('role', 'radiogroup');
      host.setAttribute('aria-label', 'description language');
      for (var i = 0; i < MODES.length; i++) {
        host.appendChild(button(MODES[i]));
      }
    }
    apply();
  }

  function button(m) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'vz-mode-btn';
    b.setAttribute('role', 'radio');
    b.dataset.mode = m;
    b.textContent = LABEL[m];
    b.onclick = function () { set(m); };
    b.onkeydown = function (e) {
      var step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      var order = MODES.slice();
      var next = order[(order.indexOf(m) + step + order.length) % order.length];
      set(next);
      var sel = e.target.parentNode.querySelector('[data-mode="' + next + '"]');
      if (sel) sel.focus();
    };
    return b;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
  else build();

  // Same cross-document sync as theme-boot: a switch in another tab lands here.
  window.addEventListener('storage', function (e) {
    if (e.key !== KEY) return;
    var next = MODES.indexOf(e.newValue) >= 0 ? e.newValue : 'plain';
    if (next === mode) return;
    mode = next;
    apply();
  });

  window.vzMode = { set: set, mode: function () { return mode; }, fromUrl: fromUrl };
})();
