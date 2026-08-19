// theme-boot.js -- light/dark/auto for the pages that are NOT ES modules.
//
// catalogue.html and tour.html are plain HTML that must open over file://, so
// they cannot import framework/theme.js. They pair this classic script with the
// generated framework/theme.css. Both halves agree with theme.js by contract:
// the same storage key, the same `data-vz-theme` attribute, the same three
// modes. The colours themselves are NOT duplicated here -- they live once in
// theme.js and reach this side through the generated stylesheet.
//
// Load it in <head>, BEFORE the body renders, so an explicit dark choice is
// applied in the same frame the page paints and there is no white flash.
(function () {
  var KEY = 'vz-theme';
  var MODES = ['light', 'dark', 'auto'];
  var mode = 'auto';

  function stored() {
    try {
      var q = new URLSearchParams(location.search).get('theme');
      if (MODES.indexOf(q) >= 0) return q;         // a deep link wins, unpersisted
    } catch (e) { /* no URL */ }
    try {
      var s = localStorage.getItem(KEY);
      if (MODES.indexOf(s) >= 0) return s;
    } catch (e) { /* no storage */ }
    return 'auto';
  }

  function apply() {
    var root = document.documentElement;
    // In auto we REMOVE the attribute rather than resolving it ourselves: the
    // stylesheet's prefers-color-scheme block then does the work, so the page
    // follows the OS live with no listener and no reload.
    if (mode === 'auto') root.removeAttribute('data-vz-theme');
    else root.setAttribute('data-vz-theme', mode);
    var btns = document.querySelectorAll('.vz-theme-btn');
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute('aria-pressed', String(btns[i].getAttribute('data-mode') === mode));
    }
  }

  function set(next) {
    mode = MODES.indexOf(next) >= 0 ? next : 'auto';
    try { localStorage.setItem(KEY, mode); } catch (e) { /* private mode */ }
    apply();
  }

  mode = stored();
  apply();

  // Fill any <span class="vz-theme" data-theme-switch> with the 3-state switch
  // once the DOM exists.
  function build() {
    var hosts = document.querySelectorAll('[data-theme-switch]');
    for (var h = 0; h < hosts.length; h++) {
      var host = hosts[h];
      if (host.dataset.built) continue;
      host.dataset.built = '1';
      host.className = (host.className ? host.className + ' ' : '') + 'vz-theme';
      host.setAttribute('role', 'group');
      host.setAttribute('aria-label', 'colour theme');
      var glyph = { light: '☀', dark: '☾', auto: 'A' };
      var title = { light: 'light theme', dark: 'dark theme', auto: 'follow the system theme' };
      for (var i = 0; i < MODES.length; i++) {
        var m = MODES[i];
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'vz-theme-btn';
        b.setAttribute('data-mode', m);
        b.textContent = glyph[m];
        b.title = title[m];
        b.setAttribute('aria-label', title[m]);
        b.onclick = (function (mm) { return function () { set(mm); }; })(m);
        host.appendChild(b);
      }
    }
    apply();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
  else build();

  // Same cross-document sync as theme.js: a switch in another tab (or in the
  // parent page of an embedded demo) lands here too.
  window.addEventListener('storage', function (e) {
    if (e.key !== KEY) return;
    var next = MODES.indexOf(e.newValue) >= 0 ? e.newValue : 'auto';
    if (next === mode) return;
    mode = next;
    apply();
  });

  window.vzTheme = { set: set, mode: function () { return mode; } };
})();
