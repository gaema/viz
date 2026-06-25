// tokenization concept page -- Byte-Pair Encoding (BPE), the step that turns text
// into the integer token IDs a model consumes. BPE starts from characters and
// repeatedly MERGES the most frequent adjacent pair into a new symbol, growing a
// vocabulary and shrinking the token count. Watch the merges form: each step
// finds the top adjacent pair across the corpus (weighted by word frequency),
// adds a merge rule, and rewrites every word. This is how GPT/LLaMA tokenizers
// are trained; at inference the learned merges are replayed on new text.
import { mount } from '../framework/layout.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df', RED = '#d1242f';
const PRESETS = {
  'low / new / wide': [['low', 6], ['lower', 3], ['lowest', 2], ['newer', 4], ['newest', 6], ['wide', 3], ['wider', 3], ['widest', 2], ['new', 5]],
  'banana / panama': [['banana', 6], ['panama', 3], ['bandana', 3], ['ananas', 2], ['canada', 2]],
};
const MAXM = 12;

// Corpus source: typed text (alpha words → frequency = #occurrences, capped at
// 12 distinct words) when the text box is non-empty, else the chosen preset.
function corpusFrom(st) {
  const txt = (st.text || '').toLowerCase().trim();
  if (txt) { const m = new Map(); for (const w of txt.split(/\s+/).filter((w) => /^[a-z]+$/.test(w)).slice(0, 60)) m.set(w, (m.get(w) || 0) + 1); const arr = [...m.entries()].slice(0, 12); if (arr.length) return arr; }
  return PRESETS[st.corpus];
}

let cur = null, freqState = null, fsKey = '', wordRects = null, dragW = -1, lastY = 0;

function build(corpus) {
  const words = corpus.map(([w, f]) => ({ sym: [...w, '_'], freq: f }));
  const charCount = words.reduce((a, w) => a + w.sym.length * w.freq, 0);
  const pcOf = () => { const m = new Map(); for (const w of words) for (let i = 0; i < w.sym.length - 1; i++) { const k = w.sym[i] + '' + w.sym[i + 1]; m.set(k, (m.get(k) || 0) + w.freq); } return m; };
  const topOf = (m) => [...m.entries()].map(([k, v]) => { const p = k.split(''); return { a: p[0], b: p[1], v }; }).sort((x, y) => y.v - x.v);
  const vocabOf = () => { const s = new Set(); for (const w of words) for (const t of w.sym) s.add(t); return s; };
  const tokOf = () => words.reduce((a, w) => a + w.sym.length * w.freq, 0);
  const snap = (winner, rules, top) => ({ words: words.map((w) => ({ sym: w.sym.slice(), freq: w.freq })), top: top.slice(0, 7), winner, rules: rules.slice(), vocab: vocabOf().size, tokens: tokOf() });
  const steps = [], rules = [];
  steps.push(snap(null, [], topOf(pcOf())));
  for (let m = 0; m < MAXM; m++) {
    const top = topOf(pcOf()); if (!top.length) break; const win = top[0], nw = win.a + win.b;
    for (const w of words) { const out = []; for (let i = 0; i < w.sym.length; i++) { if (i < w.sym.length - 1 && w.sym[i] === win.a && w.sym[i + 1] === win.b) { out.push(nw); i++; } else out.push(w.sym[i]); } w.sym = out; }
    rules.push({ a: win.a, b: win.b, nw }); steps.push(snap(win, rules, top));
  }
  return { steps, charCount };
}

mount({
  mount: 'body',
  title: 'tokenization — Byte-Pair Encoding builds the vocab',
  blurb: 'Before a model sees text it must become integer token IDs, and the dominant scheme is Byte-Pair Encoding (BPE). It starts from raw characters and repeatedly does one thing: find the most frequent ADJACENT pair of symbols across the whole corpus (weighted by how often each word appears), merge that pair into a single new symbol, and add a merge rule. Frequent letter sequences ("e"+"s"→"es", "es"+"t"→"est") thus become single tokens, the vocabulary grows, and the number of tokens needed to write the corpus shrinks. That is exactly how GPT/LLaMA tokenizers are TRAINED; at inference the learned merge rules are simply replayed on new text. Step (or play) to watch the merges form; the pair table on the right shows the candidates with the winner highlighted; drag a word\'s frequency ↕ to change which pairs win; the merge rules and the vocab/token counts update live. The trailing "_" marks a word boundary.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  controls: (c, page) => {
    c.text('text', { label: 'corpus — type your own', value: '', placeholder: 'type words (repeats = frequency)…', rebuild: true });
    c.select('corpus', { label: 'preset (when text is blank)', options: Object.keys(PRESETS), value: 'low / new / wide', rebuild: true });
    c.transport({
      compute: () => {
        const base = corpusFrom(page.state);
        const key = (page.state.text || '').trim() ? 'txt:' + page.state.text.trim().toLowerCase() : page.state.corpus;
        if (fsKey !== key) { freqState = base.map((b) => b[1]); fsKey = key; }
        cur = build(base.map((b, i) => [b[0], freqState[i]]));
        return cur.steps.map((s, i) => ({ stage: i, label: i === 0 ? 'characters' : `merge ${i}: "${s.winner.a}"+"${s.winner.b}"→"${s.winner.nw || s.winner.a + s.winner.b}"` }));
      }, loop: true, speed: 1.3,
    });
  },
  autoplay: true,
  onPointer: (page, ev) => {
    if (!wordRects) return;
    const at = (x, y) => { for (let i = 0; i < wordRects.length; i++) { const r = wordRects[i]; if (x >= r.x - 4 && x <= r.x + 34 && y >= r.y && y <= r.y + r.h) return i; } return -1; };
    if (ev.type === 'down') { dragW = at(ev.x, ev.y); lastY = ev.y; }
    else if (ev.type === 'up' || ev.type === 'leave') dragW = -1;
    else if (ev.type === 'move' && dragW >= 0 && page.pointer.down) { const d = Math.round((lastY - ev.y) / 10); if (d && freqState) { freqState[dragW] = Math.max(1, Math.min(12, freqState[dragW] + d)); lastY = ev.y; if (page.controls._transport) page.controls._transport._dirty = true; page.redraw(); } }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, Hh = page.H;
    if (page.controls._transport) page.controls._transport.rebuildIfDirty();
    if (!cur) return;
    r.clear('#ffffff');
    const cs = page.step(), k = cs ? cs.stage : 0, S = cur.steps[k], nw = k > 0 ? S.rules[k - 1].nw : null;

    // ===== corpus words + segmentation (left) =====
    const lx = 20, ly = 64; let y = ly;
    r.label('corpus — each word as symbols (drag ×freq ↕)', lx, ly - 10, { color: INK, font: '11px ui-monospace, monospace' });
    wordRects = [];
    ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left';
    for (let wi = 0; wi < S.words.length; wi++) {
      const w = S.words[wi];
      ctx.fillStyle = wi === dragW ? 'rgba(210,105,30,0.15)' : '#f4f5f7'; ctx.fillRect(lx, y, 32, 20); ctx.strokeStyle = '#d0d7de'; ctx.strokeRect(lx, y, 32, 20); ctx.fillStyle = '#586069'; ctx.font = '10px ui-monospace, monospace'; ctx.fillText('×' + w.freq, lx + 5, y + 14);
      wordRects.push({ x: lx, y, h: 20 });
      let sx = lx + 40;
      for (const s of w.sym) { ctx.font = '12px ui-monospace, monospace'; const tw = ctx.measureText(s).width + 8, isNew = s === nw && s.length > 1; ctx.fillStyle = isNew ? 'rgba(210,105,30,0.22)' : '#eef2f6'; ctx.fillRect(sx, y, tw, 20); ctx.strokeStyle = isNew ? ORANGE : '#cdd5dd'; ctx.lineWidth = isNew ? 1.4 : 0.8; ctx.strokeRect(sx, y, tw, 20); ctx.fillStyle = isNew ? ORANGE : INK; ctx.fillText(s, sx + 4, y + 14); sx += tw + 3; }
      y += 24;
    }
    const botY = y + 10;
    // counts
    const comp = (1 - S.tokens / cur.charCount) * 100;
    r.label(`vocab ${S.vocab} symbols  ·  ${S.tokens} tokens (was ${cur.charCount}, −${comp.toFixed(0)}%)`, lx, botY, { color: GREEN, font: '11px ui-monospace, monospace' });
    r.label(`merges ${k} / ${cur.steps.length - 1}  ·  "_" = word boundary`, lx, botY + 16, { color: '#586069', font: '10px ui-monospace, monospace' });

    // ===== pair frequency table (right top) =====
    const px = 430, py = 64, pw = W - px - 16;
    r.label(k === 0 ? 'most frequent adjacent pairs (next to merge →)' : 'pair frequencies at this step (winner merged)', px, py - 10, { color: INK, font: '11px ui-monospace, monospace' });
    const mx = S.top.length ? S.top[0].v : 1;
    S.top.forEach((p, i) => {
      const yy = py + i * 20, isWin = i === 0 && (k > 0 || true);
      ctx.fillStyle = (k > 0 && i === 0) ? ORANGE : 'rgba(31,111,235,0.5)'; ctx.fillRect(px + 86, yy, (pw - 110) * p.v / mx, 14);
      ctx.fillStyle = INK; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.fillText(`"${p.a}"+"${p.b}"`, px, yy + 12);
      ctx.fillStyle = '#586069'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.fillText('' + p.v, px + pw, yy + 12); ctx.textAlign = 'left';
    });

    // ===== merge rules learned (right bottom) =====
    const ry = py + Math.max(S.top.length, 4) * 20 + 24;
    r.label('merge rules learned (in order)', px, ry - 4, { color: INK, font: '11px ui-monospace, monospace' });
    if (!S.rules.length) r.label('(none yet — start stepping)', px, ry + 14, { color: '#8a939b', font: '10px ui-monospace, monospace' });
    S.rules.forEach((ru, i) => { const yy = ry + 12 + i * 16, isLast = i === S.rules.length - 1; r.label(`${i + 1}.  "${ru.a}" + "${ru.b}"  →  "${ru.nw}"`, px, yy, { color: isLast ? ORANGE : '#3a4047', font: (isLast ? 'bold ' : '') + '10px ui-monospace, monospace' }); });

    // hover
    if (page.pointer.over && dragW < 0) {
      const p = page.pointer;
      if (p.x >= px && p.x <= px + pw && p.y >= py - 4 && p.y <= py + S.top.length * 20) { const i = Math.floor((p.y - py) / 20); if (i >= 0 && i < S.top.length) { const pr = S.top[i]; page.setTip(`pair "${pr.a}"+"${pr.b}"\nappears ${pr.v}× across the corpus${i === 0 && k < cur.steps.length - 1 ? '\n← the next merge' : ''}`); } }
    }

    let o = `BPE tokenization · step ${k}/${cur.steps.length - 1}.  ${k === 0 ? 'start: every word is a sequence of characters (+ "_" boundary).' : `merged "${S.rules[k - 1].a}"+"${S.rules[k - 1].b}" → "${S.rules[k - 1].nw}" (the most frequent adjacent pair).`}   tier:${r.name}\n`;
    o += `vocab ${S.vocab} symbols, ${S.tokens} tokens encode the corpus (${cur.charCount} chars, ${comp.toFixed(0)}% fewer). The learned rules are replayed on new text at inference; drag a word's ×freq to change which pairs win.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__tkPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('corpus')) page.controls.set('corpus', q.get('corpus'));
  if (q.has('text')) page.controls.set('text', q.get('text'), { rebuild: true });
  if (q.has('step') && page.controls._transport) page.controls._transport.seek(+q.get('step'));
  page.redraw();
});
