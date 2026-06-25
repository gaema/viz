// forward-pass concept page -- the CAPSTONE: one short prompt run end-to-end
// through a tiny transformer to predict the next token, with the SAME residual
// stream visibly transforming at every stage. Chains the atomic demos:
//   tokenize -> embed -> attention (+residual) -> gated MLP (+residual)
//   -> final-norm + lm-head -> softmax sample -> next token.
// The model is a real (if tiny, randomly-initialised) transformer: d=8, 1 layer,
// 2 heads, gated MLP, tied lm-head. Weights are random, so the PREDICTION is
// arbitrary -- the point is to watch the MECHANISM, the data flowing through.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df', RED = '#d1242f';
const VOCAB = ['the', 'a', 'cat', 'dog', 'sat', 'ran', 'on', 'to', 'mat', 'park', 'big', 'red', 'quick', 'lazy', 'fox', 'jumped'];
// Type any prompt: split on spaces, lowercase, cap at 8 words. Words in the toy
// vocab use their index; unseen words map to a token by hashing (so arbitrary
// text still tokenizes — the mechanism is the point, not a trained vocab).
const hashTok = (s) => { let h = 5; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0; return h; };
function parsePrompt(s) { const w = (s || '').toLowerCase().trim().split(/\s+/).filter(Boolean).slice(0, 8); return w.length ? w : ['the']; }
const STAGES = ['tokenize', 'embed', 'attention', 'MLP', 'lm-head', 'sample'];
const D = 8, H = 2, DH = 4, DFF = 16, V = VOCAB.length;

const mk = (seed, r, c, std) => ({ d: seededRandn(seed, [r, c], { std }).data, r, c });
const mm = (A, B) => { const r = A.r, c = B.c, k = A.c, out = new Float32Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { let s = 0; for (let t = 0; t < k; t++) s += A.d[i * k + t] * B.d[t * c + j]; out[i * c + j] = s; } return { d: out, r, c }; };
const lnorm = (M) => { const out = new Float32Array(M.d.length); for (let i = 0; i < M.r; i++) { let m = 0; for (let j = 0; j < M.c; j++) m += M.d[i * M.c + j]; m /= M.c; let v = 0; for (let j = 0; j < M.c; j++) v += (M.d[i * M.c + j] - m) ** 2; v = Math.sqrt(v / M.c + 1e-5); for (let j = 0; j < M.c; j++) out[i * M.c + j] = (M.d[i * M.c + j] - m) / v; } return { d: out, r: M.r, c: M.c }; };
const add = (A, B) => ({ d: Float32Array.from(A.d, (x, i) => x + B.d[i]), r: A.r, c: A.c });
const silu = (x) => x / (1 + Math.exp(-x));
const softmax = (a) => { let m = -1e9; for (const x of a) if (x > m) m = x; let s = 0; const o = a.map((x) => { const e = Math.exp(x - m); s += e; return e; }); return o.map((e) => e / s); };

let cur = null, bsig = '', tokRects = null, dragTok = -1;

function build(words, seed, temp, headSel) {
  const tok = words.map((w) => { const i = VOCAB.indexOf(w); return i >= 0 ? i : hashTok(w) % V; });
  const n = tok.length;
  const E = mk(seed + 1, V, D, 0.9), Wq = mk(seed + 2, D, D, 0.5), Wk = mk(seed + 3, D, D, 0.5), Wv = mk(seed + 4, D, D, 0.5), Wo = mk(seed + 5, D, D, 0.5);
  const Wg = mk(seed + 6, D, DFF, 0.4), Wu = mk(seed + 7, D, DFF, 0.4), Wd = mk(seed + 8, DFF, D, 0.4);
  // embed (+ sinusoidal position)
  const X0d = new Float32Array(n * D);
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) X0d[i * D + j] = E.d[tok[i] * D + j] + 0.3 * Math.sin(i / Math.pow(1000, j / D) + (j % 2) * 1.57);
  const X0 = { d: X0d, r: n, c: D };
  // attention block
  const Xn = lnorm(X0), Q = mm(Xn, Wq), K = mm(Xn, Wk), Vv = mm(Xn, Wv);
  const attn = [];           // per head [n×n]
  const ctx = new Float32Array(n * D);
  for (let h = 0; h < H; h++) {
    const A = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      const row = []; for (let j = 0; j < n; j++) { if (j > i) { row.push(-1e9); continue; } let s = 0; for (let t = 0; t < DH; t++) s += Q.d[i * D + h * DH + t] * K.d[j * D + h * DH + t]; row.push(s / Math.sqrt(DH)); }
      const p = softmax(row); for (let j = 0; j < n; j++) A[i * n + j] = j > i ? 0 : p[j];
      for (let t = 0; t < DH; t++) { let acc = 0; for (let j = 0; j <= i; j++) acc += A[i * n + j] * Vv.d[j * D + h * DH + t]; ctx[i * D + h * DH + t] = acc; }
    }
    attn.push({ d: A, r: n, c: n });
  }
  const attnOut = mm({ d: ctx, r: n, c: D }, Wo), X1 = add(X0, attnOut);
  // gated MLP block
  const Xn2 = lnorm(X1), g = mm(Xn2, Wg), u = mm(Xn2, Wu);
  const hid = new Float32Array(n * DFF); for (let i = 0; i < n * DFF; i++) hid[i] = silu(g.d[i]) * u.d[i];
  const mlp = mm({ d: hid, r: n, c: DFF }, Wd), X2 = add(X1, mlp);
  // lm-head on the LAST token (tied: logits = h · E^T)
  const hLast = lnorm(X2); const last = n - 1, logits = new Float32Array(V);
  for (let w = 0; w < V; w++) { let s = 0; for (let j = 0; j < D; j++) s += hLast.d[last * D + j] * E.d[w * D + j]; logits[w] = s; }
  const probs = softmax(Array.from(logits, (x) => x / Math.max(0.05, temp)));
  let pred = 0; for (let w = 1; w < V; w++) if (probs[w] > probs[pred]) pred = w;
  return { tok, n, X0, X1, X2, attn, logits, probs, pred, hidLast: { g: g.d.slice(last * DFF, last * DFF + DFF), u: u.d.slice(last * DFF, last * DFF + DFF), hid: hid.slice(last * DFF, last * DFF + DFF) } };
}

// small heatmap of a [r×c] matrix
function heat(ctx, x, y, M, cw, chh, dom) { for (let i = 0; i < M.r; i++) for (let j = 0; j < M.c; j++) { const v = M.d[i * M.c + j]; const t = Math.max(-1, Math.min(1, v / dom)), m = Math.abs(t); ctx.fillStyle = t >= 0 ? `rgb(255,${Math.round(255 - m * 150)},${Math.round(255 - m * 165)})` : `rgb(${Math.round(255 - m * 165)},${Math.round(255 - m * 120)},255)`; ctx.fillRect(x + j * cw, y + i * chh, cw - 0.5, chh - 0.5); } }

mount({
  mount: 'body',
  title: 'forward-pass — one prompt → the next token, end to end',
  blurb: 'The capstone: every other page shows one mechanism in isolation; this one chains them. A short prompt is run all the way through a tiny but real transformer (d=8, 1 layer, 2 heads, gated MLP, tied lm-head) to predict the next token, and you watch the SAME residual stream transform at each stage: tokenize → embed (token + position) → attention (+ residual) → gated MLP (+ residual) → final-norm + lm-head → softmax → sampled token. The residual-stream strip on the left is the through-line — every block reads it, computes a correction, and adds the correction back. The weights are random, so the predicted word is arbitrary (this is about the mechanism, not a trained model). Step the transport (or play) to walk the stages; TYPE YOUR OWN PROMPT in the box (a toy 16-word vocab — words it has not seen are mapped to a token by hashing, so any text still flows through); drag a prompt token to swap its word and watch the whole pipeline recompute; tune temperature and the attention head; hover any tensor.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  challenges: [
    { goal: 'Make the prediction confident — top token probability ≥ 40%.', hint: 'lower the temperature toward 0.1 (try a few seeds/prompts).', check: (api) => ({ solved: (api.probe.topP ?? 0) >= 0.4, detail: `top token p = ${((api.probe.topP ?? 0) * 100).toFixed(0)}% (need ≥ 40%)` }) },
    { goal: 'Walk the whole pipeline — reach the final "sample" stage.', hint: 'step the transport (or play) to stage 6.', check: (api) => ({ solved: (api.probe.k ?? 0) === 5, detail: `at stage ${(api.probe.k ?? 0) + 1} / 6` }) },
  ],
  controls: (c, page) => {
    c.text('prompt', { label: 'prompt — type your own', value: 'the cat sat on the', placeholder: 'type a few words…' });
    c.slider('temp', { label: 'temperature', min: 0.1, max: 2, step: 0.1, value: 0.8 });
    c.slider('head', { label: 'attn head', min: 0, max: H - 1, step: 1, value: 0 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => STAGES.map((s, i) => ({ stage: i, label: `${i + 1} · ${s}` })), loop: true, speed: 1.1 });
  },
  autoplay: true,
  onPointer: (page, ev) => {
    if (!cur || !tokRects) return;
    const at = (x, y) => { for (let i = 0; i < tokRects.length; i++) { const r = tokRects[i]; if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return i; } return -1; };
    if (ev.type === 'down') { dragTok = at(ev.x, ev.y); }
    else if (ev.type === 'up' || ev.type === 'leave') dragTok = -1;
    else if (ev.type === 'move' && dragTok >= 0 && page.pointer.down) { const step = ev.dy < -4 ? 1 : ev.dy > 4 ? -1 : 0; if (step) { const words = cur._words.slice(); words[dragTok] = VOCAB[(VOCAB.indexOf(words[dragTok]) + step + V) % V]; cur = build(words, page.state.seed | 0, page.state.temp, page.state.head | 0); cur._words = words; bsig = `manual`; page.redraw(); } }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, Hh = page.H;
    const sig = `${st.prompt}|${st.seed}|${st.temp}|${st.head}`;
    if (sig !== bsig && bsig !== 'manual') { const words = parsePrompt(st.prompt); cur = build(words, st.seed | 0, st.temp, st.head | 0); cur._words = words.slice(); bsig = sig; }
    else if (bsig === 'manual' && (st.prompt + st.seed) !== cur._lock) { /* keep manual edits until a control changes */ }
    r.clear('#ffffff');
    const cs = page.step(), k = cs ? cs.stage : 5;          // current stage (default: final)
    page.probe = { topP: cur.probs[cur.pred], k };
    const n = cur.n, head = st.head | 0;

    // ===== stage ribbon =====
    const rbY = 36, chipW = (W - 40) / STAGES.length;
    STAGES.forEach((s, i) => {
      const x = 20 + i * chipW, on = i === k, done = i < k;
      ctx.save(); ctx.fillStyle = on ? 'rgba(31,111,235,0.15)' : done ? 'rgba(44,160,44,0.08)' : '#f4f5f7'; ctx.fillRect(x + 3, rbY, chipW - 12, 22); ctx.strokeStyle = on ? BLUE : done ? GREEN : '#d0d7de'; ctx.lineWidth = on ? 1.6 : 1; ctx.strokeRect(x + 3, rbY, chipW - 12, 22); ctx.fillStyle = on ? BLUE : done ? GREEN : '#586069'; ctx.font = (on ? 'bold ' : '') + '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(`${i + 1}·${s}`, x + 3 + (chipW - 12) / 2, rbY + 15); if (i < STAGES.length - 1) { ctx.strokeStyle = GREY; ctx.beginPath(); ctx.moveTo(x + chipW - 8, rbY + 11); ctx.lineTo(x + chipW + 1, rbY + 11); ctx.stroke(); } ctx.restore();
    });

    // ===== residual stream (left, persistent) =====
    const Xshow = k <= 1 ? cur.X0 : k === 2 ? cur.X1 : cur.X2, sLabel = k <= 1 ? 'X₀ (embeddings)' : k === 2 ? 'X₁ (+attention)' : 'X₂ (+MLP)';
    const lx = 20, lyTok = 86, tokH = 17;
    r.label('prompt — drag a token ↕', lx, lyTok - 8, { color: INK, font: '10px ui-monospace, monospace' });
    tokRects = [];
    for (let i = 0; i < n; i++) { const y = lyTok + i * (tokH + 3); ctx.save(); ctx.fillStyle = i === dragTok ? 'rgba(210,105,30,0.18)' : '#f2f4f6'; ctx.fillRect(lx, y, 96, tokH); ctx.strokeStyle = i === n - 1 ? PURPLE : '#d0d7de'; ctx.lineWidth = i === n - 1 ? 1.5 : 1; ctx.strokeRect(lx, y, 96, tokH); ctx.fillStyle = INK; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.fillText(`${cur.tok[i]}`.padStart(2) + ' ' + cur._words[i], lx + 4, y + 12); ctx.restore(); tokRects.push({ x: lx, y, w: 96, h: tokH }); }
    r.label('"' + cur._words[n - 1] + '" = last → predicts next', lx, lyTok + n * (tokH + 3) + 6, { color: PURPLE, font: '8px ui-monospace, monospace' });
    // residual stream heatmap
    const rsY = lyTok + n * (tokH + 3) + 28, rcw = 13, rch = 13;
    r.label(sLabel + '  [' + n + '×' + D + ']', lx, rsY - 6, { color: BLUE, font: '10px ui-monospace, monospace' });
    if (k >= 1) heat(ctx, lx, rsY, Xshow, rcw, rch, 2.2);
    else { ctx.save(); ctx.strokeStyle = '#e6e8ea'; ctx.strokeRect(lx, rsY, D * rcw, n * rch); ctx.fillStyle = '#b8bec4'; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('(not embedded yet)', lx + D * rcw / 2, rsY + n * rch / 2); ctx.restore(); }
    r.label('rows = tokens, cols = d=8 channels', lx, rsY + n * rch + 12, { color: '#8a939b', font: '8px ui-monospace, monospace' });

    // ===== stage detail (center/right) =====
    const dx = 200, dy = 80, dw = W - dx - 16;
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left';
    if (k === 0) {
      r.label('tokenize: prompt words → integer IDs (indices into a vocab of ' + V + ')', dx, dy, { color: INK, font: '11px ui-monospace, monospace' });
      for (let i = 0; i < n; i++) { const yy = dy + 22 + i * 22; ctx.fillStyle = '#f2f4f6'; ctx.fillRect(dx, yy, 90, 18); ctx.strokeStyle = '#d0d7de'; ctx.strokeRect(dx, yy, 90, 18); ctx.fillStyle = INK; ctx.fillText('"' + cur._words[i] + '"', dx + 6, yy + 13); ctx.strokeStyle = GREY; ctx.beginPath(); ctx.moveTo(dx + 94, yy + 9); ctx.lineTo(dx + 120, yy + 9); ctx.stroke(); ctx.fillStyle = PURPLE; ctx.fillText('id ' + cur.tok[i], dx + 126, yy + 13); }
      r.label('each id will index one row of the embedding table next →', dx, dy + 22 + n * 22 + 6, { color: '#586069', font: '9px ui-monospace, monospace' });
    } else if (k === 1) {
      r.label('embed: X₀[i] = E[id[i]] + position[i]   — a lookup + a positional offset', dx, dy, { color: INK, font: '11px ui-monospace, monospace' });
      r.label('the ' + n + ' selected rows of the embedding table E [' + V + '×' + D + '] become the residual stream (left).', dx, dy + 20, { color: '#586069', font: '10px ui-monospace, monospace' });
      heat(ctx, dx, dy + 34, cur.X0, 18, 16, 2.2);
      r.label('X₀  (this is what every later block reads + writes back)', dx, dy + 34 + n * 16 + 12, { color: BLUE, font: '9px ui-monospace, monospace' });
    } else if (k === 2) {
      r.label('attention (head ' + head + '): scores = QKᵀ/√dₕ → causal softmax → weighted sum of V; then X₁ = X₀ + Wₒ·ctx', dx, dy, { color: INK, font: '10px ui-monospace, monospace' });
      const A = cur.attn[head], ac = Math.min(28, 150 / n);
      heat(ctx, dx, dy + 18, A, ac, ac, 1);
      ctx.strokeStyle = '#e6e8ea'; ctx.strokeRect(dx, dy + 18, n * ac, n * ac);
      r.label('attention weights [' + n + '×' + n + '] — row i attends to cols j≤i (causal)', dx, dy + 18 + n * ac + 12, { color: ORANGE, font: '9px ui-monospace, monospace' });
      r.label('result added back into the stream → X₁ (left now shows X₁)', dx, dy + 18 + n * ac + 26, { color: '#586069', font: '9px ui-monospace, monospace' });
    } else if (k === 3) {
      r.label('gated MLP (last token): down( silu(x·W_g) ⊙ (x·W_u) ); then X₂ = X₁ + mlp', dx, dy, { color: INK, font: '10px ui-monospace, monospace' });
      const g = cur.hidLast.g, u = cur.hidLast.u, hh = cur.hidLast.hid, bw = (dw - 10) / DFF;
      const barRow = (vals, yy, col, lab) => { r.label(lab, dx, yy - 2, { color: col, font: '9px ui-monospace, monospace' }); for (let j = 0; j < DFF; j++) { const v = Math.max(-2, Math.min(2, vals[j])), bh = Math.abs(v) / 2 * 22; ctx.fillStyle = col; ctx.fillRect(dx + j * bw, v >= 0 ? yy + 22 - bh : yy + 22, bw - 1, bh); } };
      barRow(g.map(silu), dy + 30, GREEN, 'silu(gate)  [' + DFF + ' hidden]');
      barRow(u, dy + 84, BLUE, 'up');
      barRow(hh, dy + 138, PURPLE, 'silu(gate) ⊙ up  → down-projected back to d=8, added to X₂');
    } else if (k === 4) {
      r.label('lm-head: take the LAST token\'s hidden, project onto every vocab row → ' + V + ' logits (tied: logits = h·Eᵀ)', dx, dy, { color: INK, font: '10px ui-monospace, monospace' });
      const bw = (dw - 4) / V, mx = Math.max(...Array.from(cur.logits, Math.abs)) || 1;
      for (let w = 0; w < V; w++) { const v = cur.logits[w], bh = Math.abs(v) / mx * 70, yy = dy + 100; ctx.fillStyle = w === cur.pred ? RED : BLUE; ctx.fillRect(dx + w * bw, v >= 0 ? yy - bh : yy, bw - 1.5, bh); ctx.save(); ctx.translate(dx + w * bw + bw / 2, yy + 12); ctx.rotate(-Math.PI / 2.6); ctx.fillStyle = '#586069'; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.fillText(VOCAB[w], 0, 0); ctx.restore(); }
      r.label('one logit per vocab word (higher = more likely next). Softmax turns these into probabilities →', dx, dy + 150, { color: '#586069', font: '9px ui-monospace, monospace' });
    } else {
      r.label('sample: probs = softmax(logits / T=' + st.temp.toFixed(1) + ') → pick the next token', dx, dy, { color: INK, font: '11px ui-monospace, monospace' });
      const bw = (dw - 4) / V, mx = Math.max(...cur.probs);
      for (let w = 0; w < V; w++) { const bh = cur.probs[w] / mx * 80, yy = dy + 110; ctx.fillStyle = w === cur.pred ? RED : 'rgba(31,111,235,0.55)'; ctx.fillRect(dx + w * bw, yy - bh, bw - 1.5, bh); ctx.save(); ctx.translate(dx + w * bw + bw / 2, yy + 12); ctx.rotate(-Math.PI / 2.6); ctx.fillStyle = w === cur.pred ? RED : '#586069'; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.fillText(VOCAB[w], 0, 0); ctx.restore(); }
      ctx.fillStyle = INK; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText('next token →', dx, dy + 150); ctx.fillStyle = RED; ctx.font = 'bold 13px ui-monospace, monospace'; ctx.fillText('"' + VOCAB[cur.pred] + '"  (p=' + (cur.probs[cur.pred] * 100).toFixed(0) + '%)', dx + 90, dy + 150);
      r.label(cur._words.join(' ') + '  →  ' + VOCAB[cur.pred], dx, dy + 172, { color: '#586069', font: '10px ui-monospace, monospace' });
    }
    ctx.restore();

    // hover
    if (page.pointer.over && dragTok < 0) {
      const p = page.pointer; const rs = lyTok + n * (tokH + 3) + 18;
      if (k >= 1 && p.x >= lx && p.x <= lx + D * rcw && p.y >= rs && p.y <= rs + n * rch) { const i = Math.floor((p.y - rs) / rch), j = Math.floor((p.x - lx) / rcw); if (i < n && j < D) page.setTip(`${sLabel}\nrow ${i} ("${cur._words[i]}"), channel ${j}\n= ${Xshow.d[i * D + j].toFixed(3)}`); }
    }

    let o = `forward pass · stage ${k + 1}/6: ${STAGES[k]}.  prompt "${cur._words.join(' ')}" → toy transformer (d=${D}, 1 layer, ${H} heads) → next token "${VOCAB[cur.pred]}".   tier:${r.name}\n`;
    o += k === 0 ? 'words become integer IDs (vocab indices).' : k === 1 ? 'each ID looks up a row of E and adds a positional offset → the residual stream X₀.' : k === 2 ? `attention head ${head} mixes tokens (causal), its output is ADDED back: X₁ = X₀ + attnOut.` : k === 3 ? 'the gated MLP transforms each token independently, ADDED back: X₂ = X₁ + mlp.' : k === 4 ? 'the last token\'s final hidden is projected onto every vocab embedding → logits.' : `softmax(logits/${st.temp.toFixed(1)}) → "${VOCAB[cur.pred]}" (${(cur.probs[cur.pred] * 100).toFixed(0)}%). Weights are random, so the word is arbitrary — the mechanism is the point. Drag a token or step the stages to explore.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__fpPage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['prompt']) if (q.has(key)) page.controls.set(key, q.get(key));
  for (const key of ['temp', 'head', 'seed']) if (q.has(key)) page.controls.set(key, +q.get(key));
  if (q.has('step') && page.controls._transport) page.controls._transport.seek(+q.get('step'));
  page.redraw();
});
