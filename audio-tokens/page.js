// audio-tokens concept page -- how a waveform becomes something a language model
// can predict: residual vector quantization (RVQ) of a low-frame-rate encoder.
//
// The mechanism, in three moves:
//   1. ENCODE. An encoder downsamples the waveform to a low frame rate. Each
//      frame is a D-dimensional vector, not a sample -- so a second of audio is
//      tens of vectors instead of tens of thousands of samples.
//   2. RESIDUAL-QUANTIZE. Codebook 1 replaces each frame vector with its nearest
//      codeword. What codebook 1 got WRONG -- the residual -- is quantized by
//      codebook 2; that residual by codebook 3, and so on. One frame becomes a
//      small stack of integer codes, coarse to fine. Dropping the deepest
//      codebooks degrades quality gracefully instead of destroying it.
//   3. DECODE. Sum the chosen codewords, run the synthesis filter, get audio back.
//
// The trade: bitrate = frames/sec x codebooks x log2(codebook size). Each added
// codebook costs a FIXED number of bits per second and buys a SHRINKING error
// reduction -- the staircase on the right. The second trade is frame rate against
// sequence length: every frame is a token position a language model has to
// predict, so 75 frames/s x 8 codebooks is 36,000 tokens per minute of audio.
//
// Everything on screen is computed in this file: the waveform, the encoder, the
// codebooks (fit by a few Lloyd iterations on a synthetic corpus), the code
// assignment, the reconstruction and every error number.
//
// Sources for the mechanism: SoundStream, https://arxiv.org/abs/2107.03312 ;
// EnCodec, https://arxiv.org/abs/2210.13438 ; Moshi/Mimi (the 12.5 fps,
// semantic-distillation variant), https://arxiv.org/abs/2410.00037 .
import { mount } from '../framework/layout.js';
import { T, alphaOf } from '../framework/theme.js';

const DUR = 0.8;      // seconds of audio in the clip
const NS = 1920;      // waveform samples drawn across the clip
const D = 8;          // encoder channels per frame (the latent dimension)
const MAXQ = 8;       // codebooks trained (the slider selects how many are USED)
const CORPUS = 1024;  // latent vectors the codebooks are fit on
const ITERS = 4;      // Lloyd iterations per codebook
const CTX = 32768;    // the text context length we compare sequence length against
const WORDS_PER_TOKEN = 0.75;  // illustrative English text ratio
const WORDS_PER_MIN = 150;     // illustrative speaking rate

const STAGE_COL = () => [T.accent, T.violet, T.teal, T.warn, T.ok, T.goldDeep, T.bad, T.tealDeep];

// ---------------------------------------------------------------- rng + signal
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

// A couple of sine components under an envelope (or a chirp). `bumps` are the
// user's drags: localized Gaussian pushes into the signal.
function signal(preset, seed, bumps) {
  const r = rng(seed * 7919 + 13);
  const f1 = 10 + Math.round(r() * 8), f2 = 19 + Math.round(r() * 10), f3 = 31 + Math.round(r() * 14);
  const ph = r() * 6.283;
  const x = new Float32Array(NS);
  for (let i = 0; i < NS; i++) {
    const t = i / NS * DUR, u = i / NS;
    let v;
    if (preset === 'chirp') {
      const fInst = 6 + 54 * u;                       // 6 -> 60 Hz sweep
      v = Math.sin(2 * Math.PI * (6 * t + 27 * t * t / DUR)) * 1.0 + 0.25 * Math.sin(2 * Math.PI * fInst * t + ph);
      v *= Math.pow(Math.sin(Math.PI * u), 0.5);
    } else if (preset === 'burst') {
      v = Math.sin(2 * Math.PI * f1 * t) + 0.7 * Math.sin(2 * Math.PI * f3 * t + ph);
      v *= Math.exp(-Math.pow((u - 0.36) / 0.10, 2)) + 0.55 * Math.exp(-Math.pow((u - 0.72) / 0.06, 2));
    } else {                                          // 'two tones'
      v = Math.sin(2 * Math.PI * f1 * t) + 0.6 * Math.sin(2 * Math.PI * f2 * t + ph) + 0.3 * Math.sin(2 * Math.PI * f3 * t);
      v *= Math.pow(Math.sin(Math.PI * u), 0.6);
    }
    x[i] = v;
  }
  for (const b of bumps) for (let i = 0; i < NS; i++) { const u = i / NS; x[i] += b.a * Math.exp(-Math.pow((u - b.x) / 0.022, 2)); }
  let mx = 1e-6; for (let i = 0; i < NS; i++) mx = Math.max(mx, Math.abs(x[i]));
  for (let i = 0; i < NS; i++) x[i] /= mx;
  return x;
}

// ------------------------------------------------------------ encoder/decoder
// Encoder: box-average each frame's samples into D channels -> one vector/frame.
function encode(x, F, frameLen, binLen) {
  const lat = new Float32Array(F * D);
  for (let f = 0; f < F; f++) for (let d = 0; d < D; d++) {
    let s = 0, i0 = f * frameLen + d * binLen;
    for (let i = 0; i < binLen; i++) s += x[i0 + i];
    lat[f * D + d] = s / binLen;
  }
  return lat;
}
// Decoder: linear interpolation between channel centres -> samples again.
function decode(lat, F, frameLen, binLen, out) {
  const M = F * D, half = (binLen - 1) / 2;
  const ctr = (m) => ((m / D) | 0) * frameLen + (m % D) * binLen + half;
  let m = 0;
  for (let i = 0; i < NS; i++) {
    while (m < M - 2 && ctr(m + 1) <= i) m++;
    const c0 = ctr(m), c1 = ctr(m + 1);
    if (i <= c0) out[i] = lat[m];
    else if (i >= ctr(M - 1)) out[i] = lat[M - 1];
    else { const t = (i - c0) / (c1 - c0); out[i] = lat[m] * (1 - t) + lat[m + 1] * t; }
  }
  return out;
}
function rms(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const e = a[i] - (b ? b[i] : 0); s += e * e; } return Math.sqrt(s / a.length); }

// ------------------------------------------------------------- codebook fitting
function nearest(pts, i, cb, K) {
  let best = 0, bd = Infinity, o = i * D;
  for (let j = 0; j < K; j++) {
    let d2 = 0, q = j * D;
    for (let k = 0; k < D; k++) { const e = pts[o + k] - cb[q + k]; d2 += e * e; }
    if (d2 < bd) { bd = d2; best = j; }
  }
  return best;
}
function lloyd(pts, n, K, seed) {
  const r = rng(seed), cb = new Float32Array(K * D);
  for (let j = 0; j < K; j++) { const i = Math.floor(r() * n); for (let k = 0; k < D; k++) cb[j * D + k] = pts[i * D + k]; }
  const sum = new Float64Array(K * D), cnt = new Int32Array(K);
  for (let it = 0; it < ITERS; it++) {
    sum.fill(0); cnt.fill(0);
    for (let i = 0; i < n; i++) { const j = nearest(pts, i, cb, K); cnt[j]++; for (let k = 0; k < D; k++) sum[j * D + k] += pts[i * D + k]; }
    for (let j = 0; j < K; j++) {
      if (cnt[j]) { for (let k = 0; k < D; k++) cb[j * D + k] = sum[j * D + k] / cnt[j]; }
      else { const i = Math.floor(r() * n); for (let k = 0; k < D; k++) cb[j * D + k] = pts[i * D + k]; }   // revive an empty cell
    }
  }
  return cb;
}

// The codebooks are fit on a SYNTHETIC CORPUS of other clips, never on the clip
// being coded -- fitting on the clip itself would let the codebook memorize it
// and the error would collapse for reasons a real codec never enjoys.
const bookCache = new Map();
function books(fps, K, seed, F, frameLen, binLen) {
  const key = `${fps}|${K}|${seed}`;
  if (bookCache.has(key)) return bookCache.get(key);
  const clips = Math.ceil(CORPUS / F), n = clips * F, pts = new Float32Array(n * D);
  const kinds = ['two tones', 'chirp', 'burst'];
  for (let c = 0; c < clips; c++) {
    const lat = encode(signal(kinds[c % 3], seed * 131 + c * 17 + 1, []), F, frameLen, binLen);
    pts.set(lat, c * F * D);
  }
  const res = Float32Array.from(pts), out = [];
  for (let s = 0; s < MAXQ; s++) {
    const cb = lloyd(res, n, K, seed * 31 + s * 977 + 5);
    for (let i = 0; i < n; i++) { const j = nearest(res, i, cb, K); for (let k = 0; k < D; k++) res[i * D + k] -= cb[j * D + k]; }
    out.push(cb);
  }
  bookCache.set(key, out);
  return out;
}

// ------------------------------------------------------------------- the run
let cur = null, bumps = [], wavRect = null, gridRect = null, dragging = false;

function buildAll(st) {
  const fps = +st.fps, nq = st.nq | 0, K = +st.K, seed = st.seed | 0;
  const F = Math.round(fps * DUR), frameLen = NS / F, binLen = frameLen / D;
  const x = signal(st.preset, seed, bumps);
  const lat = encode(x, F, frameLen, binLen);
  const bk = books(fps, K, seed, F, frameLen, binLen);

  // residual VQ of THIS clip against the fitted codebooks
  const res = Float32Array.from(lat);
  const codes = [], recon = [Float32Array.from(lat).fill(0)], resRms = [rms(lat)];
  for (let s = 0; s < nq; s++) {
    const cs = new Int32Array(F), acc = Float32Array.from(recon[s]);
    for (let f = 0; f < F; f++) {
      const j = nearest(res, f, bk[s], K); cs[f] = j;
      for (let k = 0; k < D; k++) { res[f * D + k] -= bk[s][j * D + k]; acc[f * D + k] += bk[s][j * D + k]; }
    }
    codes.push(cs); recon.push(acc); resRms.push(rms(res));
  }

  // waveform-domain error at every stage count, plus the encoder's own floor
  const buf = new Float32Array(NS), waves = [], rmse = [];
  for (let k = 0; k <= nq; k++) { const w = new Float32Array(NS); decode(recon[k], F, frameLen, binLen, w); waves.push(w); rmse.push(rms(x, w)); }
  const floorW = decode(lat, F, frameLen, binLen, buf), floor = rms(x, floorW);

  const bitsPerCb = fps * Math.log2(K);
  return { fps, nq, K, seed, F, frameLen, binLen, x, lat, codes, recon, waves, rmse, resRms, floor, bitsPerCb, bk };
}

// ------------------------------------------------------------------- the page
mount({
  mount: 'body',
  title: 'audio tokens — a waveform becomes a code stack (RVQ)',
  blurb: 'A language model predicts integers, and a waveform is not integers — it is tens of thousands of floating-point samples per second. Neural audio codecs bridge that gap in two moves. First an ENCODER downsamples the waveform to a low frame rate: SoundStream/EnCodec-class codecs land around 50–75 frames per second, and Mimi reaches 12.5, so one second becomes a few dozen vectors instead of 24,000 samples. Then RESIDUAL VECTOR QUANTIZATION turns each vector into integers: codebook 1 replaces the frame vector with its nearest codeword; what codebook 1 got WRONG — the residual — is quantized by codebook 2; that residual by codebook 3, and so on. One frame becomes a small stack of codes, coarse to fine, and dropping the deepest codebooks degrades quality gracefully instead of destroying it. Press play and watch the reconstruction climb onto the waveform one codebook at a time. The trade is arithmetic: bitrate = frames/sec × codebooks × log2(codebook size), so every added codebook costs the SAME bits and buys LESS error reduction — that is the staircase on the right. The second trade is why the low-frame-rate codecs matter at all: every frame is a token position a model has to predict, so at 75 fps × 8 codebooks a single minute of audio is 36,000 tokens. Drag the waveform to change what is being coded; hover any frame for its code stack. One honest caveat is drawn on the chart: the encoder here is a box filter and a linear synthesis, not a trained network, so it has a FLOOR of its own — the dashed line — that no number of codebooks gets below, and at very low frame rates that floor is what dominates. Spending a trained encoder to push that floor down at 12.5 frames per second is exactly the work the low-frame-rate codecs do.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  autoplay: true,
  compare: { key: 'nq', a: 1, b: 8, labelA: '1 codebook (coarse)', labelB: '8 codebooks (fine)', rebuild: true },
  challenges: [
    { goal: 'Code it cleanly: reconstruction RMSE below 0.06 at under 6.0 kbit/s.', hint: 'a lower frame rate buys codebooks; watch the encoder floor rise as frames get longer.', check: (api) => ({ solved: (api.probe.rmse ?? 1) < 0.06 && (api.probe.kbps ?? 99) < 6, detail: `RMSE ${(api.probe.rmse ?? 1).toFixed(4)} at ${(api.probe.kbps ?? 0).toFixed(2)} kbit/s (need < 0.06 and < 6.0)` }) },
    { goal: `Fit a whole minute of audio into a ${CTX.toLocaleString()}-token context.`, hint: 'tokens/minute = fps × 60 × codebooks — the frame rate is the big lever.', check: (api) => ({ solved: (api.probe.tokMin ?? 1e9) <= CTX, detail: `${(api.probe.tokMin ?? 0).toLocaleString()} tokens/min (need ≤ ${CTX.toLocaleString()})` }) },
  ],
  controls: (c, page) => {
    c.select('preset', { label: 'source audio', options: ['two tones', 'chirp', 'burst'], value: 'two tones', rebuild: true });
    c.select('fps', { label: 'frame rate (frames/s)', options: ['12.5', '25', '50', '75', '100'], value: '75', rebuild: true });
    c.slider('nq', { label: 'codebooks (RVQ depth)', min: 1, max: MAXQ, step: 1, value: 4, rebuild: true });
    c.select('K', { label: 'codebook size', options: ['8', '16', '32', '64', '128', '256'], value: '64', rebuild: true });
    c.slider('seed', { label: 'seed', min: 0, max: 40, step: 1, value: 3, rebuild: true });
    c.button('reset drags', () => { bumps = []; if (page.controls._transport) page.controls._transport._dirty = true; page.redraw(); });
    c.transport({
      compute: () => {
        cur = buildAll(page.state);
        const out = [{ k: 0, label: '0 codebooks — the residual is the whole signal' }];
        for (let k = 1; k <= cur.nq; k++) out.push({ k, label: `+ codebook ${k}  ·  ${(k * cur.bitsPerCb / 1000).toFixed(2)} kbit/s  ·  RMSE ${cur.rmse[k].toFixed(4)}` });
        return out;
      }, loop: true, speed: 1.1,
    });
  },
  onPointer: (page, ev) => {
    if (!wavRect) return;
    const inWav = ev.x >= wavRect.x && ev.x <= wavRect.x + wavRect.w && ev.y >= wavRect.y - 8 && ev.y <= wavRect.y + wavRect.h + 8;
    const push = (x, y) => {
      const u = Math.max(0, Math.min(1, (x - wavRect.x) / wavRect.w));
      const a = Math.max(-2, Math.min(2, (wavRect.y + wavRect.h / 2 - y) / (wavRect.h / 2) * 1.2));
      const kx = Math.round(u * 60) / 60;
      const hit = bumps.find((b) => Math.abs(b.x - kx) < 1e-6);
      if (hit) hit.a = a; else { bumps.push({ x: kx, a }); if (bumps.length > 14) bumps.shift(); }
      if (page.controls._transport) page.controls._transport._dirty = true;
      page.redraw();
    };
    if (ev.type === 'down' && inWav) { dragging = true; push(ev.x, ev.y); }
    else if (ev.type === 'up' || ev.type === 'leave') dragging = false;
    else if (ev.type === 'move' && dragging && page.pointer.down) push(ev.x, ev.y);
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    if (page.controls._transport) page.controls._transport.rebuildIfDirty();
    if (!cur) return;
    r.clear(T.n0);
    const cs = page.step(), k = cs ? cs.k : 0;
    const { F, nq, K, x, waves, rmse, resRms, floor, bitsPerCb, codes } = cur;
    const wv = waves[Math.min(k, waves.length - 1)];
    const SC = STAGE_COL();

    // hovered frame (from either the waveform strip or the code grid)
    let hoverF = -1;
    const p = page.pointer;

    // ================= waveform + reconstruction =================
    const wx = 42, wy = 46, ww = W - 322, wh = 172, wzy = wy + wh / 2;
    wavRect = { x: wx, y: wy, w: ww, h: wh };
    const PX = (i) => wx + i / (NS - 1) * ww, PY = (v) => wzy - v * (wh / 2 - 4);
    r.label(`waveform — ink = original,  colour = decoded from ${k}/${nq} codebooks   (drag ↕)`, wx, wy - 8, { color: T.n14, font: '10px ui-monospace, monospace' });
    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(wx, wy, ww, wh);
    ctx.strokeStyle = T.n4; ctx.strokeRect(wx, wy, ww, wh);
    ctx.strokeStyle = T.n6; ctx.beginPath(); ctx.moveTo(wx, wzy); ctx.lineTo(wx + ww, wzy); ctx.stroke();
    // frame boundaries
    ctx.strokeStyle = alphaOf(T.n8, 0.55); ctx.lineWidth = 0.5;
    for (let f = 0; f <= F; f++) { const px = wx + f / F * ww; ctx.beginPath(); ctx.moveTo(px, wy); ctx.lineTo(px, wy + wh); ctx.stroke(); }
    // hover highlight
    if (p.over && p.x >= wx && p.x <= wx + ww && p.y >= wy && p.y <= wy + wh) hoverF = Math.min(F - 1, Math.floor((p.x - wx) / ww * F));
    if (hoverF >= 0) { ctx.fillStyle = alphaOf(T.warn, 0.16); ctx.fillRect(wx + hoverF / F * ww, wy, ww / F, wh); }
    // reconstruction from k codebooks (thick, underneath) ...
    ctx.strokeStyle = k === 0 ? T.n9 : SC[(k - 1) % SC.length]; ctx.lineWidth = 2.2; ctx.beginPath();
    for (let i = 0; i < NS; i++) { const px = PX(i), py = PY(wv[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    // ... with the original ON TOP, so the target stays visible while the
    // reconstruction climbs onto it (drawn under, it disappears once they agree)
    ctx.strokeStyle = T.n13; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i < NS; i++) { const px = PX(i), py = PY(x[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    ctx.restore();

    // ================= residual (what is still unexplained) =================
    const ry0 = wy + wh + 30, rh = 74, rzy = ry0 + rh / 2;
    r.label(`residual — original minus reconstruction (what the NEXT codebook quantizes)`, wx, ry0 - 8, { color: T.n14, font: '10px ui-monospace, monospace' });
    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(wx, ry0, ww, rh); ctx.strokeStyle = T.n4; ctx.strokeRect(wx, ry0, ww, rh);
    ctx.strokeStyle = T.n6; ctx.beginPath(); ctx.moveTo(wx, rzy); ctx.lineTo(wx + ww, rzy); ctx.stroke();
    ctx.fillStyle = alphaOf(T.bad, 0.30); ctx.strokeStyle = T.bad; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(wx, rzy);
    for (let i = 0; i < NS; i++) ctx.lineTo(PX(i), rzy - Math.max(-1, Math.min(1, (x[i] - wv[i]) * 2.2)) * (rh / 2 - 2));
    ctx.lineTo(wx + ww, rzy); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    for (let i = 0; i < NS; i++) { const px = PX(i), py = rzy - Math.max(-1, Math.min(1, (x[i] - wv[i]) * 2.2)) * (rh / 2 - 2); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke(); ctx.restore();
    r.label(`residual RMS in the latent = ${resRms[Math.min(k, resRms.length - 1)].toFixed(4)}  (×2.2 vertical gain)`, wx, ry0 + rh + 13, { color: T.n11, font: '10px ui-monospace, monospace' });

    // ================= code stack =================
    const gx = wx, gy = ry0 + rh + 38, gcw = ww / F, gch = 15;
    gridRect = { x: gx, y: gy, w: ww, h: nq * gch };
    r.label(`code stack — ${nq} × ${F} frames = ${nq * F} integers for ${DUR.toFixed(1)} s  (q1 coarse → q${nq} finest)`, gx, gy - 8, { color: T.n14, font: '10px ui-monospace, monospace' });
    if (p.over && p.x >= gx && p.x <= gx + ww && p.y >= gy && p.y <= gy + nq * gch) hoverF = Math.min(F - 1, Math.floor((p.x - gx) / ww * F));
    ctx.save();
    for (let s = 0; s < nq; s++) {
      const active = s < k, col = SC[s % SC.length];
      for (let f = 0; f < F; f++) {
        const code = codes[s] ? codes[s][f] : 0;
        ctx.fillStyle = active ? alphaOf(col, 0.20 + 0.75 * (K > 1 ? code / (K - 1) : 0)) : alphaOf(T.n9, 0.14);
        ctx.fillRect(gx + f * gcw, gy + s * gch, Math.max(1, gcw - 0.7), gch - 1.5);
      }
      r.label(`q${s + 1}`, gx - 20, gy + s * gch + gch - 5, { color: active ? col : T.n9, font: '9px ui-monospace, monospace' });
    }
    if (hoverF >= 0) { ctx.strokeStyle = T.warn; ctx.lineWidth = 1.2; ctx.strokeRect(gx + hoverF * gcw - 0.5, gy - 1, gcw + 1, nq * gch + 1); }
    ctx.restore();

    // ================= the trade: staircase + numbers =================
    const kbps = nq * bitsPerCb / 1000, kbpsNow = k * bitsPerCb / 1000;
    const tokMin = Math.round(cur.fps * 60 * nq), ctxSec = CTX / (cur.fps * nq);
    const textMin = CTX * WORDS_PER_TOKEN / WORDS_PER_MIN;
    // `curve` is the whole error-vs-depth ladder (index = codebooks applied), so
    // the staircase can be read off the page rather than re-derived.
    page.probe = { rmse: rmse[Math.min(k, rmse.length - 1)], kbps, tokMin, floor, curve: rmse.slice() };

    const cx = W - 262, cy = 46, cw = 240, ch = 170;
    r.label('error vs bitrate — the staircase (log)', cx, cy - 8, { color: T.n14, font: '10px ui-monospace, monospace' });
    // Log y: on a linear axis the first drop is so large that the last six
    // steps collapse onto the floor and cannot be read. On a log axis every
    // step is legible AND the diminishing return still shows, because each
    // step's height is the RATIO it bought.
    const eHi = Math.max(rmse[0], 1e-6), eLo = Math.max(Math.min(floor, rmse[nq]) * 0.55, 1e-5);
    const lg = Math.log10, span = Math.max(1e-6, lg(eHi) - lg(eLo));
    const EX = (b) => cx + (nq ? b / (nq * bitsPerCb) : 0) * cw;
    const EY = (e) => cy + ch - Math.max(0, Math.min(1, (lg(Math.max(e, eLo)) - lg(eLo)) / span)) * ch;
    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(cx, cy, cw, ch); ctx.strokeStyle = T.n4; ctx.strokeRect(cx, cy, cw, ch);
    // encoder floor: the asymptote no number of codebooks gets below
    ctx.strokeStyle = T.n9; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(cx, EY(floor)); ctx.lineTo(cx + cw, EY(floor)); ctx.stroke(); ctx.setLineDash([]);
    r.label(`encoder floor ${floor.toFixed(4)}`, cx + 4, EY(floor) + 11, { color: T.n10, font: '8px ui-monospace, monospace' });
    r.label(`${eHi.toFixed(3)}`, cx + cw - 40, cy + 11, { color: T.n10, font: '8px ui-monospace, monospace' });
    r.label(`${eLo.toFixed(4)}`, cx + cw - 44, cy + ch - 4, { color: T.n10, font: '8px ui-monospace, monospace' });
    // the steps themselves: right by a constant bitrate, down by a shrinking error
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(EX(0), EY(rmse[0]));
    for (let s = 1; s <= nq; s++) { ctx.lineTo(EX(s * bitsPerCb), EY(rmse[s - 1])); ctx.lineTo(EX(s * bitsPerCb), EY(rmse[s])); }
    ctx.stroke();
    for (let s = 0; s <= nq; s++) { ctx.fillStyle = s === k ? T.bad : T.accent; ctx.beginPath(); ctx.arc(EX(s * bitsPerCb), EY(rmse[s]), s === k ? 4 : 2.2, 0, 7); ctx.fill(); }
    ctx.restore();
    r.label('0', cx - 1, cy + ch + 11, { color: T.n10, font: '8px ui-monospace, monospace' });
    r.label(`${kbps.toFixed(2)} kbit/s`, cx + cw - 56, cy + ch + 11, { color: T.n10, font: '8px ui-monospace, monospace' });

    let ty = cy + ch + 32;
    const line = (s, col, font) => { r.label(s, cx, ty, { color: col, font: font || '10px ui-monospace, monospace' }); ty += 15; };
    line(`at ${k} codebook${k === 1 ? '' : 's'}:`, T.n14, '11px ui-monospace, monospace');
    line(`RMSE ${rmse[Math.min(k, rmse.length - 1)].toFixed(4)}  ·  ${kbpsNow.toFixed(2)} kbit/s`, T.bad);
    line(`encoder floor RMSE ${floor.toFixed(4)}`, T.n11);
    ty += 6;
    line('bitrate = fps × codebooks × log2(size)', T.n14, '11px ui-monospace, monospace');
    line(`${cur.fps} × ${nq} × ${Math.log2(K)} = ${(kbps * 1000).toFixed(0)} bit/s`, T.ok);
    line(`each codebook costs ${(bitsPerCb / 1000).toFixed(2)} kbit/s`, T.n11);
    ty += 6;
    line('sequence length', T.n14, '11px ui-monospace, monospace');
    line(`${tokMin.toLocaleString()} tokens / minute of audio`, T.violet);
    line(`${CTX.toLocaleString()} tokens ≈ ${ctxSec < 90 ? ctxSec.toFixed(1) + ' s' : (ctxSec / 60).toFixed(1) + ' min'} of audio`, T.violet);
    line(`the same context ≈ ${(textMin / 60).toFixed(1)} h of English text`, T.n11);

    // ================= hover =================
    if (hoverF >= 0) {
      const stack = [];
      for (let s = 0; s < nq; s++) stack.push(`  q${s + 1} = ${codes[s][hoverF]}${s < k ? '' : '  (not yet used)'}`);
      let fe = 0;
      const i0 = hoverF * cur.frameLen;
      for (let i = 0; i < cur.frameLen; i++) { const e = x[i0 + i] - wv[i0 + i]; fe += e * e; }
      page.setTip(`frame ${hoverF} of ${F}   (t = ${(hoverF / cur.fps).toFixed(3)}–${((hoverF + 1) / cur.fps).toFixed(3)} s)\n${cur.frameLen} samples → ${D} channels → ${nq} integers\ncode stack (each 0..${K - 1}):\n${stack.join('\n')}\nresidual RMSE in this frame = ${Math.sqrt(fe / cur.frameLen).toFixed(4)}`);
    }

    let o = `residual vector quantization · ${k}/${nq} codebooks applied.  encoder: ${NS} samples (${DUR.toFixed(1)} s) → ${F} frames @ ${cur.fps} fps × ${D} channels; each frame → a stack of ${nq} integers in [0, ${K - 1}].  tier:${r.name}\n`;
    o += `RMSE ${rmse[Math.min(k, rmse.length - 1)].toFixed(4)} at ${kbpsNow.toFixed(2)} kbit/s; all ${nq} codebooks give RMSE ${rmse[nq].toFixed(4)} at ${kbps.toFixed(2)} kbit/s, against an encoder floor of ${floor.toFixed(4)} that no depth gets below. `;
    o += `Bitrate = ${cur.fps} fps × ${nq} × log2(${K}) = ${(kbps * 1000).toFixed(0)} bit/s, so every added codebook costs a flat ${(bitsPerCb / 1000).toFixed(2)} kbit/s while the error reduction shrinks. Sequence length: ${tokMin.toLocaleString()} tokens per minute of audio — a ${CTX.toLocaleString()}-token context holds ${ctxSec < 90 ? ctxSec.toFixed(1) + ' s' : (ctxSec / 60).toFixed(1) + ' min'} of it, versus roughly ${(textMin / 60).toFixed(1)} h of English text. That gap is why low-frame-rate codecs exist.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__atPage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['preset', 'fps', 'K']) if (q.has(key)) page.controls.set(key, q.get(key), { rebuild: true });
  for (const key of ['nq', 'seed']) if (q.has(key)) page.controls.set(key, +q.get(key), { rebuild: true });
  if (q.has('bump')) { bumps = q.get('bump').split(';').filter(Boolean).map((s) => { const [bx, ba] = s.split(',').map(Number); return { x: bx, a: ba }; }); if (page.controls._transport) page.controls._transport._dirty = true; }
  if (page.controls._transport) {
    const tp = page.controls._transport;
    tp.rebuildIfDirty();
    // ?step=N is a statement about THIS view: park there rather than letting
    // autoplay walk off it before the frame is captured.
    if (q.has('step')) { tp.pause(); tp.seek(+q.get('step')); }
    else tp.seek(page.state.nq | 0);
  }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
