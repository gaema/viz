// ring-attention concept page -- ONE sequence, too long for any single device,
// attended to without any device ever holding all of it.
//
// The mechanism: shard the sequence across N devices. Each device keeps its own
// QUERIES permanently. The KV blocks then ROTATE around the ring: at each of N
// steps every device attends its local queries against whichever KV block it
// currently holds, folds the result into a running online-softmax state
// (max m, sum l, accumulator O), and passes the block on to its neighbour.
// After N steps every query has met every key, and no device ever stored more
// than a couple of blocks at a time.
//
// This is flash-attention's online softmax with the tiling distributed across
// MACHINES instead of across on-chip SRAM -- same rescale-by-exp(m_old - m_new)
// recurrence, same never-materialize-the-N-by-N-matrix guarantee, one ring hop
// where flash-attention has an SRAM tile load. See the flash-attention page.
//
// THE POINT IS THE OVERLAP. The send of the next KV block is issued while the
// current block's attention is still computing. Transfer cost grows with the
// block (linear in tokens); attention cost grows with the block SQUARED. So a
// large block hides the wire completely and the ring scales, while a small
// block -- or a slow link -- leaves the devices idle waiting on bytes. Both
// regimes are reachable from the controls, and the stall is drawn as real idle
// time on the per-device timeline.
//
// Two layers of number live on this page and they are deliberately different
// scales: the ring's accumulator values are a REAL online softmax over a tiny
// toy shard (3 queries, 3 keys, 4 dims per device) so hover shows a true
// derivation, while every byte / millisecond / megabyte figure is computed live
// from the sequence length, block size, model width, link bandwidth and device
// throughput the reader sets.
//
// Source: Liu, Zaharia, Abbeel, "Ring Attention with Blockwise Transformers for
// Near-Infinite Context", https://arxiv.org/abs/2310.01889

import { mount } from '../framework/layout.js';
import { categorical, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';

const BQ = 3, BK = 3, D = 4;          // toy shard per device: queries, keys, head_dim
const KV_BYTES = 2;                   // KV elements are 16-bit
const ACC_BYTES = 4;                  // the O accumulator is fp32, as in flash-attention

let cur = null;                       // { n, steps, ... } built by the transport
let rDev = [];                        // per-device node hit circles
let rWire = [];                       // per-wire hit circles
let rBar = [];                        // per-(device, step) timeline rects
let rRing = null, rTime = null;       // the two drag regions
let drag = null;                      // 'ring' | 'link' | null
let acc = { x: 0, y: 0 };             // drag accumulators (pixels -> discrete steps)

const fmtB = (b) => (b >= 1e9 ? (b / 1e9).toFixed(2) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : (b / 1e3).toFixed(1) + ' kB');
const fmtMs = (x) => (x >= 100 ? x.toFixed(0) + ' ms' : x >= 10 ? x.toFixed(1) + ' ms' : x.toFixed(2) + ' ms');
const fmtTok = (t) => (t >= 1024 ? (t / 1024).toFixed(t % 1024 ? 1 : 0) + 'K' : String(Math.round(t)));
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// --- the numbers, computed live from the reader's settings ------------------
// One attention layer, all heads, one micro-batch, non-causal (every block pair
// does equal work -- a causal mask would idle half the ring and is a separate
// scheduling problem, not drawn here).
function perf(st) {
  const n = Math.max(1, st.n | 0);
  const seq = (st.seq | 0) * 1024;              // tokens
  const hid = st.hidden | 0;                    // model width = heads x head_dim
  const block = seq / n;                        // tokens of Q, and of KV, per device
  const kvBytes = 2 * block * hid * KV_BYTES;   // K and V for one block
  const flops = 4 * block * block * hid;        // QK^T then PV, 2 flops per MAC
  const tc = (flops / (st.flops * 1e12)) * 1000;
  const tt = (kvBytes / (st.link * 1e9)) * 1000;
  const stepMs = Math.max(tc, tt);
  const idle = Math.max(0, tt - tc);
  const qB = block * hid * KV_BYTES;
  const oB = block * hid * ACC_BYTES + block * 2 * ACC_BYTES;   // O plus the m / l columns
  const mem = 2 * kvBytes + qB + oB;            // current block + incoming block + Q + accumulator
  // One device holding the WHOLE sequence: its queries and its accumulator are
  // whole-sequence too, not one shard's. Scaling only kvBytes understated this
  // baseline by ~2.1x at the defaults and by more as n grows -- against the
  // page's own headline comparison, which is the one number it exists to make.
  const single = n * (kvBytes + qB + oB);
  return { n, seq, hid, block, kvBytes, flops, tc, tt, stepMs, idle, mem, single, qB, oB,
    total: n * stepMs, totalIdle: n * idle, hidden: hid };
}

// --- the toy online softmax that actually runs ------------------------------
// Device i owns queries Q[i] forever. At step s it holds block owner(i,s), folds
// it into (m, l, O), and passes it to device i+1.
const ownerOf = (i, s, n) => ((i - s) % n + n) % n;

function buildData(st) {
  const n = Math.max(2, st.n | 0), seed = st.seed | 0, sq = Math.sqrt(D);
  const Q = [], K = [], V = [];
  for (let i = 0; i < n; i++) {
    Q.push(seededRandn(seed + i * 7 + 1, [BQ, D]));
    K.push(seededRandn(seed + i * 7 + 2, [BK, D]));
    V.push(seededRandn(seed + i * 7 + 3, [BK, D]));
  }
  const m = [], l = [], O = [];
  for (let i = 0; i < n; i++) { m.push(new Float32Array(BQ).fill(-Infinity)); l.push(new Float32Array(BQ)); O.push(new Float32Array(BQ * D)); }
  const steps = [];
  for (let s = 0; s < n; s++) {
    const per = [];
    for (let i = 0; i < n; i++) {
      const o = ownerOf(i, s, n);
      const S = new Float32Array(BQ * BK), P = new Float32Array(BQ * BK), rescale = new Float32Array(BQ);
      const mold = Float32Array.from(m[i]);
      for (let a = 0; a < BQ; a++) {
        let mloc = -Infinity;
        for (let b = 0; b < BK; b++) {
          let dp = 0; for (let c = 0; c < D; c++) dp += Q[i].data[a * D + c] * K[o].data[b * D + c];
          S[a * BK + b] = dp / sq; if (dp / sq > mloc) mloc = dp / sq;
        }
        const mnew = Math.max(m[i][a], mloc);
        const rs = isFinite(m[i][a]) ? Math.exp(m[i][a] - mnew) : 1;
        rescale[a] = rs;
        let sumP = 0; const add = new Float32Array(D);
        for (let b = 0; b < BK; b++) {
          const p = Math.exp(S[a * BK + b] - mnew); P[a * BK + b] = p; sumP += p;
          for (let c = 0; c < D; c++) add[c] += p * V[o].data[b * D + c];
        }
        l[i][a] = (isFinite(m[i][a]) ? rs : 0) * l[i][a] + sumP;
        for (let c = 0; c < D; c++) O[i][a * D + c] = (isFinite(m[i][a]) ? rs : 0) * O[i][a * D + c] + add[c];
        m[i][a] = mnew;
      }
      const last = s === n - 1;
      let norm = null;
      if (last) { norm = new Float32Array(BQ * D); for (let a = 0; a < BQ; a++) for (let c = 0; c < D; c++) norm[a * D + c] = O[i][a * D + c] / l[i][a]; }
      per.push({ dev: i, owner: o, S, P, rescale, mold, m: Float32Array.from(m[i]), l: Float32Array.from(l[i]), O: Float32Array.from(O[i]), norm });
    }
    steps.push({ s, n, per, label: `rotation ${s + 1}/${n} — every device holds a different KV block`, isLast: s === n - 1 });
  }
  cur = { n, steps };
  return steps;
}

// --- drawing helpers --------------------------------------------------------
function hatch(ctx, x, y, w, h, color) {
  if (w <= 0.4) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.fillStyle = rgbaToken('n14', 0.05); ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let k = -h; k < w + h; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); }
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'ring attention — a sequence no single device can hold',
  blurb: 'One sequence, sharded across N devices. Each device keeps its own queries forever; the KV blocks rotate around the ring, and after N hops every query has met every key — while no device ever stored more than a couple of blocks. It is flash-attention’s online softmax with the tiling distributed across MACHINES instead of across on-chip SRAM. The point is the OVERLAP: the next block’s send is issued while the current block is still computing, so if compute per block exceeds transfer the wire disappears and the ring scales — and if it does not, the devices sit idle. Drag the ring left/right for devices and up/down for sequence length; drag the timeline sideways for link bandwidth. Hover a device for what it holds and its running max/sum; hover a wire for the bytes in flight.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  animate: true,
  challenges: [
    { goal: 'Hide the communication completely — make compute per block exceed transfer per block.', hint: 'raise the link bandwidth, or make the blocks bigger (longer sequence / fewer devices) — compute grows as block², transfer only as block.',
      check: (api) => ({ solved: (api.probe.tc ?? 0) > (api.probe.tt ?? 1), detail: `compute ${fmtMs(api.probe.tc ?? 0)} vs transfer ${fmtMs(api.probe.tt ?? 0)} per block` }) },
    { goal: 'Now break it — drive the ring into a stall with real idle time on the timeline.', hint: 'drop the link bandwidth, or cut the blocks small by adding devices at a short sequence.',
      check: (api) => ({ solved: (api.probe.idle ?? 0) > 0, detail: `idle ${fmtMs(api.probe.idle ?? 0)} per step` }) },
    // The device slider alone cannot reach this at the default width: memory is
    // 14·block·hid + 8·block, so at 256K with the maximum 12 devices and
    // hidden = 4096 it is still ~1.25 GB. The width has to come down too.
    { goal: 'Hold a 256K-token sequence with under 512 MB of memory per device.', hint: 'per-device memory is set by the BLOCK and the model WIDTH, not by the sequence — add devices to shrink the block, and narrow the hidden size too; devices alone will not get you there at 4096.',
      check: (api) => ({ solved: (api.probe.seq ?? 0) >= 256 * 1024 && (api.probe.mem ?? 1e12) < 512e6, detail: `${fmtTok(api.probe.seq ?? 0)} tokens, ${fmtB(api.probe.mem ?? 0)} per device` }) },
  ],
  controls: (c, page) => {
    c.stepper('n', { label: 'devices in the ring', min: 2, max: 12, value: 8 });
    c.slider('seq', { label: 'sequence length (K tokens)', min: 16, max: 512, step: 16, value: 128, format: (v) => v + 'K' });
    c.slider('link', { label: 'link bandwidth (GB/s)', min: 5, max: 600, step: 5, value: 100 });
    c.slider('flops', { label: 'device throughput (TFLOP/s)', min: 10, max: 1000, step: 10, value: 200 });
    c.slider('hidden', { label: 'model width (heads × head_dim)', min: 1024, max: 8192, step: 512, value: 4096 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.1, loop: true });
  },

  // Direct manipulation. Ring region: horizontal drag resizes the ring (device
  // count), vertical drag resizes the problem (sequence length). Timeline
  // region: horizontal drag changes link bandwidth and the packing re-flows
  // under your hand -- stalls appear or vanish.
  onPointer: (page, ev) => {
    const inR = (r) => r && ev.x >= r.x && ev.x <= r.x + r.w && ev.y >= r.y && ev.y <= r.y + r.h;
    if (ev.type === 'down') { drag = inR(rTime) ? 'link' : inR(rRing) ? 'ring' : null; acc.x = 0; acc.y = 0; return; }
    if (ev.type === 'up' || ev.type === 'leave') { drag = null; return; }
    if (ev.type !== 'move' || !drag || !page.pointer.down) return;
    const st = page.state, C = page.controls;
    if (drag === 'ring') {
      acc.x += ev.dx; acc.y += ev.dy;
      while (Math.abs(acc.x) >= 26) { const d = Math.sign(acc.x); acc.x -= d * 26; C.set('n', clamp((st.n | 0) + d, 2, 12), { rebuild: true }); }
      while (Math.abs(acc.y) >= 10) { const d = -Math.sign(acc.y); acc.y -= Math.sign(acc.y) * 10; C.set('seq', clamp((st.seq | 0) + d * 16, 16, 512)); }
    } else if (drag === 'link') {
      acc.x += ev.dx;
      const next = clamp(Math.round((st.link * Math.exp(acc.x / 220)) / 5) * 5, 5, 600);
      if (next !== st.link) { acc.x = 0; C.set('link', next); }
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    r.clear(T.n0);
    const W = page.W, H = page.H;
    const p = perf(st);
    const s = page.step();
    const j = s ? s.s : -1;
    const n = cur.n;
    page.probe = { tc: p.tc, tt: p.tt, idle: p.idle, mem: p.mem, seq: p.seq, n, s: j, nT: n };

    // ---------------- ring ---------------------------------------------------
    const topH = H * 0.55;
    rRing = { x: 4, y: 14, w: W * 0.40, h: topH - 18 };
    const cx = rRing.x + rRing.w * 0.5, cy = rRing.y + 26 + (rRing.h - 46) * 0.5;
    const R = Math.min(rRing.w * 0.32, (rRing.h - 46) * 0.36);
    // Nodes must not overlap: the arc between two neighbours is 2πR/n, so the
    // radius has to shrink with the device count, not just with the ring.
    const nr = clamp(Math.min(R * 0.42, (R * 2.6) / n), 8, 22);

    r.label('the ring — KV blocks rotate, queries never move', rRing.x + 6, rRing.y + 4, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label('◂▸ devices   ▴▾ sequence  (drag)', rRing.x + 6, rRing.y + rRing.h - 2, { color: drag === 'ring' ? T.warn : T.n9, font: '10px ui-monospace, monospace' });

    ctx.save();
    ctx.strokeStyle = alphaOf(T.n8, 0.55); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();

    const ang = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
    const px = (a, rad) => [cx + Math.cos(a) * rad, cy + Math.sin(a) * rad];

    // wires: every device ships its current block to i+1 while it computes.
    rWire = [];
    const flight = (page.t * 0.85) % 1;
    for (let i = 0; i < n; i++) {
      const a0 = ang(i), a1 = a0 + (2 * Math.PI) / n;
      const pad = Math.min(0.55, (nr + 5) / R);
      const live = j >= 0 && j < n - 1;
      ctx.save();
      ctx.strokeStyle = live ? alphaOf(T.teal, 0.85) : alphaOf(T.n7, 0.7);
      ctx.lineWidth = live ? 2 : 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, a0 + pad, a1 - pad); ctx.stroke();
      const ae = a1 - pad, [hx, hy] = px(ae, R);
      const tx = -Math.sin(ae), ty = Math.cos(ae);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath(); ctx.moveTo(hx + tx * 6, hy + ty * 6);
      ctx.lineTo(hx - tx * 3 + Math.cos(ae) * 3.5, hy - ty * 3 + Math.sin(ae) * 3.5);
      ctx.lineTo(hx - tx * 3 - Math.cos(ae) * 3.5, hy - ty * 3 - Math.sin(ae) * 3.5);
      ctx.closePath(); ctx.fill();
      // the block in flight, actually moving
      if (live) {
        const af = a0 + pad + (a1 - pad - (a0 + pad)) * flight;
        const [fx, fy] = px(af, R);
        const own = ownerOf(i, j, n);
        ctx.fillStyle = alphaOf(categorical(own), 0.95);
        ctx.strokeStyle = alphaOf(T.n14, 0.35); ctx.lineWidth = 1;
        ctx.fillRect(fx - 4, fy - 4, 8, 8); ctx.strokeRect(fx - 4, fy - 4, 8, 8);
      }
      ctx.restore();
      const am = (a0 + a1) / 2, [wx, wy] = px(am, R);
      rWire.push({ x: wx, y: wy, from: i, to: (i + 1) % n });
    }

    // device nodes, coloured by the KV block they hold RIGHT NOW
    rDev = [];
    for (let i = 0; i < n; i++) {
      const [x, y] = px(ang(i), R);
      const own = s ? s.per[i].owner : i;
      const col = categorical(own);
      ctx.save();
      ctx.fillStyle = alphaOf(col, 0.9);
      ctx.strokeStyle = T.n12; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(x, y, nr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = inkOn(col);
      ctx.font = `${Math.round(nr * 0.75)}px ui-monospace, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('D' + i, x, y);
      ctx.restore();
      const [lx, ly] = px(ang(i), R + nr + 9);
      r.label('b' + own, lx, ly + 3, { color: alphaOf(categorical(own), 1), font: '10px ui-monospace, monospace', align: 'center' });
      rDev.push({ x, y, r: nr, dev: i });
    }

    ctx.save();
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = T.n13; ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(j >= 0 ? `step ${j + 1} / ${n}` : `${n} devices`, cx, cy - 8);
    ctx.fillStyle = T.n10; ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(`block ${fmtTok(p.block)} tok`, cx, cy + 7);
    ctx.restore();

    // ---------------- numbers panel -----------------------------------------
    const bx = rRing.x + rRing.w + 14, bw = W - bx - 10;
    const lh = 12.5;
    let ty = rRing.y + 10;
    const line = (txt, col, font) => {
      if (ty > topH + 6) return;                       // never spill into the timeline
      r.label(txt, bx, ty, { color: col || T.n12, font: font || '10.5px ui-monospace, monospace' });
      ty += lh;
    };

    line(`sequence ${fmtTok(p.seq)} tokens ÷ ${n} devices = ${fmtTok(p.block)}-token block each`, T.n13, '11px ui-monospace, monospace');
    line(`width ${p.hidden} · KV 16-bit · one attention layer, all heads`, T.n10, '10px ui-monospace, monospace');
    ty += 2;

    line(`PER-DEVICE MEMORY   ${fmtB(p.mem)}`, T.accent, '12px ui-monospace, monospace');
    line(`set by the BLOCK, not the sequence: 2 KV buffers ${fmtB(2 * p.kvBytes)}`, T.n10, '10px ui-monospace, monospace');
    line(`(holding + incoming) + Q ${fmtB(p.qB)} + fp32 accum ${fmtB(p.oB)}`, T.n10, '10px ui-monospace, monospace');

    // memory bars: the ring's per-device requirement against the single-device baseline
    const barW = bw - 4, mx = bx, bh = 8;
    const scale = Math.max(p.single, p.mem) || 1;
    r.label(`ring, per device — ${fmtB(p.mem)}`, mx, ty, { color: T.n11, font: '9.5px ui-monospace, monospace' });
    ctx.save();
    ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(mx, ty + 3, barW, bh);
    ctx.fillStyle = alphaOf(T.accent, 0.85); ctx.fillRect(mx, ty + 3, Math.max(1, (barW * p.mem) / scale), bh);
    ctx.restore();
    ty += bh + 14;
    r.label(`single device — whole KV resident, ${fmtB(p.single)}`, mx, ty, { color: T.n11, font: '9.5px ui-monospace, monospace' });
    ctx.save();
    ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(mx, ty + 3, barW, bh);
    ctx.restore();
    hatch(ctx, mx, ty + 3, (barW * p.single) / scale, bh, alphaOf(T.bad, 0.8));
    ty += bh + 13;
    line('the baseline gets no runtime on purpose: past some length it', T.n9, '10px ui-monospace, monospace');
    line('does not fit at all, and timing a run that cannot start is a fiction.', T.n9, '10px ui-monospace, monospace');
    ty += 3;

    line('PER BLOCK, PER STEP', T.n11, '10px ui-monospace, monospace');
    line(`attention compute   ${fmtMs(p.tc)}   (${(p.flops / 1e9).toFixed(0)} GFLOP, ∝ block²)`, T.n12);
    line(`KV block on the wire  ${fmtMs(p.tt)}   (${fmtB(p.kvBytes)}, ∝ block)`, T.n12);
    const hidden = p.tc >= p.tt;
    line(hidden
      ? `OVERLAPPED — wire fully hidden, ${(100 * (1 - p.tt / Math.max(p.tc, 1e-9))).toFixed(0)}% slack`
      : `STALLED — ${fmtMs(p.idle)} idle per step, waiting on bytes`,
      hidden ? T.ok : T.bad, '11px ui-monospace, monospace');
    line(`ring-wide ${fmtB(p.kvBytes * n)} per step across ${n} wires`, T.n10, '10px ui-monospace, monospace');
    line(`${n} × max(compute, transfer) = ${fmtMs(p.total)}; idle ${fmtMs(p.totalIdle)} (${(100 * p.totalIdle / Math.max(p.total, 1e-9)).toFixed(1)}%)`, T.n13);

    // ---------------- per-device timeline ------------------------------------
    const x0 = 40, x1 = W - 12;
    const ty0 = topH + 30;
    rTime = { x: 0, y: ty0 - 20, w: W, h: H - (ty0 - 20) };
    const rowH = Math.max(7, Math.min(20, (H - 16 - ty0) / n));
    const pxms = (x1 - x0) / Math.max(p.total, 1e-9);

    r.label('per-device timeline — one row per device, ' + n + ' steps', x0, ty0 - 22, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label('◂▸ link bandwidth (drag)', x1, ty0 - 22, { color: drag === 'link' ? T.warn : T.n9, font: '10px ui-monospace, monospace', align: 'right' });

    rBar = [];
    for (let i = 0; i < n; i++) {
      const y = ty0 + i * rowH;
      r.label('D' + i, x0 - 6, y + rowH * 0.62, { color: T.n11, font: '9.5px ui-monospace, monospace', align: 'right' });
      for (let sIdx = 0; sIdx < n; sIdx++) {
        const xs = x0 + sIdx * p.stepMs * pxms;
        const own = ownerOf(i, sIdx, n);
        const cw = Math.max(0.5, p.tc * pxms), iw = Math.max(0, p.idle * pxms);
        const h = rowH - 4;
        ctx.save();
        ctx.fillStyle = alphaOf(categorical(own), sIdx === j ? 0.95 : 0.45);
        ctx.fillRect(xs, y, cw, h * 0.62);
        ctx.restore();
        if (iw > 0.4) hatch(ctx, xs + cw, y, iw, h * 0.62, alphaOf(T.bad, 0.85));
        // the wire, running underneath and concurrently with the compute above
        ctx.save();
        ctx.fillStyle = alphaOf(T.teal, sIdx === j ? 0.8 : 0.4);
        ctx.fillRect(xs, y + h * 0.68, Math.max(0.5, p.tt * pxms), Math.max(2, h * 0.26));
        ctx.restore();
        rBar.push({ x: xs, y, w: Math.max(cw + iw, p.tt * pxms), h, dev: i, step: sIdx, owner: own });
      }
    }
    // playhead + current-step band
    if (j >= 0) {
      ctx.save();
      ctx.fillStyle = alphaOf(T.accent, 0.09);
      ctx.fillRect(x0 + j * p.stepMs * pxms, ty0 - 4, p.stepMs * pxms, n * rowH + 4);
      ctx.strokeStyle = alphaOf(T.accent, 0.8); ctx.lineWidth = 1.2;
      const phx = x0 + (j + 1) * p.stepMs * pxms;
      ctx.beginPath(); ctx.moveTo(phx, ty0 - 6); ctx.lineTo(phx, ty0 + n * rowH + 2); ctx.stroke();
      ctx.restore();
    }
    const legY = ty0 + n * rowH + 12;
    if (legY < H - 2) {
      r.label('0', x0, legY, { color: T.n9, font: '9.5px ui-monospace, monospace' });
      r.label(fmtMs(p.total), x1, legY, { color: T.n9, font: '9.5px ui-monospace, monospace', align: 'right' });
      r.label('▬ attention compute (coloured by which KV block)   ▬ KV block in flight, overlapped   ▨ stall, waiting on the wire',
        (x0 + x1) / 2, legY, { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'center' });
    }

    // ---------------- hover-to-inspect ---------------------------------------
    if (page.pointer.over && !drag) {
      const pt = page.pointer;
      let tip = null;
      for (const d of rDev) {
        if (Math.hypot(pt.x - d.x, pt.y - d.y) <= d.r + 2) {
          const rec = s ? s.per[d.dev] : null;
          const q0 = d.dev * p.block, q1 = (d.dev + 1) * p.block;
          tip = `device D${d.dev}\n`
              + `owns queries [${fmtTok(q0)} … ${fmtTok(q1)}) — they never move\n`
              + (rec ? `holds KV block b${rec.owner} right now (from D${rec.owner}), ${fmtB(p.kvBytes)}\n` : `holds KV block b${d.dev} (its own), ${fmtB(p.kvBytes)}\n`)
              + `resident: 2 KV buffers + Q + fp32 accum = ${fmtB(p.mem)}\n`
              + (rec
                ? `running max m = [${Array.from(rec.m).map((v) => v.toFixed(2)).join(', ')}]\n`
                + `running sum l = [${Array.from(rec.l).map((v) => v.toFixed(2)).join(', ')}]\n`
                + `rescale prior accum by exp(m_old−m_new) = [${Array.from(rec.rescale).map((v) => v.toFixed(2)).join(', ')}]\n`
                + `O accum row0 = [${Array.from(rec.O.slice(0, D)).map((v) => v.toFixed(2)).join(', ')}]`
                + (rec.norm ? `\nlast step → output = O/l = [${Array.from(rec.norm.slice(0, D)).map((v) => v.toFixed(2)).join(', ')}]` : '')
                : `(press ▶ — the accumulators fill as the blocks rotate)`);
          break;
        }
      }
      if (!tip) for (const w of rWire) {
        if (Math.hypot(pt.x - w.x, pt.y - w.y) <= 11) {
          const own = j >= 0 ? ownerOf(w.from, j, n) : w.from;
          tip = `wire D${w.from} → D${w.to}\n`
              + `in flight: KV block b${own} — K and V for ${fmtTok(p.block)} tokens × ${p.hidden} wide × 16-bit\n`
              + `= ${fmtB(p.kvBytes)} at ${st.link} GB/s = ${fmtMs(p.tt)}\n`
              + `issued WHILE block b${own} is still computing (${fmtMs(p.tc)})\n`
              + (p.tc >= p.tt ? `→ fully hidden behind compute` : `→ ${fmtMs(p.idle)} of it is NOT hidden: the receiver stalls`);
          break;
        }
      }
      if (!tip) for (const b of rBar) {
        if (pt.x >= b.x && pt.x <= b.x + b.w && pt.y >= b.y && pt.y <= b.y + b.h) {
          tip = `D${b.dev}, step ${b.step + 1}/${n} — attends its queries against KV block b${b.owner}\n`
              + `compute ${fmtMs(p.tc)}   transfer ${fmtMs(p.tt)} (concurrent)   step wall ${fmtMs(p.stepMs)}\n`
              + (p.idle > 0 ? `idle ${fmtMs(p.idle)} — compute finished first and the next block has not landed` : `idle 0 — the block landed before the compute finished`);
          break;
        }
      }
      if (tip) page.setTip(tip);
    }

    // ---------------- readout -------------------------------------------------
    let o = `ring attention — ${n} devices, ${fmtTok(p.seq)} tokens, ${fmtTok(p.block)}-token block each.  `;
    o += j >= 0 ? `step ${j + 1} / ${n}: every device attends its own queries against the block it holds, then passes it on.\n`
               : `press ▶ — the blocks rotate ${n} times and every query meets every key.\n`;
    o += `per-device memory ${fmtB(p.mem)} (block-sized, NOT sequence-sized)  ·  single-device baseline needs ${fmtB(p.single)} resident and does not fit past some length\n`;
    o += `compute/block ${fmtMs(p.tc)}  vs  transfer/block ${fmtMs(p.tt)} at ${st.link} GB/s  →  `;
    o += p.tc >= p.tt
      ? `communication fully HIDDEN; total ${fmtMs(p.total)}, idle 0.00 ms`
      : `STALL: ${fmtMs(p.idle)} idle per step; total ${fmtMs(p.total)}, idle ${fmtMs(p.totalIdle)} (${(100 * p.totalIdle / Math.max(p.total, 1e-9)).toFixed(1)}%)`;
    o += `\ntier:${r.name}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__ringPage = page;
  const q = new URLSearchParams(location.search);
  const C = page.controls, t = C._transport;
  // Restore the deep-linked control state (layout.js mirrors state INTO the URL;
  // reading it back out is the page's job).
  const num = (k, lo, hi, rebuild) => { if (!q.has(k)) return; const v = +q.get(k); if (Number.isFinite(v)) C.set(k, clamp(v, lo, hi), { rebuild: !!rebuild, silent: true }); };
  num('n', 2, 12, true);
  num('seq', 16, 512);
  num('link', 5, 600);
  num('flops', 10, 1000);
  num('hidden', 1024, 8192);
  num('seed', 0, 99, true);
  if (t) t.rebuild();
  // ?step=N (0-based rotation step) is the headless stand-in for the transport,
  // the same way ?hover=x,y stands in for a real cursor.
  if (q.has('step') && t) { t.seek(parseInt(q.get('step'), 10)); t.pause(); }
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
    if (!q.has('step') && t) { t.seek(Math.min(2, t.steps.length - 1)); t.pause(); }
  }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
