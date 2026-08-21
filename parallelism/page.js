// parallelism concept page -- four different things get called "parallelism"
// when one model is split across several GPUs at INFERENCE time, and readers
// conflate them. This page draws ONE transformer layer on an N-GPU strip and
// lets you pick a strategy PER SUBLAYER, animating what actually moves between
// devices:
//
//   TENSOR PARALLEL (tp)  -- split every matmul across the GPUs. Each device
//       owns a slice of the weights and a PARTIAL result, so every sublayer
//       ends in an ALL-REDUCE of activations. Lowest latency when the link is
//       fat; the all-reduce is on the critical path of every sublayer of every
//       layer, so it degrades hard once the group spans a slower fabric.
//   PIPELINE PARALLEL (pp) -- split LAYERS across the GPUs. A sublayer stays
//       whole on whichever device holds its layer, so the only traffic is a
//       point-to-point send of the activation at each stage boundary: cheap,
//       and it does not grow with the layer count. The cost is BUBBLES -- with
//       few micro-batches most of the pipeline is idle, and a single-token
//       decode step gets no help at all.
//   EXPERT PARALLEL (ep)  -- each GPU holds a SUBSET of the MoE experts. The
//       traffic is an ALL-TO-ALL of TOKENS, and its volume is set by the
//       ROUTER, not by the topology: it is imbalanced by construction, and the
//       slowest link decides the step.
//   DATA-PARALLEL ATTENTION (dp) -- replicate the attention layers; each rank
//       owns its own requests and its own KV. NOTHING crosses for attention.
//       The bill is N copies of the attention weights, and the ranks must
//       still line up at the MoE boundary -- a rank with no work of its own
//       runs a DUMMY forward pass so the collective does not deadlock.
//
// Counters, recomputed live: bytes crossing the wire per decode step, and
// per-GPU memory (weights + KV) against a device budget you set.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT steps through one layer's sublayers in order, so the collectives
//    animate where they actually happen. Auto-plays and loops.
//  - DIRECT MANIPULATION: drag the strip sideways to change the GPU COUNT and
//    watch bytes-on-the-wire and per-GPU memory move under your hand; drag it
//    vertically to skew the ROUTER (which re-shapes the all-to-all); click a
//    GPU's strategy chip to switch that sublayer's strategy.
//  - HOVER a GPU for exactly what it holds, or any arrow for what is crossing
//    and why (with the formula that produced the number).
//  - URL hooks reproduce every view headlessly: ?step, ?gpus, ?attn, ?moe,
//    ?kv, ?d, ?layers, ?experts, ?topk, ?reqs, ?seq, ?skew, ?micro, ?link,
//    ?budget, ?hover=x,y.
//
// OUT OF SCOPE, one line: CONTEXT parallelism -- splitting a single SEQUENCE
// across ranks so one long prompt's attention is computed cooperatively -- is a
// different axis from all four of these and is not drawn here.
//
// Public sources for the mechanisms: the vLLM parallelism-and-scaling serving
// docs, Meta's engineering write-up on tensor / context / expert parallelism
// for LLM inference, and the vLLM large-scale-serving ("wide expert
// parallelism") post, which is where the DP-attention + EP pairing comes from:
// tensor-parallel sharding of a latent-KV attention duplicates the very cache
// the latent form exists to shrink, so the deployment replicates attention
// instead and spends the parallelism on the experts.
import { mount } from '../framework/layout.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

// ---- fixed model conventions (stated on screen, not hidden) ---------------
const BPE = 2;            // bytes per bf16 activation / weight element
const GQA = 8;            // query heads per KV head, for the grouped-KV option
const HEAD = 128;         // attention head dim
const D_LATENT = 576;     // latent-KV cache width per layer (compressed KV + rope part)

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const ATTN_MODES = ['tp', 'pp', 'dp'];
const MOE_MODES = ['tp', 'pp', 'ep'];

const NAME = {
  tp: 'tensor parallel', pp: 'pipeline parallel',
  ep: 'expert parallel', dp: 'data-parallel (replicated)',
};
const SHORT = { tp: 'TP', pp: 'PP', ep: 'EP', dp: 'DP' };

function fmtB(b) {
  if (!isFinite(b)) return '–';
  if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' kB';
  return b.toFixed(0) + ' B';
}
// Split M things over N ranks the way a scheduler does: the remainder lands on
// the low ranks, so a rank CAN come up empty (which is the dummy-forward case).
const share = (M, N) => Array.from({ length: N }, (_, i) => Math.floor(M / N) + (i < M % N ? 1 : 0));

// ---- the model ------------------------------------------------------------
// Every number below is computed from the controls; nothing is annotated.
function build(st) {
  const N = clamp(st.gpus | 0, 1, 8);
  const d = st.d | 0, L = st.layers | 0, E = st.experts | 0;
  const k = Math.min(st.topk | 0, E);
  const B = st.reqs | 0;                 // requests in flight = tokens per decode step
  const S = st.seq | 0;                  // context length per request
  const dff = Math.round(d / 2);         // fine-grained expert intermediate width
  const attn = ATTN_MODES.includes(st.attn) ? st.attn : 'tp';
  const moe = MOE_MODES.includes(st.moe) ? st.moe : 'ep';
  const latent = st.kv === 'latent';

  // parameters per layer
  const attnParams = latent
    ? 2 * d * d + 2 * d * D_LATENT              // q/o full, plus the down/up latent projections
    : 2 * d * d + 2 * d * (d / GQA);            // q/o full, k/v grouped
  const moeParams = E * 3 * d * dff;            // gate + up + down per expert
  const attnBytes = L * attnParams * BPE;
  const moeBytes = L * moeParams * BPE;

  // KV cache. A grouped-KV cache shards across TP ranks only as far as it has
  // KV HEADS to give away; a latent cache is ONE head, so tensor parallel
  // cannot split it at all and every rank keeps a whole copy.
  const kvHeads = latent ? 1 : Math.max(1, Math.round(d / HEAD / GQA));
  const kvPerTok = latent ? L * D_LATENT * BPE : 2 * L * kvHeads * HEAD * BPE;
  const kvTotal = B * S * kvPerTok;

  const layersOn = share(L, N);
  const expertsOn = share(E, N);
  const toksOn = share(B, N);                   // request ownership under DP attention

  // per-GPU memory
  const gpus = [];
  for (let r = 0; r < N; r++) {
    const aw = attn === 'dp' ? attnBytes
      : attn === 'tp' ? attnBytes / N
        : attnBytes * (layersOn[r] / L);
    const mw = moe === 'ep' ? L * expertsOn[r] * 3 * d * dff * BPE
      : moe === 'tp' ? moeBytes / N
        : moeBytes * (layersOn[r] / L);
    const kv = attn === 'dp' ? toksOn[r] * S * kvPerTok
      : attn === 'tp' ? kvTotal / Math.min(N, kvHeads)
        : kvTotal * (layersOn[r] / L);
    const over = st.overhead * 1e9;             // runtime + activation workspace
    gpus.push({ r, aw, mw, kv, over, total: aw + mw + kv + over, layers: layersOn[r], experts: expertsOn[r], toks: toksOn[r] });
  }

  // router: expert popularity with a skew, experts dealt round-robin to ranks.
  const w = new Float64Array(E); let sw = 0;
  for (let e = 0; e < E; e++) { w[e] = Math.pow(1 / (1 + e), st.skew); sw += w[e]; }
  for (let e = 0; e < E; e++) w[e] /= sw;
  const pRank = new Float64Array(N);
  for (let e = 0; e < E; e++) pRank[e % N] += w[e];

  // all-to-all volume matrix (token-assignments per layer), src rank -> dst rank
  const a2a = [];
  let crossTok = 0, recvMax = 0;
  const recv = new Float64Array(N);
  for (let s = 0; s < N; s++) {
    const row = new Float64Array(N);
    for (let r2 = 0; r2 < N; r2++) {
      row[r2] = toksOn[s] * k * pRank[r2];
      recv[r2] += row[r2];
      if (s !== r2) crossTok += row[r2];
    }
    a2a.push(row);
  }
  for (let r2 = 0; r2 < N; r2++) recvMax = Math.max(recvMax, recv[r2]);
  const meanRecv = (B * k) / N;
  const imbalance = meanRecv > 0 ? recvMax / meanRecv : 1;

  // ---- bytes crossing the wire, per DECODE STEP (all L layers) ------------
  // Ring all-reduce of a P-byte payload moves 2(N-1)/N * P per GPU, so the
  // fleet-wide traffic is 2(N-1) * P.
  const payload = B * d * BPE;                 // one sublayer's activation tensor
  const allreduce = 2 * (N - 1) * payload;
  const attnWire = N < 2 ? 0
    : attn === 'tp' ? L * allreduce
      : attn === 'pp' ? (N - 1) * payload      // one send per stage boundary per pass
        : 0;                                    // dp: attention crosses NOTHING
  const dispatch = crossTok * d * BPE;          // per layer
  const moeWire = N < 2 ? 0
    : moe === 'tp' ? L * allreduce
      : moe === 'pp' ? (N - 1) * payload
        : L * 2 * dispatch;                     // ep: dispatch out + combine back
  const wire = attnWire + moeWire;
  const wirePerGpu = wire / N;
  const linkBps = st.link * 1e9;
  const commMs = linkBps > 0 ? (wirePerGpu / linkBps) * 1000 : 0;

  const anyPP = attn === 'pp' || moe === 'pp';
  const bubble = anyPP && N > 1 ? (N - 1) / (st.micro + N - 1) : 0;

  const budget = st.budget * 1e9;
  const peak = Math.max(...gpus.map((g) => g.total));
  const fits = peak <= budget;
  const fleetWeights = gpus.reduce((a, g) => a + g.aw + g.mw, 0);
  const modelWeights = attnBytes + moeBytes;

  return {
    N, d, L, E, k, B, S, dff, attn, moe, latent, kvHeads, kvPerTok, kvTotal,
    attnBytes, moeBytes, gpus, w, pRank, a2a, recv, imbalance, crossTok,
    payload, attnWire, moeWire, wire, wirePerGpu, commMs, anyPP, bubble,
    budget, peak, fits, fleetWeights, modelWeights, dup: fleetWeights / modelWeights,
    layersOn, expertsOn, toksOn,
  };
}

// ---- the sublayer walk ----------------------------------------------------
// Always the same eight stages, so ?step=N is stable; the LABEL and the
// collective drawn at each stage follow the chosen strategies.
function stages(m) {
  const a = m.attn, mo = m.moe;
  const acomm = m.N < 2 ? 'none' : a === 'tp' ? 'allreduce' : a === 'pp' ? 'p2p' : 'none';
  const mcomm = m.N < 2 ? 'none' : mo === 'tp' ? 'allreduce' : mo === 'pp' ? 'p2p' : 'alltoall';
  return [
    { kind: 'idle', part: 'in', role: 'in', label: 'layer input — where the activations already are' },
    { kind: 'compute', part: 'attn', role: 'attn', label: `attention math (${NAME[a]})` },
    { kind: acomm, part: 'attn', role: 'attn-comm', label: acomm === 'allreduce' ? 'attention ALL-REDUCE — every rank held a partial sum' : acomm === 'p2p' ? 'stage boundary — point-to-point activation send' : 'attention: NOTHING crosses the wire' },
    { kind: mo === 'ep' && a === 'dp' ? 'sync' : 'compute', part: 'moe', role: 'router', label: mo === 'ep' && a === 'dp' ? 'router + rank sync (idle ranks run a dummy forward)' : 'router — each token picks its top-k experts' },
    { kind: mcomm === 'alltoall' ? 'alltoall' : 'idle', part: 'moe', role: 'dispatch', label: mcomm === 'alltoall' ? 'ALL-TO-ALL dispatch — tokens travel to their experts' : 'no dispatch: the experts are already where the tokens are' },
    { kind: 'compute', part: 'moe', role: 'expert', label: `expert math (${NAME[mo]})` },
    { kind: mcomm, part: 'moe', role: 'moe-comm', label: mcomm === 'alltoall' ? 'ALL-TO-ALL combine — expert outputs travel home' : mcomm === 'allreduce' ? 'MoE ALL-REDUCE — every rank held a partial sum' : mcomm === 'p2p' ? 'stage boundary — point-to-point activation send' : 'MoE: nothing crosses the wire' },
    { kind: 'idle', part: 'out', role: 'out', label: 'layer output — on to the next of ' + m.L + ' layers' },
  ];
}

// ---- drawing helpers ------------------------------------------------------
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath(); ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr); ctx.closePath();
}
// Diagonal hatch = "paid for, nothing gained": over-budget memory, pipeline bubble.
function hatch(ctx, x, y, w, h, color, bg) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.fillStyle = bg; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let kk = -h; kk < w; kk += 5) { ctx.beginPath(); ctx.moveTo(x + kk, y + h); ctx.lineTo(x + kk + h, y); ctx.stroke(); }
  ctx.restore();
}
function curve(ctx, x1, y1, x2, y2, lift) {
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.quadraticCurveTo((x1 + x2) / 2, Math.min(y1, y2) - lift, x2, y2);
  ctx.stroke();
}
function arrowHead(ctx, x, y, ang, size) {
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(ang - 0.42), y - size * Math.sin(ang - 0.42));
  ctx.lineTo(x - size * Math.cos(ang + 0.42), y - size * Math.sin(ang + 0.42));
  ctx.closePath(); ctx.fill();
}

let m = null;          // current model
let cols = null;       // [{x,w,cx, memRect, chipA, chipM, g}] captured each draw
let wires = null;      // [{x1,y1,x2,y2,mid,why,color}] captured each draw
let drag = null;       // {x0,y0,g0,s0} while the strip is being dragged

function cycle(page, key, modes) {
  const cur = page.state[key];
  const i = modes.indexOf(cur);
  page.controls.set(key, modes[(i + 1) % modes.length], { rebuild: true });
}

mount({
  mount: 'body',
  title: 'parallelism — what actually crosses the wire when one model is split across GPUs',
  blurb: 'Four different things get called "parallelism", and they are not variants of each other: they move different bytes at different moments for different reasons. One transformer layer is drawn across an N-GPU strip, and you pick a strategy per sublayer. TENSOR parallel splits each matmul, so every sublayer ends in an all-reduce of activations — lowest latency, but it wants a fat link and degrades once the group spans a slower fabric. PIPELINE parallel splits layers, so only a point-to-point activation crosses at a stage boundary — cheap comms, but bubbles, and no help at all for single-token latency. EXPERT parallel gives each GPU a subset of the MoE experts and moves TOKENS in an all-to-all whose volume the ROUTER decides, so it is imbalanced by construction. DATA-PARALLEL attention replicates attention — nothing crosses for it at all — and pays in N copies of the weights, with idle ranks running dummy forward passes so the MoE collective still lines up. Drag the strip ↔ to change the GPU count and ↕ to skew the router; click a chip to switch that sublayer; hover a GPU for what it holds and an arrow for what is crossing. Context parallelism — splitting one SEQUENCE across ranks — is a separate axis and is not shown here.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  animate: true,
  compare: { key: 'attn', a: 'tp', b: 'dp', labelA: 'tensor-parallel attention', labelB: 'data-parallel attention', rebuild: true },
  challenges: [
    {
      goal: 'Build a configuration that does not fit: push one GPU past its memory budget.',
      hint: 'replicate attention (DP) on a long context, or shrink the GPU count until the shards stop being small.',
      check: (api) => ({ solved: !!api.probe.over, detail: `peak per-GPU ${fmtB(api.probe.peak || 0)} vs budget ${fmtB(api.probe.budget || 0)}` }),
    },
    {
      goal: 'Get attention completely off the wire while every GPU still fits.',
      hint: 'data-parallel attention crosses nothing — but each rank then keeps a whole copy of the attention weights.',
      check: (api) => ({ solved: api.probe.attnWire === 0 && !api.probe.over && (api.probe.N || 1) > 1, detail: `attention ${fmtB(api.probe.attnWire || 0)}/step, ${api.probe.over ? 'over budget' : 'fits'}` }),
    },
    {
      goal: 'Make the all-to-all lopsided: get one rank receiving more than 1.6× its fair share.',
      hint: 'drag the strip vertically to skew the router — the all-to-all volume is routing-dependent, not topological.',
      check: (api) => ({ solved: (api.probe.imbalance || 1) > 1.6, detail: `busiest rank at ${(api.probe.imbalance || 1).toFixed(2)}× the mean — needs > 1.60×` }),
    },
  ],
  controls: (c, page) => {
    c.stepper('gpus', { label: 'GPUs (N)', min: 1, max: 8, value: 4 });
    c.select('attn', { label: 'attention sublayer', options: [{ value: 'tp', label: 'tensor parallel' }, { value: 'pp', label: 'pipeline parallel' }, { value: 'dp', label: 'data-parallel (replicated)' }], value: 'tp', rebuild: true });
    c.select('moe', { label: 'MoE sublayer', options: [{ value: 'tp', label: 'tensor parallel' }, { value: 'pp', label: 'pipeline parallel' }, { value: 'ep', label: 'expert parallel' }], value: 'ep', rebuild: true });
    c.select('kv', { label: 'KV cache form', options: [{ value: 'gqa', label: 'grouped KV heads' }, { value: 'latent', label: 'latent (single-head) KV' }], value: 'gqa', rebuild: true });
    c.slider('d', { label: 'hidden size d', min: 1024, max: 8192, step: 512, value: 4096, rebuild: true });
    c.slider('layers', { label: 'layers L', min: 8, max: 96, step: 4, value: 48, rebuild: true });
    c.slider('experts', { label: 'experts E', min: 8, max: 128, step: 8, value: 64, rebuild: true });
    c.stepper('topk', { label: 'experts per token (k)', min: 1, max: 8, value: 6 });
    c.slider('reqs', { label: 'requests in flight', min: 1, max: 256, step: 1, value: 48, rebuild: true });
    c.slider('seq', { label: 'context per request (tokens)', min: 512, max: 65536, step: 512, value: 8192, rebuild: true });
    c.slider('skew', { label: 'router skew', min: 0, max: 3, step: 0.05, value: 0.4, rebuild: true });
    c.stepper('micro', { label: 'micro-batches (pipeline)', min: 1, max: 16, value: 1 });
    c.slider('link', { label: 'per-GPU link (GB/s)', min: 10, max: 1000, step: 10, value: 400, rebuild: true });
    c.slider('budget', { label: 'per-GPU memory budget (GB)', min: 16, max: 192, step: 8, value: 80, rebuild: true });
    c.slider('overhead', { label: 'runtime + workspace (GB)', min: 0, max: 16, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => { m = build(page.state); return stages(m); }, speed: 1.4, loop: true });
  },

  // Direct manipulation: the strip itself is the control. Horizontal drag sets
  // the GPU count, vertical drag skews the router, and a click on a strategy
  // chip cycles that sublayer's strategy.
  onPointer: (page, ev) => {
    if (!cols) return;
    if (ev.type === 'down') {
      for (const c of cols) {
        const inA = ev.x >= c.chipA.x && ev.x <= c.chipA.x + c.chipA.w && ev.y >= c.chipA.y && ev.y <= c.chipA.y + c.chipA.h;
        const inM = ev.x >= c.chipM.x && ev.x <= c.chipM.x + c.chipM.w && ev.y >= c.chipM.y && ev.y <= c.chipM.y + c.chipM.h;
        if (inA) { cycle(page, 'attn', ATTN_MODES); return; }
        if (inM) { cycle(page, 'moe', MOE_MODES); return; }
      }
      drag = { x0: ev.x, y0: ev.y, g0: page.state.gpus | 0, s0: +page.state.skew };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      const wantG = clamp(drag.g0 + Math.round((ev.x - drag.x0) / 58), 1, 8);
      if (wantG !== (page.state.gpus | 0)) page.controls.set('gpus', wantG, { rebuild: true });
      const wantS = clamp(+(drag.s0 - (ev.y - drag.y0) * 0.01).toFixed(2), 0, 3);
      if (wantS !== +page.state.skew) page.controls.set('skew', wantS, { rebuild: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    // Rebuild from the live state every frame: the A/B compare panes and the
    // canvas drag both change `state` without going through compute().
    m = build(st);
    r.clear(T.n0);
    const W = page.W, H = page.H;
    const sg = stages(m);
    const tr = page.controls._transport;
    const sIdx = tr && tr.index >= 0 ? Math.min(tr.index, sg.length - 1) : sg.length - 1;
    const stage = sg[sIdx];

    page.probe = {
      over: !m.fits, peak: m.peak, budget: m.budget, attnWire: m.attnWire,
      imbalance: m.imbalance, N: m.N, wire: m.wire,
    };

    const COL_A = T.accent, COL_M = T.violet, COL_KV = T.tealDeep, COL_OV = T.n8;
    const kindColor = { allreduce: T.accent, alltoall: T.violet, p2p: T.ok, sync: T.warn, none: T.n9, compute: T.n9, idle: T.n9 };
    const wireCol = kindColor[stage.kind] || T.n9;

    // ---- header ----------------------------------------------------------
    const pad = 16;
    r.label(`one layer of ${m.L} · d=${m.d} · ${m.E} experts, top-${m.k} · ${m.B} requests × ${m.S} ctx · ${m.N} GPU${m.N > 1 ? 's' : ''}`,
      pad, 18, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label(`attention: ${NAME[m.attn]}   ·   MoE: ${NAME[m.moe]}   ·   KV: ${m.latent ? 'latent (1 head)' : `grouped (${m.kvHeads} KV heads)`}`,
      pad, 34, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label(`step ${sIdx + 1}/${sg.length} — ${stage.label}`, pad, 50, { color: wireCol, font: '12px ui-monospace, monospace' });
    r.label('↔ GPUs · ↕ router skew · click a chip', W - pad, 50, { color: T.accent, font: '10px ui-monospace, monospace', align: 'right' });

    // ---- geometry --------------------------------------------------------
    const bandY0 = 62, bandH = Math.max(74, H * 0.20);
    const bandY1 = bandY0 + bandH;
    const memTop = bandY1 + 40;
    const memBase = H - 100;
    const memH = Math.max(40, memBase - memTop);
    const slot = (W - 2 * pad) / m.N;
    const colW = Math.min(122, slot * 0.78);

    // memory scale: budget line sits at 78% of the column height unless
    // something has blown past it, in which case the tallest column sets it.
    const scaleTop = Math.max(m.budget / 0.78, m.peak * 1.06, 1);
    const px = (bytes) => (bytes / scaleTop) * memH;

    cols = [];
    wires = [];

    // ---- the GPU strip ---------------------------------------------------
    for (let i = 0; i < m.N; i++) {
      const g = m.gpus[i];
      const x = pad + i * slot + (slot - colW) / 2, cx = x + colW / 2;
      // frame
      ctx.save();
      roundRect(ctx, x, memTop - 4, colW, memBase - memTop + 8, 8);
      ctx.fillStyle = rgbaToken('n14', 0.035); ctx.fill();
      ctx.strokeStyle = g.total > m.budget ? T.bad : rgbaToken('n14', 0.15);
      ctx.lineWidth = g.total > m.budget ? 1.8 : 1; ctx.stroke();
      ctx.restore();

      // stacked memory: attention weights, MoE weights, KV, workspace
      let y = memBase;
      const seg = (bytes, color, hl) => {
        const h = px(bytes);
        if (h <= 0.4) return;
        ctx.save();
        ctx.fillStyle = alphaOf(color, hl ? 0.95 : 0.6);
        ctx.fillRect(x + 4, y - h, colW - 8, h);
        if (hl) { ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.strokeRect(x + 4, y - h, colW - 8, h); }
        ctx.restore();
        y -= h;
      };
      seg(g.over, COL_OV, false);
      seg(g.kv, COL_KV, stage.part === 'attn');
      seg(g.aw, COL_A, stage.part === 'attn');
      seg(g.mw, COL_M, stage.part === 'moe');
      // the part that does not fit
      if (g.total > m.budget) {
        const yb = memBase - px(m.budget);
        // transparent ground: the segments still read through the hatch, so you
        // can see WHICH of them pushed the column past the budget line.
        hatch(ctx, x + 4, y, colW - 8, yb - y, alphaOf(T.bad, 0.7), alphaOf(T.bad, 0));
      }

      // labels under the column
      r.label(`GPU ${i}`, cx, memBase + 15, { color: T.n13, font: '11px ui-monospace, monospace', align: 'center' });
      r.label(fmtB(g.total), cx, memBase + 28, { color: g.total > m.budget ? T.bad : T.n11, font: '10px ui-monospace, monospace', align: 'center' });

      // strategy chips above the column -- clickable
      const chipH = 15, chipW = colW;
      const chipA = { x, y: memTop - 40, w: chipW, h: chipH };
      const chipM = { x, y: memTop - 22, w: chipW, h: chipH };
      const chip = (c, color, text) => {
        ctx.save();
        roundRect(ctx, c.x, c.y, c.w, c.h, 4);
        ctx.fillStyle = alphaOf(color, 0.22); ctx.fill();
        ctx.strokeStyle = alphaOf(color, 0.75); ctx.lineWidth = 1; ctx.stroke();
        ctx.fillStyle = color; ctx.font = '9.5px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(text, c.x + c.w / 2, c.y + c.h / 2 + 0.5);
        ctx.restore();
      };
      chip(chipA, COL_A, m.attn === 'pp' ? `attn ${SHORT[m.attn]} · L${g.layers}` : m.attn === 'dp' ? `attn DP · ${g.toks} req` : `attn TP 1/${m.N}`);
      chip(chipM, COL_M, m.moe === 'ep' ? `MoE EP · ${g.experts} exp` : m.moe === 'pp' ? `MoE PP · L${g.layers}` : `MoE TP 1/${m.N}`);

      // dummy-forward badge: a rank with no requests of its own still has to
      // enter the MoE collective, or the ranks that do have work deadlock.
      if (m.attn === 'dp' && g.toks === 0) {
        ctx.save();
        ctx.fillStyle = T.warn; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
        ctx.fillText('dummy fwd', cx, memBase + 40);
        ctx.restore();
      }

      cols.push({ i, x, cx, w: colW, g, chipA, chipM, top: memTop - 4, bot: memBase });
    }

    // budget line across the strip
    {
      const yb = memBase - px(m.budget);
      ctx.save();
      ctx.strokeStyle = T.bad; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.moveTo(pad, yb); ctx.lineTo(W - pad, yb); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      r.label(`device budget ${st.budget} GB`, W - pad, yb - 4, { color: T.bad, font: '10px ui-monospace, monospace', align: 'right' });
    }

    // ---- the collective ---------------------------------------------------
    const yMid = bandY0 + bandH * 0.62;
    const phase = (page.t || 0);
    const packet = (x1, y1, x2, y2, lift, col, off) => {
      const tt = ((phase * 0.55 + off) % 1);
      const mx = (x1 + x2) / 2, my = Math.min(y1, y2) - lift;
      const u = 1 - tt;
      const bx = u * u * x1 + 2 * u * tt * mx + tt * tt * x2;
      const by = u * u * y1 + 2 * u * tt * my + tt * tt * y2;
      ctx.save(); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(bx, by, 2.6, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    };

    const bandLabel = (txt, col) => r.label(txt, pad, bandY0 + 12, { color: col, font: '11px ui-monospace, monospace' });

    if (m.N < 2) {
      bandLabel('one GPU — no parallelism, nothing crosses a wire', T.n10);
    } else if (stage.kind === 'allreduce') {
      const per = 2 * (m.N - 1) / m.N * m.payload;
      bandLabel(`ALL-REDUCE · ${fmtB(m.payload)} payload · ${fmtB(per)} out of EVERY GPU · ${fmtB(2 * (m.N - 1) * m.payload)} across the fabric, per layer`, T.accent);
      for (let i = 0; i < m.N; i++) {
        const a = cols[i], b = cols[(i + 1) % m.N];
        const wrap = i === m.N - 1;
        const y1 = yMid, y2 = yMid, lift = wrap ? bandH * 0.52 : 18;
        ctx.save();
        ctx.strokeStyle = alphaOf(T.accent, 0.85); ctx.lineWidth = 2.4; ctx.fillStyle = alphaOf(T.accent, 0.85);
        curve(ctx, a.cx, y1, b.cx, y2, lift);
        arrowHead(ctx, b.cx, y2, wrap ? -0.6 : 0, 8);
        ctx.restore();
        packet(a.cx, y1, b.cx, y2, lift, T.accent, i / m.N);
        wires.push({
          x1: a.cx, y1, x2: b.cx, y2, lift, mid: { x: (a.cx + b.cx) / 2, y: yMid - lift * 0.5 },
          why: `ring ALL-REDUCE · GPU ${a.i} → GPU ${b.i}\n` +
            `each rank computed a PARTIAL sum of the same [${m.B} × ${m.d}] activation,\n` +
            `so the shard results have to be summed and handed back to everyone.\n` +
            `payload ${m.B}×${m.d}×${BPE} B = ${fmtB(m.payload)}; a ring moves 2(N−1)/N × payload\n` +
            `per GPU = ${fmtB(per)}, once per sublayer, in every one of the ${m.L} layers.\n` +
            `This sits on the critical path of the step: a slower fabric shows up as latency.`,
        });
      }
    } else if (stage.kind === 'alltoall') {
      const isCombine = stage.label.indexOf('combine') >= 0;
      bandLabel(`ALL-TO-ALL ${isCombine ? 'combine' : 'dispatch'} · ${fmtB(m.crossTok * m.d * BPE)} leaves the GPUs per layer · busiest rank ${m.imbalance.toFixed(2)}× the mean`, T.violet);
      let maxv = 0;
      for (let s = 0; s < m.N; s++) for (let dst = 0; dst < m.N; dst++) if (s !== dst) maxv = Math.max(maxv, m.a2a[s][dst]);
      for (let s = 0; s < m.N; s++) {
        for (let dst = 0; dst < m.N; dst++) {
          if (s === dst || maxv <= 0) continue;
          const v = m.a2a[s][dst], t = v / maxv;
          if (t < 0.005) continue;
          const a = cols[isCombine ? dst : s], b = cols[isCombine ? s : dst];
          const lift = 14 + Math.abs(dst - s) * (bandH * 0.16);
          ctx.save();
          ctx.strokeStyle = alphaOf(T.violet, 0.2 + 0.7 * t);
          ctx.fillStyle = alphaOf(T.violet, 0.2 + 0.7 * t);
          ctx.lineWidth = 0.8 + 3.4 * t;
          curve(ctx, a.cx, yMid, b.cx, yMid, lift);
          arrowHead(ctx, b.cx, yMid, b.cx > a.cx ? 0.9 : Math.PI - 0.9, 7);
          ctx.restore();
          packet(a.cx, yMid, b.cx, yMid, lift, T.violet, (s * m.N + dst) / (m.N * m.N));
          wires.push({
            x1: a.cx, y1: yMid, x2: b.cx, y2: yMid, lift, mid: { x: (a.cx + b.cx) / 2, y: yMid - lift * 0.55 },
            why: `ALL-TO-ALL ${isCombine ? 'combine' : 'dispatch'} · GPU ${a.i} → GPU ${b.i}\n` +
              `${v.toFixed(1)} token-assignments × ${m.d} × ${BPE} B = ${fmtB(v * m.d * BPE)} per layer\n` +
              `GPU ${s} holds ${m.toksOn[s]} of the ${m.B} tokens; ${(100 * m.pRank[dst]).toFixed(1)}% of routed\n` +
              `assignments want an expert that lives on GPU ${dst} (it holds ${m.expertsOn[dst]} of ${m.E}).\n` +
              `That percentage is the ROUTER's, not the topology's — skew the router and this\n` +
              `link's volume changes while its neighbours' do not. Busiest rank right now:\n` +
              `${m.imbalance.toFixed(2)}× the fair share, and the step waits for it.`,
          });
        }
      }
    } else if (stage.kind === 'p2p') {
      const b0 = Math.min(m.N - 2, Math.floor((phase * 0.5) % Math.max(1, m.N - 1)));
      bandLabel(`POINT-TO-POINT · ${fmtB(m.payload)} per stage boundary · ${m.N - 1} boundaries per pass · bubble ${(100 * m.bubble).toFixed(0)}% at ${st.micro} micro-batch${st.micro > 1 ? 'es' : ''}`, T.ok);
      for (let i = 0; i < m.N - 1; i++) {
        const a = cols[i], b = cols[i + 1], live = i === b0;
        ctx.save();
        ctx.strokeStyle = alphaOf(T.ok, live ? 0.95 : 0.28);
        ctx.fillStyle = alphaOf(T.ok, live ? 0.95 : 0.28);
        ctx.lineWidth = live ? 3 : 1.4;
        curve(ctx, a.cx, yMid, b.cx, yMid, 16);
        arrowHead(ctx, b.cx, yMid, 0.9, 8);
        ctx.restore();
        if (live) packet(a.cx, yMid, b.cx, yMid, 16, T.ok, 0);
        wires.push({
          x1: a.cx, y1: yMid, x2: b.cx, y2: yMid, lift: 16, mid: { x: (a.cx + b.cx) / 2, y: yMid - 9 },
          why: `PIPELINE stage boundary · GPU ${i} → GPU ${i + 1}\n` +
            `GPU ${i} holds ${m.layersOn[i]} whole layers; when it finishes them it hands the\n` +
            `[${m.B} × ${m.d}] activation on — ${fmtB(m.payload)}, one send, no collective.\n` +
            `That is the cheapest traffic on this page and it does NOT grow with layer count.\n` +
            `The bill is idleness: with ${st.micro} micro-batch${st.micro > 1 ? 'es' : ''} across ${m.N} stages the pipeline is\n` +
            `${(100 * m.bubble).toFixed(0)}% bubble, and a single decode token can only ever be in one stage,\n` +
            `so pipelining does nothing at all for single-token latency.`,
        });
      }
      // bubble strip
      const bw = (W - 2 * pad) * m.bubble;
      if (bw > 1) {
        hatch(ctx, pad, bandY1 - 16, bw, 11, alphaOf(T.warn, 0.8), alphaOf(T.warn, 0.14));
        r.label('pipeline bubble', pad + bw + 6, bandY1 - 7, { color: T.warnDeep, font: '9.5px ui-monospace, monospace' });
      }
    } else if (stage.kind === 'sync') {
      const idle = m.gpus.filter((g) => g.toks === 0).length;
      bandLabel(`RANK SYNC · every rank must enter the MoE collective${idle ? ` · ${idle} rank(s) have no requests and run a DUMMY forward` : ''}`, T.warn);
      for (let i = 0; i < m.N; i++) {
        const c = cols[i], empty = c.g.toks === 0;
        ctx.save();
        ctx.strokeStyle = alphaOf(empty ? T.warn : T.ok, 0.8);
        ctx.setLineDash(empty ? [4, 3] : []);
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(c.cx, yMid - 16); ctx.lineTo(c.cx, yMid + 16); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        r.label(empty ? 'dummy' : `${c.g.toks} tok`, c.cx, yMid + 30, { color: empty ? T.warn : T.n11, font: '9.5px ui-monospace, monospace', align: 'center' });
      }
      ctx.save();
      ctx.strokeStyle = alphaOf(T.warn, 0.6); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(cols[0].cx, yMid); ctx.lineTo(cols[m.N - 1].cx, yMid); ctx.stroke();
      ctx.restore();
    } else if (stage.kind === 'compute') {
      bandLabel(stage.role === 'attn'
        ? (m.attn === 'tp' ? `each GPU multiplies its 1/${m.N} slice of every attention matrix — the result is PARTIAL`
          : m.attn === 'pp' ? 'the GPU that owns this layer runs the whole attention sublayer'
            : 'each GPU runs the WHOLE attention sublayer, on ITS OWN requests and ITS OWN KV')
        : stage.role === 'router'
          ? `the router scores every token against all ${m.E} experts and keeps its top ${m.k} — this is what decides the traffic that follows`
          : (m.moe === 'tp' ? `each GPU multiplies its 1/${m.N} slice of every expert matrix — the result is PARTIAL`
            : m.moe === 'pp' ? 'the GPU that owns this layer runs the whole MoE sublayer'
              : 'each GPU runs the experts it holds, on whatever tokens arrived for them'), T.n11);
      for (let i = 0; i < m.N; i++) {
        const c = cols[i];
        ctx.save();
        ctx.strokeStyle = alphaOf(stage.part === 'attn' ? COL_A : COL_M, 0.55);
        ctx.lineWidth = 1.6;
        roundRect(ctx, c.cx - 22, yMid - 14, 44, 28, 5); ctx.stroke();
        ctx.fillStyle = alphaOf(stage.part === 'attn' ? COL_A : COL_M, 0.16); ctx.fill();
        ctx.restore();
        const busy = stage.role === 'expert' && m.moe === 'ep' ? `${m.recv[i].toFixed(0)} tok`
          : stage.role === 'router' ? (m.attn === 'dp' ? `${c.g.toks} tok` : 'route')
            : stage.role === 'attn' && m.attn === 'dp' ? `${c.g.toks} req` : 'math';
        r.label(busy, c.cx, yMid + 4, { color: T.n12, font: '9.5px ui-monospace, monospace', align: 'center' });
      }
    } else {
      bandLabel(stage.kind === 'none'
        ? (stage.part === 'attn' && m.attn === 'dp'
          ? 'NOTHING crosses: data-parallel attention keeps each rank\'s requests and KV local'
          : 'nothing crosses the wire at this stage')
        : stage.label, stage.kind === 'none' && m.attn === 'dp' ? T.ok : T.n10);
      for (let i = 0; i < m.N; i++) {
        const c = cols[i];
        ctx.save();
        ctx.strokeStyle = alphaOf(T.n9, 0.5); ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(c.cx, yMid, 13, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]); ctx.restore();
      }
    }

    // ---- the wire-bytes bar ----------------------------------------------
    const barY = H - 34, barH = 13, barW = W - 2 * pad;
    const total = Math.max(1, m.attnWire + m.moeWire);
    const aw = barW * (m.attnWire / total), mw = barW * (m.moeWire / total);
    ctx.save();
    ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(pad, barY, barW, barH);
    ctx.fillStyle = alphaOf(COL_A, 0.8); ctx.fillRect(pad, barY, aw, barH);
    ctx.fillStyle = alphaOf(COL_M, 0.8); ctx.fillRect(pad + aw, barY, mw, barH);
    ctx.restore();
    r.label(`wire / decode step: ${fmtB(m.wire)} fabric · ${fmtB(m.wirePerGpu)} per GPU · ${m.commMs.toFixed(2)} ms @ ${st.link} GB/s`,
      pad, barY - 5, { color: T.n12, font: '10.5px ui-monospace, monospace' });
    // memory legend, right-aligned on the same line so it never crowds the strip
    {
      const lg = [['KV', COL_KV], ['expert w', COL_M], ['attn w', COL_A]];
      ctx.save();
      ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      let lx = W - pad;
      for (const [txt, col] of lg) {
        const tw = ctx.measureText(txt).width;
        lx -= tw;
        ctx.fillStyle = T.n11; ctx.fillText(txt, lx, barY - 5);
        lx -= 12; ctx.fillStyle = alphaOf(col, 0.85); ctx.fillRect(lx, barY - 13, 9, 9);
        lx -= 14;
      }
      ctx.restore();
    }
    if (aw > 60) r.label(`attention ${fmtB(m.attnWire)}`, pad + 5, barY + barH - 3.5, { color: inkOn(COL_A), font: '9.5px ui-monospace, monospace' });
    if (mw > 60) r.label(`MoE ${fmtB(m.moeWire)}`, pad + aw + 5, barY + barH - 3.5, { color: inkOn(COL_M), font: '9.5px ui-monospace, monospace' });
    if (m.wire === 0) r.label('nothing crosses the wire in this configuration', pad + 5, barY + barH - 3.5, { color: T.n11, font: '9.5px ui-monospace, monospace' });

    // ---- hover ------------------------------------------------------------
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      let tip = null;
      // arrows first: they sit on top
      let best = 1e9, hit = null;
      for (const wr of wires) {
        const dx = p.x - wr.mid.x, dy = p.y - wr.mid.y, dd = dx * dx + dy * dy;
        if (dd < best && dd < 26 * 26) { best = dd; hit = wr; }
      }
      if (hit) tip = hit.why;
      if (!tip) {
        for (const c of cols) {
          if (p.x >= c.x - 4 && p.x <= c.x + c.w + 4 && p.y >= c.top - 44 && p.y <= c.bot + 30) {
            const g = c.g;
            const attnWhy = m.attn === 'tp' ? `1/${m.N} of every attention matrix (a shard of each head's projections)`
              : m.attn === 'pp' ? `the WHOLE attention sublayer, for its ${g.layers} of ${m.L} layers`
                : `a FULL COPY of every attention layer — ${m.N} copies exist across the strip`;
            const moeWhy = m.moe === 'ep' ? `${g.experts} of the ${m.E} experts, in all ${m.L} layers`
              : m.moe === 'tp' ? `1/${m.N} of every expert matrix, in all ${m.L} layers`
                : `every expert, for its ${g.layers} of ${m.L} layers`;
            const kvWhy = m.attn === 'dp' ? `KV for its OWN ${g.toks} request(s) — request ownership, not sharding`
              : m.attn === 'tp' ? (m.kvHeads <= 1
                ? `a FULL COPY of the KV cache: a latent cache is ONE head, so tensor parallel has nothing to split — every rank keeps all of it`
                : `KV split ${Math.min(m.N, m.kvHeads)} ways (only ${m.kvHeads} KV heads exist to give away)`)
                : `KV for its ${g.layers} of ${m.L} layers, all ${m.B} requests`;
            tip = `GPU ${c.i} holds\n` +
              `attention weights ${fmtB(g.aw)} — ${attnWhy}\n` +
              `expert weights    ${fmtB(g.mw)} — ${moeWhy}\n` +
              `KV cache          ${fmtB(g.kv)} — ${kvWhy}\n` +
              `workspace         ${fmtB(g.over)}\n` +
              `total             ${fmtB(g.total)} of a ${st.budget} GB budget${g.total > m.budget ? '  ← DOES NOT FIT' : ''}` +
              (m.attn === 'dp' && g.toks === 0 ? `\nno requests of its own: it still runs a DUMMY forward pass so the\nMoE collective has every rank present — otherwise the ranks that do\nhave work block forever waiting for this one.` : '') +
              `\nclick a chip above the column to switch that sublayer's strategy`;
            break;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout ----------------------------------------------------------
    const dupTxt = m.dup > 1.01 ? `${m.dup.toFixed(2)}× the model's weights are resident across the strip (replication)` : 'weights are sharded, not replicated';
    let o = `attention ${SHORT[m.attn]} · MoE ${SHORT[m.moe]} · ${m.N} GPU${m.N > 1 ? 's' : ''} · ${m.latent ? 'latent' : 'grouped'} KV     tier:${r.name}\n`;
    o += `WIRE  ${fmtB(m.wire)}/decode step across the fabric (${fmtB(m.wirePerGpu)} per GPU = ${m.commMs.toFixed(2)} ms at ${st.link} GB/s) — attention ${fmtB(m.attnWire)}, MoE ${fmtB(m.moeWire)}`;
    o += m.moe === 'ep' ? ` (all-to-all, busiest rank ${m.imbalance.toFixed(2)}× the mean — routing decides this, not the topology)\n` : '\n';
    o += `MEM   peak per GPU ${fmtB(m.peak)} of ${st.budget} GB — ${m.fits ? 'fits' : 'DOES NOT FIT'}; ${dupTxt}; KV ${fmtB(m.kvTotal)} total`;
    o += m.attn === 'tp' && m.kvHeads <= 1 ? `, and tensor parallel cannot shard a single-head latent cache — every rank keeps a whole copy.\n` : `.\n`;
    o += m.anyPP ? `PIPE  ${(100 * m.bubble).toFixed(0)}% bubble at ${st.micro} micro-batch${st.micro > 1 ? 'es' : ''} across ${m.N} stages; pipelining buys cheap comms and buys nothing for single-token latency.\n`
      : `TRADE ${m.attn === 'tp' || m.moe === 'tp' ? 'an all-reduce sits on the critical path of every sublayer of every layer, so a slower fabric shows up directly as latency. ' : ''}${m.attn === 'dp' ? 'attention crosses nothing, at the price of a whole copy of the attention weights per rank. ' : ''}${m.moe === 'ep' ? 'expert memory scales with GPU count, but the all-to-all is routing-dependent and load-imbalanced.' : ''}\n`;
    o += `Context parallelism — splitting one SEQUENCE across ranks — is a different axis and is not drawn here.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__parPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k, key, lo, hi) => { if (q.has(k)) page.controls.set(key, clamp(parseFloat(q.get(k)) || lo, lo, hi), { rebuild: true, silent: true }); };
  const pick = (k, key, allowed) => { if (q.has(k) && allowed.includes(q.get(k))) page.controls.set(key, q.get(k), { rebuild: true, silent: true }); };
  num('gpus', 'gpus', 1, 8);
  pick('attn', 'attn', ATTN_MODES);
  pick('moe', 'moe', MOE_MODES);
  pick('kv', 'kv', ['gqa', 'latent']);
  num('d', 'd', 1024, 8192);
  num('layers', 'layers', 8, 96);
  num('experts', 'experts', 8, 128);
  num('topk', 'topk', 1, 8);
  num('reqs', 'reqs', 1, 256);
  num('seq', 'seq', 512, 65536);
  num('skew', 'skew', 0, 3);
  num('micro', 'micro', 1, 16);
  num('link', 'link', 10, 1000);
  num('budget', 'budget', 16, 192);
  num('overhead', 'overhead', 0, 16);
  m = build(page.state);
  if (t) t.rebuild();
  // ?hover=x,y fakes the cursor (a screenshot run has no pointer). Canvas px.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause before seeking so autoplay does not
  // advance off the requested step.
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
