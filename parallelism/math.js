// Device-parallel traffic for one transformer layer.
// tp / pp / dp / ep are the original branches. kvp shards the sequence for
// attention and gathers cache bytes; it is not the expert-token all-to-all
// and it is not the ring on the context-parallelism page.

export const BPE = 2;
const GQA = 8;
const HEAD = 128;
const D_LATENT = 576;

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const ATTN_MODES = ['tp', 'pp', 'dp', 'kvp'];
export const MOE_MODES = ['tp', 'pp', 'ep'];

export const NAME = {
  tp: 'tensor parallel', pp: 'pipeline parallel',
  ep: 'expert parallel', dp: 'data-parallel (replicated)',
  kvp: 'KV-parallel',
};
export const SHORT = { tp: 'TP', pp: 'PP', ep: 'EP', dp: 'DP', kvp: 'KV' };

export function fmtB(b) {
  if (!isFinite(b)) return '–';
  if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' kB';
  return b.toFixed(0) + ' B';
}

const share = (M, N) => Array.from({ length: N }, (_, i) => Math.floor(M / N) + (i < M % N ? 1 : 0));

export function build(st) {
  const N = clamp(st.gpus | 0, 1, 8);
  const d = st.d | 0, L = st.layers | 0, E = st.experts | 0;
  const k = Math.min(st.topk | 0, E);
  const B = st.reqs | 0;
  const S = st.seq | 0;
  const dff = Math.round(d / 2);
  const attn = ATTN_MODES.includes(st.attn) ? st.attn : 'tp';
  const moe = MOE_MODES.includes(st.moe) ? st.moe : 'ep';
  const latent = st.kv === 'latent';

  const attnParams = latent
    ? 2 * d * d + 2 * d * D_LATENT
    : 2 * d * d + 2 * d * (d / GQA);
  const moeParams = E * 3 * d * dff;
  const attnBytes = L * attnParams * BPE;
  const moeBytes = L * moeParams * BPE;

  const kvHeads = latent ? 1 : Math.max(1, Math.round(d / HEAD / GQA));
  const kvPerTok = latent ? L * D_LATENT * BPE : 2 * L * kvHeads * HEAD * BPE;
  const kvTotal = B * S * kvPerTok;

  const layersOn = share(L, N);
  const expertsOn = share(E, N);
  const toksOn = share(B, N);

  const gpus = [];
  for (let r = 0; r < N; r++) {
    const aw = attn === 'dp' ? attnBytes
      : attn === 'tp' || attn === 'kvp' ? attnBytes / N
        : attnBytes * (layersOn[r] / L);
    const mw = moe === 'ep' ? L * expertsOn[r] * 3 * d * dff * BPE
      : moe === 'tp' ? moeBytes / N
        : moeBytes * (layersOn[r] / L);
    const kv = attn === 'dp' ? toksOn[r] * S * kvPerTok
      : attn === 'kvp' ? kvTotal / N
        : attn === 'tp' ? kvTotal / Math.min(N, kvHeads)
          : kvTotal * (layersOn[r] / L);
    const over = st.overhead * 1e9;
    gpus.push({ r, aw, mw, kv, over, total: aw + mw + kv + over, layers: layersOn[r], experts: expertsOn[r], toks: toksOn[r] });
  }

  const w = new Float64Array(E); let sw = 0;
  for (let e = 0; e < E; e++) { w[e] = Math.pow(1 / (1 + e), st.skew); sw += w[e]; }
  for (let e = 0; e < E; e++) w[e] /= sw;
  const pRank = new Float64Array(N);
  for (let e = 0; e < E; e++) pRank[e % N] += w[e];

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

  const payload = B * d * BPE;
  const allreduce = 2 * (N - 1) * payload;
  const attnWire = N < 2 ? 0
    : attn === 'tp' ? L * allreduce
      : attn === 'kvp' ? 2 * (N - 1) * (kvTotal / N)
        : 0;
  const dispatch = crossTok * d * BPE;
  const moeWire = N < 2 ? 0
    : moe === 'tp' ? L * allreduce
      : moe === 'pp' ? 0
        : L * 2 * dispatch;
  const ppWire = N < 2 || (attn !== 'pp' && moe !== 'pp') ? 0 : (N - 1) * payload;
  const wire = attnWire + moeWire + ppWire;
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
    payload, attnWire, moeWire, ppWire, wire, wirePerGpu, commMs, anyPP, bubble,
    budget, peak, fits, fleetWeights, modelWeights, dup: fleetWeights / modelWeights,
    layersOn, expertsOn, toksOn,
  };
}

export function stages(m) {
  const a = m.attn, mo = m.moe;
  const acomm = m.N < 2 ? 'none' : a === 'tp' ? 'allreduce' : a === 'pp' ? 'p2p' : a === 'kvp' ? 'kvgather' : 'none';
  const mcomm = m.N < 2 ? 'none' : mo === 'tp' ? 'allreduce' : mo === 'pp' ? 'p2p' : 'alltoall';
  return [
    { kind: 'idle', part: 'in', role: 'in', label: 'layer input — where the activations already are' },
    { kind: 'compute', part: 'attn', role: 'attn', label: `attention math (${NAME[a]})` },
    { kind: acomm, part: 'attn', role: 'attn-comm', label: acomm === 'allreduce' ? 'attention ALL-REDUCE — every rank held a partial sum' : acomm === 'p2p' ? 'stage boundary — point-to-point activation send' : acomm === 'kvgather' ? 'KV gather — sequence shards of the cache cross; expert tokens do not' : 'attention: NOTHING crosses the wire' },
    { kind: mo === 'ep' && a === 'dp' ? 'sync' : 'compute', part: 'moe', role: 'router', label: mo === 'ep' && a === 'dp' ? 'router + rank sync (idle ranks run a dummy forward)' : 'router — each token picks its top-k experts' },
    { kind: mcomm === 'alltoall' ? 'alltoall' : 'idle', part: 'moe', role: 'dispatch', label: mcomm === 'alltoall' ? 'ALL-TO-ALL dispatch — tokens travel to their experts' : 'no dispatch: the experts are already where the tokens are' },
    { kind: 'compute', part: 'moe', role: 'expert', label: `expert math (${NAME[mo]})` },
    { kind: mcomm, part: 'moe', role: 'moe-comm', label: mcomm === 'alltoall' ? 'ALL-TO-ALL combine — expert outputs travel home' : mcomm === 'allreduce' ? 'MoE ALL-REDUCE — every rank held a partial sum' : mcomm === 'p2p' ? 'stage boundary — point-to-point activation send' : 'MoE: nothing crosses the wire' },
    { kind: 'idle', part: 'out', role: 'out', label: 'layer output — on to the next of ' + m.L + ' layers' },
  ];
}

export const cardBlurb = 'Tensor parallel, pipeline parallel, expert parallel, and data-parallel attention move different bytes at different moments. KV-parallel is an attention choice on the same devices: each GPU keeps a sequence shard of the cache and one gather moves those cache bytes, a different quantity from an activation all-reduce or the expert-token all-to-all. Data-parallel attention still crosses nothing. The ring that passes keys or queries around one sequence is the context-parallelism page.';
