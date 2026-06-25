// Minimal GPT-2 forward pass in vanilla JS — enough to compute REAL attention
// matrices for the real-attention viz page. We can't get attentions
// out of transformers.js (the ONNX export drops them — Optimum #325), so we
// fetch the raw safetensors weights and run the forward ourselves, capturing the
// softmaxed attention A[layer][head] = softmax(QKᵀ/√d + causal mask) at every
// layer. Verified against PyTorch GPT-2 (output_attentions) — see README.
//
// GPT-2 specifics that bite if you get them wrong (all confirmed against the
// openai-community/gpt2 model.safetensors header):
//   - Conv1D, not Linear: c_attn/c_proj/c_fc weights are [n_in, n_out], so
//     y = x·W + b with W row-major [in,out].
//   - c_attn output is [q | k | v] concatenated along the 2304 dim, in that order.
//   - LayerNorm uses biased (population) variance, eps 1e-5.
//   - Activation is gelu_new (the tanh approximation), not erf-GELU.
//   - dtype is F32 throughout (no fp16/bf16 conversion needed).

export const GPT2_CONFIG = { gpt2: { nLayer: 12, nHead: 12, nEmbd: 768 }, distilgpt2: { nLayer: 6, nHead: 12, nEmbd: 768 } };

// ---- safetensors parse (header = u64 LE length + JSON; then raw tensor bytes) --
export function parseSafetensors(buf) {
  const dv = new DataView(buf);
  const hlen = Number(dv.getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
  // The data section starts at 8+hlen, which is generally NOT 4-byte aligned, so
  // Float32Array views over `buf` would throw. Copy the data section once into a
  // fresh (8-aligned) ArrayBuffer; every F32 tensor offset within it is then a
  // multiple of 4 (all tensors here are F32 + contiguous from 0).
  const data = buf.slice(8 + hlen), tensors = new Map();
  for (const name in hdr) {
    if (name === '__metadata__') continue;
    const t = hdr[name];
    if (t.dtype !== 'F32') throw new Error(`${name}: expected F32, got ${t.dtype}`);
    const [s, e] = t.data_offsets;
    tensors.set(name, { shape: t.shape, data: new Float32Array(data, s, (e - s) / 4) });
  }
  return tensors;
}

// y[i,o] = b[o] + Σ_d x[i,d]·W[d,o]   (Conv1D: W is row-major [din, dout])
function linear(x, n, din, W, b, dout) {
  const out = new Float32Array(n * dout);
  for (let i = 0; i < n; i++) {
    const xi = i * din, oi = i * dout;
    if (b) out.set(b, oi); // copy bias row
    for (let d = 0; d < din; d++) {
      const xv = x[xi + d]; if (xv === 0) continue;
      const wr = d * dout;
      for (let o = 0; o < dout; o++) out[oi + o] += xv * W[wr + o];
    }
  }
  return out;
}

function layernorm(x, n, d, g, b) {
  const out = new Float32Array(n * d), eps = 1e-5;
  for (let i = 0; i < n; i++) {
    const o = i * d; let m = 0;
    for (let j = 0; j < d; j++) m += x[o + j]; m /= d;
    let v = 0; for (let j = 0; j < d; j++) { const t = x[o + j] - m; v += t * t; } v /= d;
    const inv = 1 / Math.sqrt(v + eps);
    for (let j = 0; j < d; j++) out[o + j] = (x[o + j] - m) * inv * g[j] + b[j];
  }
  return out;
}

const C = Math.sqrt(2 / Math.PI);
function geluNew(x) { for (let i = 0; i < x.length; i++) { const v = x[i]; x[i] = 0.5 * v * (1 + Math.tanh(C * (v + 0.044715 * v * v * v))); } return x; }

export class GPT2 {
  constructor(weights, cfg) { this.w = weights; this.cfg = cfg; }
  g(name) { const t = this.w.get(name); if (!t) throw new Error('missing weight ' + name); return t.data; }

  // _run(ids, wantAttn, wantHidden) -> { x, attentions: A[L][H]|null,
  //   hiddens: (Float32Array[n×D])[L+1]|null (residual stream: embed, then after
  //   each block), n }. Block math is identical regardless of what's captured.
  _run(ids, wantAttn, wantHidden) {
    const { nLayer: L, nHead: H, nEmbd: D } = this.cfg, n = ids.length, dh = D / H, scale = 1 / Math.sqrt(dh);
    const wte = this.g('wte.weight'), wpe = this.g('wpe.weight');
    let x = new Float32Array(n * D);
    for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) x[i * D + j] = wte[ids[i] * D + j] + wpe[i * D + j];

    const attentions = wantAttn ? [] : null;
    const hiddens = wantHidden ? [x.slice()] : null;   // residual stream: pre-block embedding
    for (let l = 0; l < L; l++) {
      const p = `h.${l}.`;
      // --- attention ---
      const xn = layernorm(x, n, D, this.g(p + 'ln_1.weight'), this.g(p + 'ln_1.bias'));
      const qkv = linear(xn, n, D, this.g(p + 'attn.c_attn.weight'), this.g(p + 'attn.c_attn.bias'), 3 * D);
      const ctx = new Float32Array(n * D), Alayer = [];
      for (let h = 0; h < H; h++) {
        const qo = h * dh, ko = D + h * dh, vo = 2 * D + h * dh;
        const A = new Float32Array(n * n);
        for (let i = 0; i < n; i++) {
          const qi = i * (3 * D) + qo;
          let mx = -Infinity;
          for (let j = 0; j <= i; j++) { // causal: only j ≤ i
            let s = 0; const kj = j * (3 * D) + ko;
            for (let c = 0; c < dh; c++) s += qkv[qi + c] * qkv[kj + c];
            s *= scale; A[i * n + j] = s; if (s > mx) mx = s;
          }
          let sum = 0;
          for (let j = 0; j <= i; j++) { const e = Math.exp(A[i * n + j] - mx); A[i * n + j] = e; sum += e; }
          for (let j = 0; j <= i; j++) A[i * n + j] /= sum;
          // context = Σ_j A[i,j]·v_j
          for (let j = 0; j <= i; j++) { const a = A[i * n + j], vj = j * (3 * D) + vo; for (let c = 0; c < dh; c++) ctx[i * D + qo + c] += a * qkv[vj + c]; }
        }
        Alayer.push(A);
      }
      const attnOut = linear(ctx, n, D, this.g(p + 'attn.c_proj.weight'), this.g(p + 'attn.c_proj.bias'), D);
      for (let i = 0; i < n * D; i++) x[i] += attnOut[i]; // residual
      // --- MLP ---
      const xn2 = layernorm(x, n, D, this.g(p + 'ln_2.weight'), this.g(p + 'ln_2.bias'));
      const hid = geluNew(linear(xn2, n, D, this.g(p + 'mlp.c_fc.weight'), this.g(p + 'mlp.c_fc.bias'), 4 * D));
      const mlpOut = linear(hid, n, 4 * D, this.g(p + 'mlp.c_proj.weight'), this.g(p + 'mlp.c_proj.bias'), D);
      for (let i = 0; i < n * D; i++) x[i] += mlpOut[i]; // residual
      if (wantAttn) attentions.push(Alayer);
      if (wantHidden) hiddens.push(x.slice());
    }
    return { x, attentions, hiddens, n };
  }

  // forward(ids) -> { n, nLayer, nHead, attentions: A[L][H] each Float32Array(n*n) }
  forward(ids) { const r = this._run(ids, true); return { n: r.n, nLayer: this.cfg.nLayer, nHead: this.cfg.nHead, attentions: r.attentions }; }

  // ln_f(hiddenRow) → tied lm_head (= wteᵀ, no bias): logit[v] = Σ_d ln_f(row)[d]·wte[v,d]
  _head(row) {
    const { nEmbd: D } = this.cfg, wte = this.g('wte.weight'), V = wte.length / D;
    const xf = layernorm(row, 1, D, this.g('ln_f.weight'), this.g('ln_f.bias'));
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) { let s = 0; const o = v * D; for (let d = 0; d < D; d++) s += xf[d] * wte[o + d]; logits[v] = s; }
    return logits;
  }

  // logits(ids) -> { logits: Float32Array(V) for the LAST position, V, n }
  logits(ids) {
    const { nEmbd: D } = this.cfg, r = this._run(ids, false), n = r.n;
    return { logits: this._head(r.x.subarray((n - 1) * D, n * D)), V: this.g('wte.weight').length / D, n };
  }

  // lens(ids) -> { perLayer: Float32Array(V)[nLayer+1], V, n, nLayer } — the "logit
  // lens": the final ln_f + lm_head applied to the LAST position's residual stream
  // at every depth (embedding, then after each block), so you can watch the
  // next-token prediction form across layers.
  lens(ids) {
    const { nEmbd: D } = this.cfg, r = this._run(ids, false, true), n = r.n;
    const perLayer = r.hiddens.map((h) => this._head(h.subarray((n - 1) * D, n * D)));
    return { perLayer, V: this.g('wte.weight').length / D, n, nLayer: this.cfg.nLayer };
  }
}

// Stream-fetch the safetensors with progress, then parse + wrap.
export async function loadGPT2(url, cfg, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} → HTTP ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader(), chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (onProgress && total) onProgress(got / total);
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return new GPT2(parseSafetensors(buf.buffer), cfg);
}

// Fetch a SINGLE F32 tensor from a safetensors file via HTTP Range requests (the
// header, then just that tensor's bytes) — ~MBs instead of the whole 548 MB file.
// HF resolve URLs sit on a CDN that honours Range. Returns { shape, data:Float32Array }.
export async function fetchTensor(url, name) {
  const rng = async (a, b) => {
    const r = await fetch(url, { headers: { Range: `bytes=${a}-${b}` } });
    if (!r.ok) throw new Error(`range fetch → HTTP ${r.status}`);
    const ab = await r.arrayBuffer();
    if (ab.byteLength !== b - a + 1) throw new Error(`server ignored Range (got ${ab.byteLength}B, wanted ${b - a + 1}B) — needs a Range-capable host (the HF CDN is)`);
    return ab;
  };
  const hlen = Number(new DataView(await rng(0, 7)).getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(await rng(8, 8 + hlen - 1))));
  const t = hdr[name];
  if (!t) throw new Error(`tensor ${name} not in safetensors`);
  if (t.dtype !== 'F32') throw new Error(`${name}: expected F32, got ${t.dtype}`);
  const base = 8 + hlen, [s, e] = t.data_offsets;
  const tb = await rng(base + s, base + e - 1);                 // fresh ArrayBuffer → 4-aligned
  return { shape: t.shape, data: new Float32Array(tb) };
}
