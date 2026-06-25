// WebGPU compute backend for the real-attention page — the SAME GPT-2 forward as
// gpt2.js, but the heavy work (Conv1D matmuls, LayerNorm, causal attention,
// gelu_new) runs as WGSL compute dispatches instead of CPU loops. Educational:
// you can watch the exact attention math run on the GPU. The CPU gpt2.js stays
// the verified ground truth + fallback; the page self-checks that this backend
// reproduces it at runtime (max|Δ|, auto-fallback on mismatch) — see page.js.
//
// Design: the token+position embedding gather stays on the CPU (so the 154 MB
// wte never has to go to the GPU); the resulting [n×D] hidden state is uploaded
// and the 12 transformer blocks run fully on-device, keeping activations
// resident across layers. Only the attention matrices A[layer][head] are read
// back (that's what the page draws). Same return shape as GPT2.forward().
//
// Each op is its OWN shader module (no @binding overlap across entry points,
// which some WebGPU implementations reject); every module binds 0,1,2,… in the
// order _pass() passes its buffers.
const D32 = 'struct Dims { a:u32, b:u32, c:u32, d:u32 };\n';
const SHADERS = {
  // Y[i,o] = (d==1 ? B[o] : 0) + Σ_k X[i,k]·W[k,o]   ; W row-major [din,dout]
  matmul: D32 + `
    @group(0) @binding(0) var<storage,read> X:array<f32>;
    @group(0) @binding(1) var<storage,read> W:array<f32>;
    @group(0) @binding(2) var<storage,read> B:array<f32>;
    @group(0) @binding(3) var<storage,read_write> Y:array<f32>;
    @group(0) @binding(4) var<uniform> P:Dims;            // n, din, dout, hasBias
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let dout=P.c; let idx=g.x; if(idx>=P.a*dout){return;}
      let i=idx/dout; let o=idx%dout;
      var acc:f32=0.0; if(P.d==1u){acc=B[o];}
      for(var k=0u;k<P.b;k=k+1u){acc=acc+X[i*P.b+k]*W[k*dout+o];}
      Y[idx]=acc;
    }`,
  // out[i,j] = (x[i,j]-mean_i)/sqrt(var_i+1e-5)·G[j]+Bn[j]   (biased var)
  layernorm: D32 + `
    @group(0) @binding(0) var<storage,read> X:array<f32>;
    @group(0) @binding(1) var<storage,read> G:array<f32>;
    @group(0) @binding(2) var<storage,read> Bn:array<f32>;
    @group(0) @binding(3) var<storage,read_write> Y:array<f32>;
    @group(0) @binding(4) var<uniform> P:Dims;            // n, D
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let D=P.b; let i=g.x; if(i>=P.a){return;}
      var m:f32=0.0; for(var j=0u;j<D;j=j+1u){m=m+X[i*D+j];} m=m/f32(D);
      var v:f32=0.0; for(var j=0u;j<D;j=j+1u){let t=X[i*D+j]-m; v=v+t*t;} v=v/f32(D);
      let inv=1.0/sqrt(v+1e-5);
      for(var j=0u;j<D;j=j+1u){Y[i*D+j]=(X[i*D+j]-m)*inv*G[j]+Bn[j];}
    }`,
  // per (head h, query i): causal scaled-dot scores over j≤i, softmax in place
  attn: D32 + `
    @group(0) @binding(0) var<storage,read> QKV:array<f32>;   // [n, 3D]
    @group(0) @binding(1) var<storage,read_write> A:array<f32>; // [H, n, n]
    @group(0) @binding(2) var<uniform> P:Dims;                 // n, D, H
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let n=P.a; let D=P.b; let H=P.c; let dh=D/H; let idx=g.x; if(idx>=H*n){return;}
      let h=idx/n; let i=idx%n; let scale=1.0/sqrt(f32(dh));
      let qoff=i*3u*D + h*dh; let abase=h*n*n + i*n;
      var mx:f32=-1e30;
      for(var j=0u;j<=i;j=j+1u){
        let koff=j*3u*D + D + h*dh; var s:f32=0.0;
        for(var c=0u;c<dh;c=c+1u){s=s+QKV[qoff+c]*QKV[koff+c];}
        s=s*scale; A[abase+j]=s; if(s>mx){mx=s;}
      }
      var sum:f32=0.0;
      for(var j=0u;j<=i;j=j+1u){let e=exp(A[abase+j]-mx); A[abase+j]=e; sum=sum+e;}
      for(var j=0u;j<=i;j=j+1u){A[abase+j]=A[abase+j]/sum;}
      for(var j=i+1u;j<n;j=j+1u){A[abase+j]=0.0;}
    }`,
  // ctx[i,d] = Σ_{j≤i} A[head(d),i,j]·V[j,d]   (V at offset 2D+d in qkv)
  context: D32 + `
    @group(0) @binding(0) var<storage,read> QKV:array<f32>;
    @group(0) @binding(1) var<storage,read> A:array<f32>;
    @group(0) @binding(2) var<storage,read_write> O:array<f32>;  // [n, D]
    @group(0) @binding(3) var<uniform> P:Dims;                   // n, D, H
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let n=P.a; let D=P.b; let H=P.c; let dh=D/H; let idx=g.x; if(idx>=n*D){return;}
      let i=idx/D; let d=idx%D; let head=d/dh; let abase=head*n*n + i*n;
      var acc:f32=0.0;
      for(var j=0u;j<=i;j=j+1u){acc=acc+A[abase+j]*QKV[j*3u*D + 2u*D + d];}
      O[idx]=acc;
    }`,
  // gelu_new (tanh approx), in place
  gelu: D32 + `
    @group(0) @binding(0) var<storage,read_write> X:array<f32>;
    @group(0) @binding(1) var<uniform> P:Dims;                  // count
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let idx=g.x; if(idx>=P.a){return;} let v=X[idx];
      X[idx]=0.5*v*(1.0+tanh(0.7978845608028654*(v+0.044715*v*v*v)));
    }`,
  // X += Y
  addinto: D32 + `
    @group(0) @binding(0) var<storage,read_write> X:array<f32>;
    @group(0) @binding(1) var<storage,read> Y:array<f32>;
    @group(0) @binding(2) var<uniform> P:Dims;                  // count
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let idx=g.x; if(idx>=P.a){return;} X[idx]=X[idx]+Y[idx];
    }`,
  // tied lm_head: O[v] = Σ_d xf[lastRow,d]·wte[v,d]   (no transpose — wte[v,:] is a row)
  lmhead: D32 + `
    @group(0) @binding(0) var<storage,read> xf:array<f32>;      // [n, D]  (ln_f output)
    @group(0) @binding(1) var<storage,read> wte:array<f32>;     // [V, D]
    @group(0) @binding(2) var<storage,read_write> O:array<f32>; // [V]
    @group(0) @binding(3) var<uniform> P:Dims;                  // V, D, lastRow
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){
      let V=P.a; let D=P.b; let v=g.x; if(v>=V){return;}
      let base=P.c*D; var acc:f32=0.0;
      for(var d=0u;d<D;d=d+1u){acc=acc+xf[base+d]*wte[v*D+d];}
      O[v]=acc;
    }`,
};

export async function getWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return { adapter, device };
  } catch { return null; }
}

export class GPT2WebGPU {
  constructor(weights, cfg, device) { this.w = weights; this.cfg = cfg; this.device = device; this.pipes = {}; this.wbuf = new Map(); }
  g(name) { const t = this.w.get(name); if (!t) throw new Error('missing weight ' + name); return t.data; }

  async init() {
    const dev = this.device;
    for (const k in SHADERS) this.pipes[k] = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: SHADERS[k] }), entryPoint: 'main' } });
    // upload every per-layer weight (NOT wte: embedding is gathered on the CPU)
    for (let l = 0; l < this.cfg.nLayer; l++) for (const s of ['ln_1.weight', 'ln_1.bias', 'attn.c_attn.weight', 'attn.c_attn.bias', 'attn.c_proj.weight', 'attn.c_proj.bias', 'ln_2.weight', 'ln_2.bias', 'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias']) {
      const name = `h.${l}.${s}`, d = this.g(name);
      const b = dev.createBuffer({ size: d.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(b, 0, d); this.wbuf.set(name, b);
    }
  }

  _sb(bytes) { return this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); }
  _ub(arr) { const b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(b, 0, new Uint32Array([arr[0] || 0, arr[1] || 0, arr[2] || 0, arr[3] || 0])); return b; }
  _from(data) { const b = this._sb(data.byteLength); this.device.queue.writeBuffer(b, 0, data); return b; }
  _pass(enc, pipe, bufs, count) {
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
    const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(count / 64)); p.end();
  }

  // Encode the 12 transformer blocks into `enc`, leaving the final hidden state
  // resident in the returned buffer `x`. All transient buffers are pushed to
  // `tmp` (caller destroys after submit). captureA → also copy each layer's A
  // into a MAP_READ staging buffer (returned in `stagings`) for readback.
  _encodeBlocks(enc, ids, tmp, captureA) {
    const dev = this.device, { nLayer: L, nHead: H, nEmbd: D } = this.cfg, n = ids.length;
    const wte = this.g('wte.weight'), wpe = this.g('wpe.weight'), x0 = new Float32Array(n * D);
    for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) x0[i * D + j] = wte[ids[i] * D + j] + wpe[i * D + j];
    const x = this._from(x0); tmp.push(x);
    const sb = (bytes) => { const b = this._sb(bytes); tmp.push(b); return b; };
    const mm = (X, wName, bName, din, dout) => { const Y = sb(n * dout * 4); this._pass(enc, this.pipes.matmul, [X, this.wbuf.get(wName), this.wbuf.get(bName), Y, this._ub([n, din, dout, 1])], n * dout); return Y; };
    const stagings = [];
    for (let l = 0; l < L; l++) {
      const p = `h.${l}.`;
      const xn = sb(n * D * 4); this._pass(enc, this.pipes.layernorm, [x, this.wbuf.get(p + 'ln_1.weight'), this.wbuf.get(p + 'ln_1.bias'), xn, this._ub([n, D])], n);
      const qkv = mm(xn, p + 'attn.c_attn.weight', p + 'attn.c_attn.bias', D, 3 * D);
      const A = sb(H * n * n * 4); this._pass(enc, this.pipes.attn, [qkv, A, this._ub([n, D, H])], H * n);
      const ctx = sb(n * D * 4); this._pass(enc, this.pipes.context, [qkv, A, ctx, this._ub([n, D, H])], n * D);
      const attnOut = mm(ctx, p + 'attn.c_proj.weight', p + 'attn.c_proj.bias', D, D);
      this._pass(enc, this.pipes.addinto, [x, attnOut, this._ub([n * D])], n * D);
      const xn2 = sb(n * D * 4); this._pass(enc, this.pipes.layernorm, [x, this.wbuf.get(p + 'ln_2.weight'), this.wbuf.get(p + 'ln_2.bias'), xn2, this._ub([n, D])], n);
      const hid = mm(xn2, p + 'mlp.c_fc.weight', p + 'mlp.c_fc.bias', D, 4 * D);
      this._pass(enc, this.pipes.gelu, [hid, this._ub([n * 4 * D])], n * 4 * D);
      const mlpOut = mm(hid, p + 'mlp.c_proj.weight', p + 'mlp.c_proj.bias', 4 * D, D);
      this._pass(enc, this.pipes.addinto, [x, mlpOut, this._ub([n * D])], n * D);
      if (captureA) { const stage = dev.createBuffer({ size: H * n * n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); enc.copyBufferToBuffer(A, 0, stage, 0, H * n * n * 4); stagings.push(stage); }
    }
    return { x, n, stagings };
  }

  async forward(ids) {
    const dev = this.device, { nLayer: L, nHead: H } = this.cfg;
    const tmp = [], enc = dev.createCommandEncoder();
    const { x, n, stagings } = this._encodeBlocks(enc, ids, tmp, true); void x;
    dev.queue.submit([enc.finish()]);
    const attentions = [];
    for (let l = 0; l < L; l++) {
      await stagings[l].mapAsync(GPUMapMode.READ);
      const all = new Float32Array(stagings[l].getMappedRange().slice(0));
      stagings[l].unmap(); stagings[l].destroy();
      const layer = []; for (let h = 0; h < H; h++) layer.push(all.subarray(h * n * n, (h + 1) * n * n));
      attentions.push(layer);
    }
    for (const b of tmp) b.destroy();
    return { n, nLayer: L, nHead: H, attentions };
  }

  // Lazily upload the lm_head weights (wte 154 MB + ln_f) — only when logits() is
  // first needed, so attention-only use never pays for it.
  _initLMHead() {
    if (this._lmReady) return;
    const dev = this.device;
    for (const name of ['wte.weight', 'ln_f.weight', 'ln_f.bias']) {
      const d = this.g(name), b = dev.createBuffer({ size: d.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(b, 0, d); this.wbuf.set(name, b);
    }
    this._lmReady = true;
  }

  // logits(ids) -> { logits: Float32Array(V) for the LAST position, V, n }
  // blocks (resident) → ln_f on GPU → tied lm_head matmul on GPU → readback.
  async logits(ids) {
    this._initLMHead();
    const dev = this.device, { nEmbd: D } = this.cfg, V = this.g('wte.weight').length / D;
    const tmp = [], enc = dev.createCommandEncoder();
    const { x, n } = this._encodeBlocks(enc, ids, tmp, false);
    const xf = this._sb(n * D * 4); tmp.push(xf);
    this._pass(enc, this.pipes.layernorm, [x, this.wbuf.get('ln_f.weight'), this.wbuf.get('ln_f.bias'), xf, this._ub([n, D])], n);
    const lg = this._sb(V * 4); tmp.push(lg);
    this._pass(enc, this.pipes.lmhead, [xf, this.wbuf.get('wte.weight'), lg, this._ub([V, D, n - 1])], V);
    const stage = dev.createBuffer({ size: V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(lg, 0, stage, 0, V * 4);
    dev.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap(); stage.destroy();
    for (const b of tmp) b.destroy();
    return { logits, V, n };
  }
}
