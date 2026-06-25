// render.js -- shared render scaffold for viz concept pages.
//
// Tier negotiation (WebGPU -> WebGL2 -> Canvas2D) + tier-agnostic draw
// primitives. Design: plan/framework.md. The visible canvas is ALWAYS a
// Canvas2D surface -- every vector primitive (grid/vector/arrow/bar/cell)
// draws onto it, so overlays stay crisp and identical on every tier. Only
// the heatmap *raster* is produced on the negotiated tier (an offscreen
// WebGL2/WebGPU canvas, or pure 2D) and blitted into the visible canvas.
//
// All three tiers map scalars to color through ONE shared 256-entry LUT, so
// a heatmap is pixel-identical across Canvas2D and WebGL2 -- the Phase-1
// gate. The tier only changes WHERE the raster's scaling/compositing runs.
//
// No build step: import as an ES module, or load via <script> and use the
// window.VizRender global.

// ---------------------------------------------------------------------------
// Color ramps -- continuous t in [0,1] -> [r,g,b] (0..255). Multi-stop lerp.
// ---------------------------------------------------------------------------
function _lerp(a, b, t) { return a + (b - a) * t; }

function _rampFromStops(stops) {
  // stops: [[pos, [r,g,b]], ...] sorted by pos in [0,1]. Returns t -> [r,g,b].
  return (t) => {
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
        const f = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
        return [_lerp(c0[0], c1[0], f), _lerp(c0[1], c1[1], f), _lerp(c0[2], c1[2], f)];
      }
    }
    return stops[stops.length - 1][1].slice();
  };
}

// sequential: low->high magnitude (viridis-ish: deep indigo -> teal -> yellow).
const _sequential = _rampFromStops([
  [0.0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 144, 140]],
  [0.75, [93, 201, 99]], [1.0, [253, 231, 37]],
]);

// diverging: signed weights, blue (neg) -> white (0) -> red (pos). 0 at t=0.5.
const _diverging = _rampFromStops([
  [0.0, [33, 102, 172]], [0.5, [247, 247, 247]], [1.0, [178, 24, 43]],
]);

export const ramps = { sequential: _sequential, diverging: _diverging };

// categorical: discrete index -> distinct color (expert / head / class ids).
const _CAT = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207],
];
export function categorical(i) { return _CAT[((i % _CAT.length) + _CAT.length) % _CAT.length]; }

function _rgb(c, a) { return `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a == null ? 1 : a})`; }

// Build a 256x4 (RGBA, alpha=255) Uint8 LUT from a continuous ramp. Both the
// Canvas2D and GPU paths sample THIS LUT, so colors match exactly.
export function buildLUT(ramp) {
  const lut = new Uint8Array(256 * 4);
  for (let k = 0; k < 256; k++) {
    const c = ramp(k / 255);
    lut[k * 4] = c[0]; lut[k * 4 + 1] = c[1]; lut[k * 4 + 2] = c[2]; lut[k * 4 + 3] = 255;
  }
  return lut;
}

// ---------------------------------------------------------------------------
// Data coercion + scalar->index quantization (shared by all tiers).
// ---------------------------------------------------------------------------
// Accepts Float32Array/number[] (needs rows/cols), or number[][], or
// {data,rows,cols}. Returns {data: Float32Array (flat, row-major), rows, cols}.
function _coerce(data, rows, cols) {
  if (data && data.data && data.rows) return { data: Float32Array.from(data.data), rows: data.rows, cols: data.cols };
  if (Array.isArray(data) && Array.isArray(data[0])) {
    rows = data.length; cols = data[0].length;
    const flat = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) flat[r * cols + c] = data[r][c];
    return { data: flat, rows, cols };
  }
  const flat = Float32Array.from(data);
  if (!rows || !cols) { cols = flat.length; rows = 1; }
  return { data: flat, rows, cols };
}

// Resolve a [lo,hi] domain. 'auto' picks symmetric-around-0 for the diverging
// ramp (so 0 lands on white) and min..max otherwise.
function _domain(flat, domain, ramp) {
  if (Array.isArray(domain)) return domain;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < flat.length; i++) { const v = flat[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (ramp === _diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-m, m]; }
  if (lo === hi) hi = lo + 1;
  return [lo, hi];
}

// Quantize scalars to LUT indices (Uint8). idx = round(norm*255); identical on
// every tier -> identical colors.
function _toIdx(flat, lo, hi) {
  const span = hi - lo || 1, out = new Uint8Array(flat.length);
  for (let i = 0; i < flat.length; i++) {
    let n = (flat[i] - lo) / span; n = n < 0 ? 0 : n > 1 ? 1 : n;
    out[i] = (n * 255 + 0.5) | 0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tier negotiation. Returns the lowest of (preferred, available).
// ---------------------------------------------------------------------------
const TIER_RANK = { canvas2d: 1, webgl2: 2, webgpu: 3 };
// WebGPU init must not hang the page: on some headless / SwiftShader builds
// navigator.gpu exists but requestAdapter() never resolves AND never throws,
// which would hang init() (and the whole page) instead of falling back. init()
// races _initWebGPU against this timeout and degrades to WebGL2 / Canvas2D.
const WEBGPU_INIT_TIMEOUT_MS = 1500;

export function caps() {
  const r = { canvas2d: true, webgl2: false, webgpu: false };
  if (typeof navigator !== 'undefined' && navigator.gpu) r.webgpu = true;
  try { r.webgl2 = !!document.createElement('canvas').getContext('webgl2'); } catch (e) {}
  return r;
}

// Synchronous best-guess (WebGPU availability is confirmed async at init).
export function pickTier(prefer) {
  const c = caps();
  const want = TIER_RANK[prefer] || TIER_RANK.webgl2;
  if (want >= TIER_RANK.webgpu && c.webgpu) return 'webgpu';
  if (want >= TIER_RANK.webgl2 && c.webgl2) return 'webgl2';
  return 'canvas2d';
}

// ---------------------------------------------------------------------------
// Renderer.
// ---------------------------------------------------------------------------
export class Renderer {
  constructor() {
    this.tier = 'canvas2d';
    this.name = 'Canvas2D';
    this.W = 0; this.H = 0; this.dpr = 1;
    this.layout = null;          // {x,y,w,h,rows,cols,cellW,cellH} from heatmap/setGrid
    this.frame = null;           // optional math->pixel transform
    this._gl = null; this._gpu = null;
  }

  // canvas: a visible <canvas>. opts.prefer: 'webgpu'|'webgl2'|'canvas2d'.
  async init(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    const want = opts.prefer || 'webgl2';
    if (TIER_RANK[want] >= TIER_RANK.webgpu && typeof navigator !== 'undefined' && navigator.gpu) {
      // Race the WebGPU init against a timeout so a hung requestAdapter() (never
      // resolves, never throws) degrades to WebGL2 instead of hanging the page.
      try {
        const res = await Promise.race([
          this._initWebGPU().then((ok) => (ok ? 'ok' : 'unavailable')),
          new Promise((r) => setTimeout(() => r('timeout'), WEBGPU_INIT_TIMEOUT_MS)),
        ]);
        if (res === 'ok') this.tier = 'webgpu';
        else if (res === 'timeout') console.warn(`VizRender: WebGPU init timed out (${WEBGPU_INIT_TIMEOUT_MS}ms), falling back to WebGL2`);
      } catch (e) { console.warn('VizRender: WebGPU init failed, falling back ->', e); }
    }
    if (this.tier !== 'webgpu' && TIER_RANK[want] >= TIER_RANK.webgl2) {
      try { if (this._initWebGL2()) { this.tier = 'webgl2'; } }
      catch (e) { console.warn('VizRender: WebGL2 init failed, falling back ->', e); }
    }
    this.name = { canvas2d: 'Canvas2D', webgl2: 'WebGL2', webgpu: 'WebGPU' }[this.tier];
    this.resize();
    return this;
  }

  // Size the visible canvas to its CSS box at devicePixelRatio. All primitive
  // coordinates below are in LOGICAL (CSS) pixels.
  resize() {
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    const w = this.canvas.clientWidth || this.canvas.width || 640;
    const h = this.canvas.clientHeight || this.canvas.height || 480;
    this.dpr = dpr; this.W = w; this.H = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return this;
  }

  clear(color) {
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (color) { this.ctx.fillStyle = color; this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height); }
    else this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
    return this;
  }

  // Optional math-coordinate frame: map (u,v) in math units to logical pixels.
  // origin = pixel position of (0,0); unit = pixels per math unit; flipY puts +v up.
  setFrame(origin, unit, flipY = true) {
    this.frame = { ox: origin.x, oy: origin.y, unit, sy: flipY ? -1 : 1 };
    return this;
  }
  _px(p, space) {
    if (space !== 'math' || !this.frame) return p;
    const f = this.frame;
    return { x: f.ox + p.x * f.unit, y: f.oy + f.sy * p.y * f.unit };
  }

  // ----- Heatmap: the one primitive that uses the negotiated tier ----------
  // data: Float32Array|number[]|number[][]|{data,rows,cols}.
  // opts: {rows, cols, rect:{x,y,w,h}, ramp, domain:[lo,hi]|'auto', smooth:false}
  // Records this.layout for grid()/cell() to reuse the same geometry.
  heatmap(data, opts = {}) {
    const { data: flat, rows, cols } = _coerce(data, opts.rows, opts.cols);
    const ramp = opts.ramp || _sequential;
    const [lo, hi] = _domain(flat, opts.domain || 'auto', ramp);
    const idx = _toIdx(flat, lo, hi);
    const lut = buildLUT(ramp);
    const rect = opts.rect || { x: 0, y: 0, w: this.W, h: this.H };
    const smooth = !!opts.smooth;

    if (this.tier === 'webgpu') this._heatmapGPU(idx, rows, cols, lut, rect, smooth);
    else if (this.tier === 'webgl2') this._heatmapGL(idx, rows, cols, lut, rect, smooth);
    else this._heatmap2D(idx, rows, cols, lut, rect, smooth);

    this.layout = {
      x: rect.x, y: rect.y, w: rect.w, h: rect.h,
      rows, cols, cellW: rect.w / cols, cellH: rect.h / rows,
      domain: [lo, hi],
    };
    return this.layout;
  }

  // Pure-2D raster: paint cols x rows ImageData via LUT, blit scaled.
  _heatmap2D(idx, rows, cols, lut, rect, smooth) {
    const off = (this._oc2d = this._oc2d || document.createElement('canvas'));
    off.width = cols; off.height = rows;
    const octx = off.getContext('2d');
    const img = octx.createImageData(cols, rows);
    for (let i = 0; i < idx.length; i++) {
      const k = idx[i] * 4, j = i * 4;
      img.data[j] = lut[k]; img.data[j + 1] = lut[k + 1]; img.data[j + 2] = lut[k + 2]; img.data[j + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const ctx = this.ctx;
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(off, 0, 0, cols, rows, rect.x, rect.y, rect.w, rect.h);
    ctx.imageSmoothingEnabled = true;
  }

  // WebGL2 raster: upload idx as an R8 data texture + LUT as a 256x1 RGBA
  // texture; a fragment shader does idx -> LUT. GPU handles the upscale.
  _heatmapGL(idx, rows, cols, lut, rect, smooth) {
    const g = this._gl;
    const dw = Math.min(4096, Math.max(1, Math.round(rect.w * this.dpr)));
    const dh = Math.min(4096, Math.max(1, Math.round(rect.h * this.dpr)));
    if (g.canvas.width !== dw || g.canvas.height !== dh) { g.canvas.width = dw; g.canvas.height = dh; }
    const gl = g.gl;

    gl.bindTexture(gl.TEXTURE_2D, g.dataTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, cols, rows, 0, gl.RED, gl.UNSIGNED_BYTE, idx);
    const filt = smooth ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);

    gl.bindTexture(gl.TEXTURE_2D, g.lutTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut);

    gl.viewport(0, 0, dw, dh);
    gl.useProgram(g.prog);
    gl.bindVertexArray(g.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, g.dataTex); gl.uniform1i(g.uData, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, g.lutTex); gl.uniform1i(g.uLUT, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    this.ctx.imageSmoothingEnabled = smooth;
    this.ctx.drawImage(g.canvas, 0, 0, dw, dh, rect.x, rect.y, rect.w, rect.h);
    this.ctx.imageSmoothingEnabled = true;
  }

  _heatmapGPU(idx, rows, cols, lut, rect, smooth) {
    const p = this._gpu, dev = p.dev;
    const dw = Math.min(4096, Math.max(1, Math.round(rect.w * this.dpr)));
    const dh = Math.min(4096, Math.max(1, Math.round(rect.h * this.dpr)));
    if (p.canvas.width !== dw || p.canvas.height !== dh) { p.canvas.width = dw; p.canvas.height = dh; }

    // R8 data texture (row pitch must be 256-byte aligned for writeTexture).
    const pitch = Math.ceil(cols / 256) * 256;
    const buf = new Uint8Array(pitch * rows);
    for (let r = 0; r < rows; r++) buf.set(idx.subarray(r * cols, r * cols + cols), r * pitch);
    const dataTex = dev.createTexture({
      size: [cols, rows], format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    dev.queue.writeTexture({ texture: dataTex }, buf, { bytesPerRow: pitch }, [cols, rows]);

    const lutRGBA = new Uint8Array(256 * 4); lutRGBA.set(lut);
    dev.queue.writeTexture({ texture: p.lutTex }, lutRGBA, { bytesPerRow: 256 * 4 }, [256, 1]);

    const samp = smooth ? p.sampLinear : p.sampNearest;
    const bind = dev.createBindGroup({
      layout: p.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: samp },
        { binding: 1, resource: dataTex.createView() },
        { binding: 2, resource: p.sampNearest },
        { binding: 3, resource: p.lutTex.createView() },
      ],
    });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: p.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(p.pipe); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
    dev.queue.submit([enc.finish()]);
    dataTex.destroy();

    this.ctx.imageSmoothingEnabled = smooth;
    this.ctx.drawImage(p.canvas, 0, 0, dw, dh, rect.x, rect.y, rect.w, rect.h);
    this.ctx.imageSmoothingEnabled = true;
  }

  // ----- Vector-overlay primitives (always Canvas2D, tier-agnostic) --------

  // Declare a grid geometry without drawing a heatmap (grid-only pages).
  setGrid(rows, cols, rect) {
    this.layout = {
      x: rect.x, y: rect.y, w: rect.w, h: rect.h,
      rows, cols, cellW: rect.w / cols, cellH: rect.h / rows, domain: null,
    };
    return this.layout;
  }

  cellRect(r, c) {
    const L = this.layout;
    return { x: L.x + c * L.cellW, y: L.y + r * L.cellH, w: L.cellW, h: L.cellH };
  }

  // Lattice lines over the current layout.
  grid(opts = {}) {
    const L = this.layout; if (!L) return this;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = opts.stroke || 'rgba(0,0,0,0.15)';
    ctx.lineWidth = opts.width || 1;
    ctx.beginPath();
    for (let c = 0; c <= L.cols; c++) { const x = L.x + c * L.cellW; ctx.moveTo(x, L.y); ctx.lineTo(x, L.y + L.h); }
    for (let r = 0; r <= L.rows; r++) { const y = L.y + r * L.cellH; ctx.moveTo(L.x, y); ctx.lineTo(L.x + L.w, y); }
    ctx.stroke();
    ctx.restore();
    return this;
  }

  // Highlight / outline a single cell; optional fill + centered label.
  cell(r, c, opts = {}) {
    const R = this.cellRect(r, c), ctx = this.ctx;
    ctx.save();
    if (opts.fill) { ctx.fillStyle = opts.fill; ctx.fillRect(R.x, R.y, R.w, R.h); }
    if (opts.stroke !== false) {
      ctx.strokeStyle = opts.stroke || '#111'; ctx.lineWidth = opts.width || 2;
      ctx.strokeRect(R.x + 0.5, R.y + 0.5, R.w - 1, R.h - 1);
    }
    if (opts.label != null) {
      ctx.fillStyle = opts.labelColor || '#111';
      ctx.font = opts.font || '11px ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(opts.label), R.x + R.w / 2, R.y + R.h / 2);
    }
    ctx.restore();
    return this;
  }

  // Arrow from `from` to `to`. opts.weight in [0,1] scales width + opacity
  // (attention/routing strength). space: 'px' (default) | 'math'.
  arrow(from, to, opts = {}) {
    const a = this._px(from, opts.space), b = this._px(to, opts.space), ctx = this.ctx;
    const w = opts.weight == null ? 1 : Math.max(0, Math.min(1, opts.weight));
    const col = opts.color || '#222';
    const lw = (opts.width || 2) * (0.3 + 0.7 * w);
    const head = opts.head || 7 + lw;
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    ctx.save();
    ctx.globalAlpha = opts.alpha == null ? (0.25 + 0.75 * w) : opts.alpha;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(b.x - head * Math.cos(ang - 0.4), b.y - head * Math.sin(ang - 0.4));
    ctx.lineTo(b.x - head * Math.cos(ang + 0.4), b.y - head * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
    ctx.restore();
    return this;
  }

  // Vector from origin in direction `vec` (drawn as an arrow). scale multiplies
  // vec length. space: 'px' | 'math'.
  vector(origin, vec, opts = {}) {
    const s = opts.scale == null ? 1 : opts.scale;
    const to = { x: origin.x + vec.x * s, y: origin.y + vec.y * s };
    this.arrow(origin, to, opts);
    if (opts.label != null) {
      const p = this._px(to, opts.space), ctx = this.ctx;
      ctx.save();
      ctx.fillStyle = opts.color || '#222';
      ctx.font = opts.font || '12px ui-monospace, monospace';
      ctx.fillText(String(opts.label), p.x + 4, p.y - 4);
      ctx.restore();
    }
    return this;
  }

  // Bar chart inside rect (softmax / logit distributions). ramp colors by value.
  bar(values, opts = {}) {
    const rect = opts.rect || { x: 0, y: 0, w: this.W, h: this.H };
    const ramp = opts.ramp || _sequential, ctx = this.ctx;
    const n = values.length; if (!n) return this;
    let lo = opts.min, hi = opts.max;
    if (lo == null || hi == null) {
      lo = Math.min(0, ...values); hi = Math.max(...values); if (lo === hi) hi = lo + 1;
    }
    const span = hi - lo, gap = opts.gap == null ? 2 : opts.gap;
    const bw = (rect.w - gap * (n - 1)) / n, base = rect.y + rect.h;
    ctx.save();
    for (let i = 0; i < n; i++) {
      const t = (values[i] - lo) / span;
      const h = Math.max(0, t) * rect.h;
      ctx.fillStyle = opts.color || _rgb(ramp(t < 0 ? 0 : t > 1 ? 1 : t));
      const x = rect.x + i * (bw + gap);
      ctx.fillRect(x, base - h, bw, h);
    }
    ctx.restore();
    return this;
  }

  // Free text label in logical pixels.
  label(text, x, y, opts = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = opts.color || '#111';
    ctx.font = opts.font || '12px ui-monospace, monospace';
    ctx.textAlign = opts.align || 'left'; ctx.textBaseline = opts.baseline || 'alphabetic';
    ctx.fillText(String(text), x, y);
    ctx.restore();
    return this;
  }

  // ----- Tier backends -----------------------------------------------------
  _initWebGL2() {
    const off = document.createElement('canvas');
    const gl = off.getContext('webgl2', { premultipliedAlpha: false });
    if (!gl) return false;
    const VS = `#version 300 es
      const vec2 P[3] = vec2[3](vec2(-1.,-1.), vec2(3.,-1.), vec2(-1.,3.));
      out vec2 v_uv;
      void main(){ vec2 p = P[gl_VertexID]; v_uv = p*0.5+0.5; gl_Position = vec4(p,0.,1.); }`;
    const FS = `#version 300 es
      precision highp float;
      in vec2 v_uv; out vec4 frag;
      uniform sampler2D u_data;   // R8: scalar index/255 in .r
      uniform sampler2D u_lut;    // 256x1 RGBA color ramp
      void main(){
        float k = texture(u_data, vec2(v_uv.x, 1.0 - v_uv.y)).r * 255.0;  // row 0 = top
        frag = texture(u_lut, vec2((k + 0.5)/256.0, 0.5));
      }`;
    const prog = _glLink(gl, VS, FS);
    if (!prog) return false;
    const vao = gl.createVertexArray();           // attribute-less full-screen triangle
    const mkTex = () => {
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      return t;
    };
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this._gl = {
      canvas: off, gl, prog, vao,
      dataTex: mkTex(), lutTex: mkTex(),
      uData: gl.getUniformLocation(prog, 'u_data'),
      uLUT: gl.getUniformLocation(prog, 'u_lut'),
    };
    return true;
  }

  async _initWebGPU() {
    const adapter = await navigator.gpu.requestAdapter({ featureLevel: 'compatibility' })
                 || await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    const dev = await adapter.requestDevice();
    const off = document.createElement('canvas');
    const ctx = off.getContext('webgpu');
    if (!ctx) return false;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device: dev, format: fmt, alphaMode: 'opaque' });
    const WGSL = `
      struct VO { @builtin(position) pos:vec4f, @location(0) uv:vec2f };
      @vertex fn vs(@builtin(vertex_index) i:u32) -> VO {
        var p = array<vec2f,3>(vec2f(-1.,-1.), vec2f(3.,-1.), vec2f(-1.,3.));
        var o:VO; o.pos = vec4f(p[i],0.,1.); o.uv = p[i]*vec2f(0.5,-0.5)+vec2f(0.5,0.5); return o;
      }
      @group(0) @binding(0) var dataSamp: sampler;
      @group(0) @binding(1) var dataTex: texture_2d<f32>;
      @group(0) @binding(2) var lutSamp: sampler;
      @group(0) @binding(3) var lutTex: texture_2d<f32>;
      @fragment fn fs(in:VO) -> @location(0) vec4f {
        let k = textureSample(dataTex, dataSamp, in.uv).r * 255.0;
        return textureSample(lutTex, lutSamp, vec2f((k+0.5)/256.0, 0.5));
      }`;
    const mod = dev.createShaderModule({ code: WGSL });
    const pipe = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' },
    });
    const lutTex = dev.createTexture({
      size: [256, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this._gpu = {
      canvas: off, ctx, dev, pipe, lutTex,
      sampNearest: dev.createSampler({ magFilter: 'nearest', minFilter: 'nearest' }),
      sampLinear: dev.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
    };
    return true;
  }
}

function _glLink(gl, vsSrc, fsSrc) {
  const mk = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.warn('VizRender shader:', gl.getShaderInfoLog(s)); return null; }
    return s;
  };
  const vs = mk(gl.VERTEX_SHADER, vsSrc), fs = mk(gl.FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) return null;
  const p = gl.createProgram(); gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { console.warn('VizRender link:', gl.getProgramInfoLog(p)); return null; }
  return p;
}

// Pure helpers exposed for testing + reuse. quantize() + resolveDomain() are
// the SINGLE shared scalar->LUT-index path every tier feeds from -- the
// guarantee behind pixel-identical heatmaps across Canvas2D/WebGL2/WebGPU.
export { _toIdx as quantize, _domain as resolveDomain };

// Hit-test for hover/drag on a heatmap/grid: which {r,c} cell of a rows x cols
// grid laid out in `rect` ({x,y,w,h}) does the point (x,y) fall in? null if
// outside. Mirrors the cell layout heatmap()/grid() use, so a page reuses the
// same rect it drew with.
export function cellAt(rect, rows, cols, x, y) {
  if (!rect || x < rect.x || y < rect.y || x >= rect.x + rect.w || y >= rect.y + rect.h) return null;
  const c = Math.floor((x - rect.x) / (rect.w / cols));
  const r = Math.floor((y - rect.y) / (rect.h / rows));
  if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
  return { r, c };
}

// Convenience async factory.
export async function createRenderer(canvas, opts = {}) {
  return await new Renderer().init(canvas, opts);
}

// <script>-tag global (non-module pages).
if (typeof window !== 'undefined') {
  window.VizRender = { Renderer, createRenderer, pickTier, caps, ramps, categorical, buildLUT };
}
