// The hyper-connection stack the page draws. Read weights, a stand-in block,
// write weights, then the width mix. c is how much of the readout that block wrote.

import { seededRand, seededRandn } from '../framework/tensor.js';
import { rowSums, sinkhorn } from './mhc.js';

export const rmsOf = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s / v.length); };

function sublayer(x, Wb, D) {
  let ms = 0; for (let j = 0; j < D; j++) ms += x[j] * x[j];
  const inv = 1 / Math.sqrt(ms / D + 1e-6);
  const g = new Float32Array(D);
  for (let j = 0; j < D; j++) { const u = x[j] * inv; g[j] = u / (1 + Math.exp(-u)); }
  const y = new Float32Array(D);
  for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += Wb[i * D + j] * g[j]; y[i] = s * 0.6; }
  return y;
}

function eye(n, diag) { return Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, m) => (k === m ? diag : 0))); }

const MIX_SWEEPS = 40, MIX_TOL = 1e-4, MIX_MAX = 400;
function doublyStochastic(Mb) {
  let P = Mb;
  for (let s = 0; s < MIX_MAX; s++) {
    P = sinkhorn(P, MIX_SWEEPS);
    let worst = 0;
    for (const v of rowSums(P)) worst = Math.max(worst, Math.abs(v - 1));
    if (worst <= MIX_TOL) return P;
  }
  return P;
}

const mixMemo = new Map();
function mixProjected(Mb) {
  const key = Mb.map((row) => row.join(',')).join(';');
  let P = mixMemo.get(key);
  if (!P) {
    if (mixMemo.size > 512) mixMemo.clear();
    P = doublyStochastic(Mb);
    mixMemo.set(key, P);
  }
  return P;
}

export function presetWeights(preset, n, L, seed) {
  const noise = seededRand(seed * 31 + 7, [L * n * (n + 2)]);
  let p = 0; const nx = () => noise[(p++) % noise.length];
  const A = [], Bw = [], M = [];
  for (let b = 0; b < L; b++) {
    let a, w, m;
    if (preset === 'classic' || preset === 'wide-id') {
      a = new Array(n).fill(1 / n); w = new Array(n).fill(1); m = eye(n, 1);
    } else if (preset === 'strong') {
      a = new Array(n).fill(1 / n); w = new Array(n).fill(1); m = eye(n, 0.72);
    } else if (preset === 'collapsed') {
      const f = Math.pow(0.42, b + 1);
      a = new Array(n).fill(1 / n); w = new Array(n).fill(f); m = eye(n, 1);
    } else {
      const rd = b % n, wr = (b + 1) % n;
      a = Array.from({ length: n }, (_, k) => +((k === rd ? 0.72 : 0.10) + 0.18 * nx()).toFixed(3));
      w = Array.from({ length: n }, (_, k) => +((k === wr ? 0.95 : 0.22) + 0.30 * nx()).toFixed(3));
      m = Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, mm) => (
        k === mm ? +(0.86 + 0.12 * nx()).toFixed(3)
          : (mm === k + 1 ? +(0.10 + 0.16 * nx()).toFixed(3) : 0)
      )));
    }
    A.push(a); Bw.push(w); M.push(m);
  }
  return { A, Bw, M };
}

export function forward(c, project) {
  const { n, L, D, A, Bw, M, W, x0 } = c;
  const mixes = project ? M.map(mixProjected) : M;
  let Hs = Array.from({ length: n }, () => Float32Array.from(x0));
  const levels = [Hs.map((h) => Float32Array.from(h))];
  const readouts = [], blocks = [];
  const reduce = (S) => { const o = new Float32Array(D); for (let m = 0; m < n; m++) for (let j = 0; j < D; j++) o[j] += S[m][j]; for (let j = 0; j < D; j++) o[j] /= n; return o; };
  readouts.push(reduce(Hs));
  for (let b = 0; b < L; b++) {
    const x = new Float32Array(D);
    for (let m = 0; m < n; m++) { const a = A[b][m], h = Hs[m]; for (let j = 0; j < D; j++) x[j] += a * h[j]; }
    const y = sublayer(x, W[b], D);
    const Hn = Array.from({ length: n }, () => new Float32Array(D));
    for (let k = 0; k < n; k++) {
      const out = Hn[k], bw = Bw[b][k];
      for (let j = 0; j < D; j++) out[j] = bw * y[j];
      for (let m = 0; m < n; m++) { const w = mixes[b][k][m]; if (!w) continue; const h = Hs[m]; for (let j = 0; j < D; j++) out[j] += w * h[j]; }
    }
    Hs = Hn;
    levels.push(Hs.map((h) => Float32Array.from(h)));
    const ro = reduce(Hs); readouts.push(ro);
    let mw = 0; for (let k = 0; k < n; k++) mw += Bw[b][k]; mw /= n;
    const add = Float32Array.from(y, (v) => mw * v);
    blocks.push({ x, y, add, rIn: rmsOf(x), rY: rmsOf(y), c: rmsOf(add) / (rmsOf(ro) + 1e-9) });
  }
  let r = Float32Array.from(x0);
  const plain = [Float32Array.from(r)];
  for (let b = 0; b < L; b++) { const y = sublayer(r, W[b], D); for (let j = 0; j < D; j++) r[j] = r[j] + y[j]; plain.push(Float32Array.from(r)); }
  const hcOut = readouts[L], plOut = plain[L];
  let diff = 0; for (let j = 0; j < D; j++) diff = Math.max(diff, Math.abs(hcOut[j] - plOut[j]));
  const cs = blocks.map((x2) => x2.c);
  const s1 = cs.reduce((a, v) => a + v, 0), s2 = cs.reduce((a, v) => a + v * v, 0);
  const eff = s2 > 1e-12 ? (s1 * s1) / s2 : 0;
  return { levels, readouts, plain, blocks, diff, eff, hcOut, plOut, mixes };
}
