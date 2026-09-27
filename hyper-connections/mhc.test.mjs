import { readFileSync } from 'node:fs';
import { seededRandn } from '../framework/tensor.js';
import { contribLabel, depthLabel, sinkhorn, rowSums, colSums } from './mhc.js';
import { forward, presetWeights, rmsOf } from './stack.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const raw = [[2, 0.2, 0.4], [0.1, 3, 0.5], [0.3, 0.2, 1.5]];
const P = sinkhorn(raw);
const near = (v) => Math.abs(v - 1) < 1e-6;
ok(rowSums(P).every(near), 'every row of the projected mix sums to 1');
ok(colSums(P).every(near), 'every column of the projected mix sums to 1');
ok(rowSums(raw).some((v) => Math.abs(v - 1) > 0.1), 'the unprojected mix is not already doubly stochastic');

function run(preset, n, L, D, seed) {
  const { A, Bw, M } = presetWeights(preset, n, L, seed);
  const W = [];
  for (let b = 0; b < L; b++) W.push(seededRandn(seed + 101 * (b + 1), [D, D], { std: 1 / Math.sqrt(D) }).data);
  const x0 = seededRandn(seed + 5, [D], { std: 1 });
  return forward({ n, L, D, A, Bw, M, W, x0 }, false);
}
const narrow = run('learned', 2, 4, 4, 7);
const depth = depthLabel(narrow.blocks.map((b) => b.c));
const s1 = depth.shown.reduce((a, v) => a + Number(v), 0);
const s2 = depth.shown.reduce((a, v) => a + Number(v) * Number(v), 0);
const fromShown = (s1 * s1 / s2).toFixed(2);
const fromTrue = narrow.eff.toFixed(2);
ok(depth.eff === fromShown, 'printed depth divides the printed contributions (' + depth.shown.join(' ') + ' → ' + depth.eff + ')');
ok(fromShown !== fromTrue && depth.eff !== fromTrue, 'rejects effective depth ' + fromTrue + ' beside those contributions');
const home = run('learned', 4, 4, 8, 7);
let shareSplit = null;
home.blocks.forEach((b, i) => {
  const share = contribLabel(rmsOf(b.add), rmsOf(home.readouts[i + 1]));
  const quot = (Number(share.w) / (Number(share.r) + 1e-9)).toFixed(4);
  const old = b.c.toFixed(4);
  ok(share.c === quot, 'printed contribution divides the printed magnitudes (' + share.w + ' / ' + share.r + ' = ' + share.c + ')');
  if (share.c !== old && !shareSplit) shareSplit = { share, old };
});
ok(shareSplit && !('c = ' + shareSplit.old).includes(shareSplit.share.c) && shareSplit.share.c !== shareSplit.old, 'rejects contribution c = ' + (shareSplit ? shareSplit.old : '?') + ' when the printed magnitudes give ' + (shareSplit ? shareSplit.share.c : '?'));
const pageSrc = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(pageSrc.includes('depthLabel(') && pageSrc.includes('contribLabel('), 'the page calls the printed depth and contribution labels');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS mhc');
