import { readFileSync } from 'node:fs';
import { softmax, seededRandn } from '../framework/tensor.js';
import { auxMix, biasRoute, biasRun, biasBarLoad, shownDrops, moveSentence, compareCaptions, lossReadout, loadShare } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const flipped = biasRoute([2, 0], [0, 3], 1, 0.5, 10);
ok(flipped.picked[0] === 1, 'bias flips top-1 onto the lower raw score');
ok(flipped.combine.length === 1 && Math.abs(flipped.combine[0].w - 1) < 1e-12, 'a single pick combines at 1');
ok(flipped.lam === 0, 'bias mode reports auxiliary weight 0');
const down = flipped.moves.find((m) => m.e === 1);
const up = flipped.moves.find((m) => m.e === 0);
ok(down && down.dir === 'down', 'the overloaded expert bias decreases');
ok(up && up.dir === 'up', 'the underloaded expert bias increases');
ok(down.after === (Number(down.before) - Number(down.step)).toFixed(2), 'down step identity');
ok(up.after === (Number(up.before) + Number(up.step)).toFixed(2), 'up step identity');
ok(moveSentence(down) === `${down.before} - ${down.step} = ${down.after}`, 'down sentence');
ok(moveSentence(up) === `${up.before} + ${up.step} = ${up.after}`, 'up sentence');

const both = biasRoute([2, 0], [0, 5], 2, 0.25, 4);
const z = 1 + Math.exp(-2);
ok(Math.abs(both.combine.find((c) => c.i === 0).w - 1 / z) < 1e-9, 'combine uses the raw score');
ok(Math.abs(both.combine.find((c) => c.i === 1).w - Math.exp(-2) / z) < 1e-9, 'the biased score is not the combine weight');

const skew = [0.7, 0.2, 0.1];
const loss0 = auxMix(skew, 0, null);
const loss1 = auxMix(skew, 1, null);
ok(loss0.lam === 0 && Math.abs(loss0.eff[0] / loss0.eff[1] - 0.7 / 0.2) < 1e-4, 'λ=0 keeps the skewed route');
ok(loss1.lam === 1 && Math.abs(loss1.eff[0] - 1 / 3) < 1e-5 && Math.abs(loss1.eff[2] - 1 / 3) < 1e-5, 'λ=1 is uniform');
ok(compareCaptions('loss').a.indexOf('λ=0') === 0, 'loss captions name λ');
ok(!compareCaptions('loss').a.includes('collapse') && !compareCaptions('loss').b.includes('uniform'), 'loss captions do not call the bars collapsed or uniform');
const readme = readFileSync(new URL('./README.md', import.meta.url), 'utf8');
ok(!/no expert starves/i.test(readme), 'readme does not say a high λ starves nobody');
ok(readme.includes('a drag can starve an expert'), 'readme says a drag can starve an expert');
ok(readme.includes('does not force that on every seed'), 'readme does not make λ = 0 a universal collapse');
ok(compareCaptions('bias').a.indexOf('bias route') === 0, 'bias captions do not claim a λ collapse');

function buildSkew(seed, E) {
  const aff = seededRandn(seed | 0, E, { std: 1.5 });
  return softmax(Float32Array.from(aff, (x) => x * 1.6));
}
let evenAtFull = 0;
let openAtZero = 0;
for (let E = 3; E <= 8; E++) {
  for (let seed = 0; seed <= 99; seed++) {
    const full = auxMix(Array.from(buildSkew(seed, E)), 1, null);
    if (Array.from(full.eff).filter((x) => x < 0.4 / E).length === 0) evenAtFull++;
    const raw = auxMix(Array.from(buildSkew(seed, E)), 0, null);
    if (Array.from(raw.eff).filter((x) => x < 0.4 / E).length === 0) openAtZero++;
  }
}
ok(evenAtFull === 600, 'λ=1 with the drag at 1 starves nobody on the seed grid');
ok(openAtZero > 0, 'λ=0 leaves at least one seed with no starved expert');
const wide = auxMix(Array.from(buildSkew(39, 8)), 0.75, null);
const wideLoad = Array.from(wide.eff, (x) => Math.round(x * 120));
const wideStarved = Array.from(wide.eff).filter((x) => x < 0.4 / 8).length;
const wideReport = lossReadout({ lam: 0.75, load: wideLoad, starved: wideStarved, tokens: 120, experts: 8 });
ok(wideReport.near === false, 'λ=0.75 on eight experts is not near uniform');
ok(!wideReport.sentence.includes('no starvation') && !wideReport.sentence.includes('near uniform'), 'a high λ does not claim an even load');
ok(wideReport.sentence.includes(`${wideStarved} expert`), 'the readout counts the starved experts');
ok(wideReport.sentence.includes('most of the mix is the even target'), 'a high λ still names the mix weight');
const drag = new Float32Array(8).fill(1);
drag[0] = 0.04;
const dragged = auxMix(Array.from(buildSkew(5, 8)), 1, drag);
const dragStarved = Array.from(dragged.eff).filter((x) => x < 0.4 / 8).length;
const dragReport = lossReadout({ lam: 1, load: Array.from(dragged.eff, (x) => Math.round(x * 120)), starved: dragStarved, tokens: 120, experts: 8 });
ok(dragStarved > 0, 'a drag at λ=1 can starve an expert');
ok(dragReport.sentence.includes(`${dragStarved} expert`) && !dragReport.sentence.includes('no starvation'), 'the starved drag is counted');
const even = lossReadout({ lam: 1, load: [20, 20, 20, 20, 20, 20], starved: 0, tokens: 120, experts: 6 });
ok(even.near && even.sentence.includes('near uniform (20/expert)') && even.sentence.includes('0 experts starve'), 'an even load is near uniform and starves nobody');
const share = loadShare(47, 120);
ok(share.pct === (47 / 120 * 100).toFixed(1) && share.pct === '39.2', 'the hover percent is the printed count over the batch');
ok(share.sentence === '47 / 120 tokens (39.2%)', 'the hover prints the batch beside the count');

const def = biasRun([1, 0.2, -1, 0.5, -0.4, 0.1], 2, 0.25, 120, 1);
const bars = biasBarLoad(def);
ok(def.picked.length === 2, 'default k keeps two experts');
for (const i of def.picked) {
  ok(def.load[i] === 120, 'default route load is the token count');
  ok(bars[i] === def.load[i], 'the bar is that route load');
}
ok(!def.picked.some((i) => bars[i] === 60), 'default bars are not the renormalized half');
const cap = Math.ceil(1.3 * 120 / bars.length);
ok(shownDrops('loss', bars, cap) > 0, 'those selection counts sit above a loss-mode capacity');
ok(shownDrops('bias', bars, cap) === 0, 'bias mode does not report selection counts as drops');

const page = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(page.includes('auxMix('), 'page calls auxMix');
ok(page.includes('biasRun('), 'page calls biasRun');
ok(page.includes('biasBarLoad('), 'page draws the route load');
ok(!page.includes('biasRow.load.reduce'), 'page does not divide the route load by its sum');
ok(page.includes("key: 'lam'"), 'compare key stays λ');
ok(page.includes("compareCaptions('loss').a") && page.includes("compareCaptions('loss').b"), 'compare panes use the shipped captions');
ok(!page.includes('router collapse') && !page.includes('balanced (uniform)'), 'page does not install the old compare captions');
ok(page.includes('the bias step sets these bars'), 'the label still names the token load');
ok(page.includes('lossReadout('), 'page uses the shipped load sentence');
ok(page.includes('loadShare('), 'page uses the shipped hover share');
ok(!page.includes('no starvation'), 'page does not claim a high λ has no starvation');
ok(page.includes('shownDrops('), 'page uses the shipped drop count');
ok(page.includes("biasRow ? 'bias' : 'loss'"), 'bias mode passes the bias balance into the drop count');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS moe-balance');
