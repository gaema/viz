import { readFileSync } from 'node:fs';
import { ropeAngles, seededRandn } from '../framework/tensor.js';
import { angleEquation, applyPosition, rotateLabel, roundShown, spinRatio } from './position.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const v = [0.4, -1.2, 0.7, 0.2];
const plain = applyPosition(v, 5, 10000, true);
ok(plain.length === v.length && plain.every((x, i) => x === v[i]), 'no-position mode returns the input vector');
const spun = applyPosition(v, 5, 10000, false);
ok(spun.some((x, i) => x !== v[i]), 'the same vector at position 5 is rotated when position is on');

const pos = 7.5, base = 10000, dim = 64, pair = 1;
const theta = Math.pow(base, (-2 * pair) / dim);
const delta = ropeAngles(dim, pos, { theta: base })[pair];
ok(Math.abs(delta - pos * theta) < 1e-5, 'the angle is p times θ');
const rounded = Number(roundShown(pos)) * Number(roundShown(theta));
ok(roundShown(rounded) !== roundShown(delta), 'rounded p times rounded θ prints ' + roundShown(rounded) + ', not the angle ' + roundShown(delta));
const line = angleEquation(delta);
ok(line === `Δ = p·θᵢ = ${roundShown(delta)} rad`, 'the equation states the angle');
ok(!line.includes(roundShown(pos) + '·' + roundShown(theta)), 'the equation does not multiply the rounded factors');

const vec = seededRandn(4, 8);
const turned = rotateLabel(vec[0], vec[1], 0);
const trueRx = (vec[0] * Math.cos(0) - vec[1] * Math.sin(0)).toFixed(3);
const trueRy = (vec[0] * Math.sin(0) + vec[1] * Math.cos(0)).toFixed(3);
const fromPrintedX = (Number(turned.a) * Number(turned.c) - Number(turned.b) * Number(turned.s)).toFixed(3);
const fromPrintedY = (Number(turned.a) * Number(turned.s) + Number(turned.b) * Number(turned.c)).toFixed(3);
ok(turned.rx === fromPrintedX && turned.ry === fromPrintedY, 'the rotated pair is the printed rotation (' + turned.a + ', ' + turned.b + ') → (' + turned.rx + ', ' + turned.ry + ')');
ok(turned.rx !== trueRx && turned.ry !== trueRy, 'rejects (' + turned.a + ', ' + turned.b + ') → (' + trueRx + ', ' + trueRy + ')');
const pageSrc = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(pageSrc.includes('rotateLabel('), 'the hover calls rotateLabel');

function shownSpin(dim, pos, base) {
  const ang = ropeAngles(dim, pos, { theta: base });
  const spin = spinRatio(ang[0], ang[ang.length - 1], ang.length - 1);
  const fromPrinted = Number(spin.slow) === 0 ? spin.ratio : roundShown(Number(spin.fast) / Number(spin.slow));
  const fromTrue = roundShown(ang[0] / ang[ang.length - 1]);
  return { spin, fromPrinted, fromTrue };
}
const wide = shownSpin(4, 4, 1000);
ok(wide.spin.ratio === wide.fromPrinted && wide.spin.text.includes('Δ=' + wide.spin.fast) && wide.spin.text.includes('Δ=' + wide.spin.slow), 'the spin ratio divides the printed angles (' + wide.spin.text + ')');
ok(wide.fromPrinted !== wide.fromTrue && !wide.spin.text.includes(wide.fromTrue + '×'), 'rejects Δ=' + wide.spin.fast + ' and Δ=' + wide.spin.slow + ' = ' + wide.fromTrue + '×');
const slow = shownSpin(4, 0.2, 1000);
ok(slow.fromPrinted !== slow.fromTrue && !slow.spin.text.includes('31.6×'), 'rejects Δ=' + slow.spin.fast + ' and Δ=' + slow.spin.slow + ' = 31.6× (' + slow.spin.text + ')');
const fine = shownSpin(8, 1.234, 10000);
ok(fine.fromPrinted !== fine.fromTrue && !fine.spin.text.includes(fine.fromTrue + '×'), 'rejects Δ=' + fine.spin.fast + ' and Δ=' + fine.spin.slow + ' = ' + fine.fromTrue + '× (' + fine.spin.text + ')');
ok(pageSrc.includes('spinRatio('), 'the readout calls spinRatio');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS position');
