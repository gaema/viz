import { qatForward, qatSentence, steStep, steSentence, steNextSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identityForward(row) {
  ok(row.forward === (row.code * Number(row.scale)).toFixed(2), 'forward is code times the printed scale at ' + row.weight);
  ok(qatSentence(row) === `${row.code} * ${row.scale} = ${row.forward}`, 'forward sentence prints that product');
}

function identityStep(row) {
  ok(row.step === (Number(row.grad) * Number(row.lr)).toFixed(2), 'step is grad times the printed lr');
  ok(row.next === (Number(row.weight) - Number(row.step)).toFixed(2), 'next weight subtracts the printed step');
  ok(steSentence(row) === `${row.grad} * ${row.lr} = ${row.step}`, 'step sentence');
  ok(steNextSentence(row) === `${row.weight} - ${row.step} = ${row.next}`, 'next sentence');
}

const mid = qatForward(1.2, 0.5, 8);
identityForward(mid);
ok(mid.code === 2 && mid.forward === '1.00', '1.20 at scale 0.50 rounds to code 2 and dequantizes to 1.00');

const moved = steStep(1.2, 1, 0.05);
identityStep(moved);
ok(moved.step === '0.05' && moved.next === '1.15', 'the high-precision weight takes the full step');
const still = qatForward(moved.next, 0.5, 8);
identityForward(still);
ok(still.code === mid.code, 'a flat bin keeps the forward code');
ok(still.weight !== mid.weight, 'the stored weight moved while the code stayed');

const boundary = qatForward(1.25, 0.5, 8);
identityForward(boundary);
ok(boundary.code === 3, '1.25 is the next rounding boundary');

for (let bits = 2; bits <= 8; bits++) identityForward(qatForward(1.2, 0.5, bits));
for (let s = 1; s <= 10; s++) identityForward(qatForward(1.2, s / 10, 8));
for (let g = 0; g <= 4; g++) identityStep(steStep(1.2, g / 2, 0.05));
identityForward(qatForward(1.2, 0, 8));
identityStep(steStep(1.2, 0, 0.05));

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS quantization-aware-training');
