import { pruneGroup, magnitudeSentence, dropSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(row) {
  ok(row.sum === (Number(row.a) + Number(row.b)).toFixed(2), 'magnitude sum ' + row.sum);
  ok(row.dropped === String(Number(row.nonzero) - Number(row.kept)), 'dropped count');
  ok(magnitudeSentence(row) === `${row.a} + ${row.b} = ${row.sum}`, 'magnitude sentence');
  ok(dropSentence(row) === `${row.nonzero} - ${row.kept} = ${row.dropped}`, 'drop sentence');
  ok(row.out.filter((v) => Number(v) !== 0).length <= 2, 'at most two nonzeros');
  ok(Number(row.kept) === row.out.filter((v) => Number(v) !== 0).length, 'kept counts the printed nonzeros');
}

const dense = pruneGroup([0.9, 0.2, -0.8, 0.1]);
identity(dense);
ok(dense.keep.join(',') === '0,2', 'the two largest magnitudes are 0.90 and 0.80');
ok(dense.out[0] === '0.90' && dense.out[2] === '-0.80', 'signs of the kept entries stay');
ok(dense.out[1] === '0.00' && dense.out[3] === '0.00', 'the smaller entries are rewritten to zero');

const already = pruneGroup([0.5, 0, 0, -0.2]);
identity(already);
ok(Number(already.kept) <= 2 && already.dropped === '0', 'a sparser group drops nothing');

const ties = pruneGroup([0.5, 0.5, 0.5, 0.1]);
identity(ties);
ok(ties.keep.join(',') === '0,1', 'ties keep the earlier index');

const zeros = pruneGroup([0, 0, 0, 0]);
identity(zeros);
ok(zeros.kept === '0' && zeros.dropped === '0', 'an empty group stays empty');

for (let i = 0; i <= 10; i++) identity(pruneGroup([i / 10, 0.2, -0.8, 0.1]));

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS structured-sparsity');
