import { ternaryBill } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(row) {
  row.recon.forEach((r, i) => {
    ok(r === (row.codes[i] * Number(row.scale)).toFixed(2), 'recon ' + i + ' is the code times the printed scale');
  });
  const full = row.weights.map(Number).reduce((a, b) => a + b, 0);
  const tern = row.recon.map(Number).reduce((a, b) => a + b, 0);
  ok(row.full === full.toFixed(2), 'the full dot is the sum of the printed weights');
  ok(row.tern === tern.toFixed(2), 'the ternary dot is the sum of the printed reconstructions');
  let abs = 0;
  for (let i = 0; i < row.weights.length; i++) abs += Math.abs(Number(row.weights[i]) - Number(row.recon[i]));
  ok(row.abs === abs.toFixed(2), 'the absolute error is the sum of the printed gaps');
}

const mid = ternaryBill(0.5);
identity(mid);
ok(mid.codes.join(',') === '1,0,-1,0,-1', 'scale 0.50 zeros the two small weights and clips the large ones to a sign');
ok(mid.nonzero === 3 && mid.count === 5, '3 of 5 weights survive');
ok(mid.full === '0.00' && mid.tern === '-0.50', 'the dots are 0.00 and -0.50');
ok(mid.abs === '1.10', 'the absolute error is 1.10');

const coarse = ternaryBill(1);
identity(coarse);
ok(coarse.codes.join(',') === '1,0,-1,0,0', 'a scale of 1.00 also zeros -0.40');

const fine = ternaryBill(0.1);
identity(fine);
ok(fine.nonzero === 5, 'a scale of 0.10 keeps every weight');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS ternary-weights');
