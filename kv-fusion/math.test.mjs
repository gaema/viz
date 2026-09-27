import { fusionBill, fusionCompare } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(b) {
  ok(Number(b.prefixText) === b.header + (b.header === 0 ? 0 : b.tokens), 'prefix total is header plus the whole chunk, or header alone when the chunk is the prefix');
  ok(Number(b.fusionText) === b.header + Number(b.frac) * b.tokens, 'fusion total is header plus fraction times chunk, from the printed strings');
  ok(b.error === ((1 - Number(b.frac)) * Number(b.mismatch)).toFixed(2), 'residual error is the unrecomputed fraction times the printed mismatch');
}

const mid = fusionBill(8, 4, 0.25, 1);
identity(mid);
ok(mid.prefixText === '12', 'a non-prefix chunk of 8 behind a header of 4 costs the prefix cache 12');
ok(mid.fusionText === '6', 'recomputing a quarter of that chunk costs fusion 6');
ok(mid.error === '0.75', 'leaving three quarters unrecomputed at full mismatch leaves error 0.75');
ok(Number(mid.fusionText) < Number(mid.prefixText), 'fusion pays fewer tokens than the prefix cache when the chunk is not a prefix');

const exact = fusionBill(8, 0, 0.25, 1);
identity(exact);
ok(exact.prefixText === '0', 'a chunk at the start is free for the prefix cache');
ok(exact.fusionText === '2', 'fusion still recomputes its fraction of a true prefix');
ok(Number(exact.fusionText) > Number(exact.prefixText), 'on a true prefix the prefix cache pays less than fusion');

const full = fusionBill(8, 4, 1, 0.8);
identity(full);
ok(full.fusionText === full.prefixText, 'recomputing the whole chunk ties the prefix cache');
ok(full.error === '0.00', 'a full recompute leaves no residual');

const clean = fusionBill(8, 4, 0, 0);
identity(clean);
ok(clean.fusionText === '4', 'a zero fraction recomputes only the new header');
ok(clean.error === '0.00', 'a chunk that still matches has no residual even when nothing is recomputed');

ok(fusionCompare(mid) === 'Fusion pays fewer tokens than the prefix cache.', 'a partial recompute of a non-prefix chunk pays fewer tokens');
ok(fusionCompare(exact) === 'The prefix cache pays fewer tokens than fusion.', 'a true prefix is cheaper for the prefix cache when fusion still recomputes');
ok(fusionCompare(full) === 'Fusion and the prefix cache pay the same number of tokens.', 'a full recompute ties');
ok(fusionCompare(fusionBill(8, 0, 0, 1)) === 'Fusion and the prefix cache pay the same number of tokens.', 'a zero fraction on a true prefix also ties');

const half = fusionBill(7, 0, 0.5, 0.5);
identity(half);
ok(half.fusionText === '3.50', 'a half recompute of 7 tokens prints 3.50, the product of the printed factors');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS kv-fusion');
