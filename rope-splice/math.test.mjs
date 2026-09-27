import { ROPE_BASE, ROPE_DIM, moveErrorText, scoreFromPrinted, spliceError } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const moved = spliceError(1, 0, 0, 16, 0, ROPE_DIM, ROPE_BASE);
ok(moved.err < 1e-9, 'rotating a key by the position gap matches a key computed at the destination');
ok(Math.abs((moved.aSrc + moved.gap) - moved.aDest) < 1e-12, 'the gap angle is the destination angle minus the source angle');
ok(moveErrorText(moved.err) === (moved.err === 0 ? '0' : moved.err.toExponential(1)), 'the printed move error is the function value');

const raw = scoreFromPrinted(16, 0, 0, ROPE_DIM, ROPE_BASE);
const native = scoreFromPrinted(16, 16, 0, ROPE_DIM, ROPE_BASE);
ok(raw.score === Math.cos(Number(raw.diff)).toFixed(3), 'the unrotated score is the cosine of the printed angle difference');
ok(native.score === Math.cos(Number(native.diff)).toFixed(3), 'the destination score is the cosine of the printed angle difference');
ok(native.diff === '0.0000' && native.score === '1.000', 'a query and a key at the same position score 1');
ok(raw.score !== native.score, 'leaving the key at the source position changes the score');

const slow = scoreFromPrinted(16, 0, 3, ROPE_DIM, ROPE_BASE);
ok(Math.abs(Number(slow.diff)) < Math.abs(Number(raw.diff)), 'a later pair turns less for the same position gap');

const stayed = spliceError(0.3, -0.7, 5, 5, 1, ROPE_DIM, ROPE_BASE);
ok(stayed.err < 1e-9, 'a chunk that does not move is already at the destination');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS rope-splice');
