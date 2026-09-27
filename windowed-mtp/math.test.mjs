import { acceptLabel, draftSpan, readSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const span = draftSpan(24, 2, 4);
ok(span.visible === span.sink + span.window - span.overlap, 'visible tokens are sink plus window minus the overlap');
ok(span.dropped === span.length - span.visible, 'dropped tokens are the rest of the length');
ok(span.visible === 6 && span.dropped === 18, 'a length of 24 with sink 2 and window 4 keeps 6');

const mid = acceptLabel(12, 24, 2, 4, 0.25);
ok(mid.seen === false && mid.accept === '0.25', 'a fact in the gap is accepted at the printed blind rate');
ok(mid.accept === mid.blind, 'the gap acceptance is the printed blind rate');

const head = acceptLabel(0, 24, 2, 4, 0.25);
const tail = acceptLabel(22, 24, 2, 4, 0.25);
ok(head.seen && head.accept === '1.00', 'a fact in the sink is accepted outright');
ok(tail.seen && tail.accept === '1.00', 'a fact in the window is accepted outright');

const covered = draftSpan(8, 4, 6);
ok(covered.overlap === 2 && covered.visible === 8 && covered.dropped === 0, 'sink and window that overlap cover the whole context once');
ok(acceptLabel(3, 8, 4, 6, 0.1).accept === '1.00', 'nothing is blind once the draft can see every token');

const over = acceptLabel(3, 8, 4, 6, 0.25);
ok(over.overlap === 2, 'four sink tokens and six window tokens on a length of 8 share two tokens');
ok(over.visible === over.sink + over.window - over.overlap, 'the visible count subtracts the printed overlap');
ok(readSentence(over) === 'length 8, sink 4, window 6, overlap 2: draft reads 8 of 8 (4 + 6 - 2)', 'the sentence is the printed subtraction');
ok(readSentence(span) === 'length 24, sink 2, window 4, overlap 0: draft reads 6 of 24 (2 + 4 - 0)', 'a gapless span still prints the subtraction');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS windowed-mtp');
