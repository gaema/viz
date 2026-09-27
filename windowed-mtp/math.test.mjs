import { acceptLabel, draftSpan } from './math.js';

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

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS windowed-mtp');
