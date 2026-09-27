import { skipped, drift, scheduleShare } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const residuals = [0.1, 0.4, 0.2, 0.8];
ok(skipped(residuals, 0.15).length < skipped(residuals, 0.5).length, 'a higher threshold skips more steps');
ok(drift(residuals, 0.5) > drift(residuals, 0.15), 'the drift rises with the threshold');
ok(scheduleShare(1, 4) > scheduleShare(1, 40), 'one skipped step is a larger share of a few-step schedule');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS teacache');
