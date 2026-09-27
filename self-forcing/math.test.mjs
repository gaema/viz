import { frameErrors } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const row = frameErrors(6, 0.04);
ok(row.step === '0.040', 'the step is printed at three decimals');
row.teacher.forEach((v, i) => {
  ok(v === (Number(row.step) * (i + 1)).toFixed(3), 'teacher frame ' + (i + 1) + ' is the frame number times the printed step');
});
row.self.forEach((v) => ok(v === row.step, 'every self-forced frame stays at the printed step'));
ok(Number(row.teacher[5]) > Number(row.self[5]), 'by the last frame teacher forcing has drifted');
ok(row.teacher[0] === row.self[0], 'frame 1 is the same under both');

const flat = frameErrors(6, 0);
ok(flat.teacher.every((v, i) => v === flat.self[i]), 'a zero step leaves both trains flat');

const one = frameErrors(1, 0.04);
ok(one.teacher[0] === one.self[0], 'a single frame has not had time to drift');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS self-forcing');
