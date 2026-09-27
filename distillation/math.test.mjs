import { copiesTeacherError, softMass } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const teacher = [4, 0];
const truth = 1;
ok(copiesTeacherError(teacher, truth, 1) === true, 'at T=1 the student target prefers the teacher\'s wrong class');
ok(softMass(teacher, 0, 1) > 0.5, 'the wrong class carries most of the soft target');
ok(softMass(teacher, truth, 1) < softMass(teacher, 0, 1), 'the truth is the smaller target');
ok(softMass(teacher, truth, 8) > softMass(teacher, truth, 1), 'a higher temperature moves mass back toward the truth');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS distillation');
