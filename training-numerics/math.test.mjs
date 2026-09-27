import { scaledFate, masterBytes, fp8Bytes, FP8_E4M3_MAX, FP8_E4M3_MIN_NORMAL } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const tiny = FP8_E4M3_MIN_NORMAL / 100;
ok(scaledFate(tiny, 1) === 'flush', 'scale 1 flushes a gradient under the fp8 floor');
ok(scaledFate(tiny, 1000) === 'kept', 'a larger scale lifts the same gradient into range');
ok(scaledFate(1, FP8_E4M3_MAX * 4) === 'overflow', 'a scale past the fp8 max overflows');
ok(masterBytes(1000) === 4 * fp8Bytes(1000), 'fp32 master weights cost four bytes per parameter');
ok(masterBytes(1000) > fp8Bytes(1000), 'the master copy is the memory price');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS training-numerics');
