import { orderDtypes } from './order.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const zoo = [
  { key: 'fp32', bits: 32 },
  { key: 'tf32', bits: 19 },
  { key: 'fp16', bits: 16 },
  { key: 'bf16', bits: 16 },
  { key: 'fp8', bits: 8 },
  { key: 'mxfp4', bits: 4.25 },
  { key: 'int8', bits: 8 },
  { key: 'uint8', bits: 8 },
  { key: 'int4', bits: 4 },
];
const keys = (mode) => orderDtypes(zoo, mode).map((d) => d.key).join(' ');

ok(keys('family') === 'fp32 tf32 fp16 bf16 fp8 mxfp4 int8 uint8 int4', 'family order is the catalogue order (' + keys('family') + ')');
ok(keys('bits') === 'fp32 tf32 fp16 bf16 fp8 int8 uint8 mxfp4 int4', 'bits order is widest first, ties keep catalogue order (' + keys('bits') + ')');
ok(keys('bits') !== keys('family'), 'bits order is not the family order');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS order');
