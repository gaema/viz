import { packMask, crossDocumentPairs } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const lengths = [2, 3, 1];
const good = packMask(lengths, 'block');
const bad = packMask(lengths, 'full');
ok(good.mask.length === 6 && good.mask[0].length === 6, 'three documents pack into one row of 6');
ok(crossDocumentPairs(good.mask, good.doc) === 0, 'the block-diagonal mask has no cross-document pair');
ok(crossDocumentPairs(bad.mask, bad.doc) > 0, 'a mask that is true everywhere attends across documents');
ok(good.mask[0][1] === true && good.mask[0][2] === false, 'token 0 sees its own document and not the next');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS sequence-packing');
