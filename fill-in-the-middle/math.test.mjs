import { fimLabel, fimSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const open = fimLabel(3, 2, 2, false);
ok(open.one === 7, 'one example is prefix plus suffix plus middle');
ok(open.suffixSeen === open.suffix, 'the first middle token sees every suffix token');
ok(open.cross === open.one * open.one, 'an open mask lets the second example see the square of one example');
ok(fimSentence(open) === 'prefix 3, suffix 2, middle 2. the first middle token sees 2 of 2 suffix tokens. cross pairs 49 = 7 × 7.', 'the open sentence is the printed product');

const shut = fimLabel(3, 2, 2, true);
ok(shut.cross === 0, 'isolating the examples removes the cross pairs');
ok(shut.suffixSeen === 2, 'isolation does not hide an example from its own suffix');
ok(fimSentence(shut).endsWith('cross pairs 0.'), 'the blocked sentence prints 0 and not a false product');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS fill-in-the-middle');
