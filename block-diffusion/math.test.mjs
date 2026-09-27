import { positionAccept, tailAccept } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

ok(positionAccept(0, 0.8) === 0.8, 'the first position is quality^1');
ok(tailAccept(4, 0.8) === 0.8 ** 4, 'the tail of width w is quality^w');
ok(positionAccept(0, 0.8) > positionAccept(3, 0.8), 'later positions inside one block are accepted less often');
ok(tailAccept(8, 0.8) < tailAccept(2, 0.8), 'a wider block has a lower tail acceptance');
ok(tailAccept(4, 0.5) < tailAccept(4, 0.9), 'a weaker draft lowers the same tail');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS block-diffusion');
