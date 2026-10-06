import { readFileSync } from 'node:fs';
import { nestedRead, nestedSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(row) {
  ok(row.gap === (Number(row.pTail) - Number(row.fTail)).toFixed(3), 'gap identity at cut ' + row.cut);
  ok(nestedSentence(row) === `${row.pTail} - ${row.fTail} = ${row.gap}`, 'nested sentence');
  ok(Number(row.pTail) === 1, 'the prefix ties the tail-only pair');
  ok(Number(row.fTail) < 1, 'the full vector separates the tail-only pair');
  ok(Number(row.pCoarse) < 1, 'the prefix separates a pair that differs inside the cut');
}

identity(nestedRead(4, 2));
for (let cut = 1; cut <= 7; cut++) identity(nestedRead(8, cut));

const page = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(page.includes('nestedRead('), 'page calls nestedRead');
ok(page.includes('nestedSentence('), 'page calls nestedSentence');
ok(!page.includes('unmodified'), 'page does not call the nested vectors unmodified');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS real-embeddings');
