import { shipCompare, shipLabel } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(row) {
  const pd = 2 * Number(row.layers) * Number(row.kv) * Number(row.seq) * Number(row.bytes);
  const afd = 2 * Number(row.layers) * Number(row.batch) * Number(row.hidden) * Number(row.bytes);
  ok(row.pd === String(pd), 'KV bytes are 2 × layers × kv × seq × bytes');
  ok(row.afd === String(afd), 'hidden-state bytes are 2 × layers × batch × hidden × bytes');
}

const long = shipLabel(4, 32, 4);
identity(long);
ok(long.pd === '2048' && long.afd === '512', 'a long context makes the one-time KV ship larger');
ok(shipCompare(long) === 'The one-time KV ship is larger.', 'the sentence follows the two printed sizes');

const short = shipLabel(4, 4, 4);
identity(short);
ok(short.pd === '256' && short.afd === '512', 'a short context makes the per-step ship larger');
ok(shipCompare(short) === 'The per-step hidden-state ship is larger.', 'the sentence flips with the sizes');

const tie = shipLabel(2, 8, 4);
identity(tie);
ok(Number(tie.pd) === Number(tie.afd), 'the two ships can tie');
ok(shipCompare(tie) === 'The two ships are the same size.', 'a tie is said as a tie');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS attn-expert-split');
