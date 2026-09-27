import { hitPct, poolHits } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function pctIdentity(hits, requests) {
  const label = hitPct(hits, requests);
  const expect = Number(label.requests) === 0 ? '0.0' : ((100 * Number(label.hits)) / Number(label.requests)).toFixed(1);
  ok(label.pct === expect, `percent ${label.pct} is 100 * ${label.hits} / ${label.requests}`);
}

const spread = poolHits(4, 4, 2);
ok(spread.requests === 8, '4 prefixes, twice, is 8 requests');
ok(spread.privateHits === 0, 'the second wave lands on a different machine, so every private cache misses');
ok(spread.sharedHits === 4, 'the shared pool hits the whole second wave');
ok(spread.sharedHits > spread.privateHits, 'across machines the shared pool hits more often');
pctIdentity(spread.privateHits, spread.requests);
pctIdentity(spread.sharedHits, spread.requests);
ok(hitPct(spread.sharedHits, spread.requests).pct === '50.0', '4 of 8 is 50.0%');

const one = poolHits(4, 1, 2);
ok(one.privateHits === 4 && one.sharedHits === 4, 'one machine makes the private cache and the shared pool the same');

const once = poolHits(4, 4, 1);
ok(once.privateHits === 0 && once.sharedHits === 0, 'a prefix seen once is a miss for both');
pctIdentity(0, once.requests);

const wrap = poolHits(4, 2, 3);
ok(wrap.sharedHits === 8, 'every arrival after the first wave is in the pool');
ok(wrap.privateHits === 4, 'only the wave that returns to the same machine is a private hit');
ok(wrap.sharedHits > wrap.privateHits, 'the shared pool still leads after a wave comes back home');
pctIdentity(1, 3);

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS kv-fabric');
