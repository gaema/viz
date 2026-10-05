import { kvShare, shareSentence, writesAt } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const half = kvShare(8, 2);
ok(half.writers === 4 && half.readers === 4, 'sharing every second layer writes half the caches');
ok(half.writers + half.readers === half.layers, 'writers and readers add up to the layers');
ok(half.writers === Math.ceil(half.layers / half.share), 'the writer count is the layers divided by the share, rounded up');
ok(shareSentence(half) === 'layers 8, share 2: 4 write, 4 reuse (4 + 4 = 8). cache 4 of 8', 'the sentence is the printed sum');

const all = kvShare(8, 1);
ok(all.writers === 8 && all.readers === 0, 'a share of 1 means every layer writes its own cache');
ok(shareSentence(all).includes('(8 + 0 = 8)'), 'the no-sharing sum is printed');

const wide = kvShare(8, 3);
ok(wide.writers === 3 && wide.readers === 5, 'eight layers shared by threes still write a cache for the tail');
ok(wide.writers + wide.readers === 8, 'the tail writer is counted once');

let painted = 0;
for (let i = 0; i < 8; i++) if (writesAt(i, 2)) painted++;
ok(painted === half.writers, 'the drawn writer rows are the writer count');

function cacheSizes(L, S) {
  const sizes = [];
  let run = 0;
  for (let i = 0; i < L; i++) {
    if (writesAt(i, S)) {
      if (run) sizes.push(run);
      run = 1;
    } else run++;
  }
  if (run) sizes.push(run);
  return sizes;
}

ok(cacheSizes(8, 3).join(',') === '3,3,2', 'share 3 on 8 layers leaves the last cache with 2 layers');
ok(cacheSizes(5, 4).join(',') === '4,1', 'share 4 on 5 layers leaves the last cache with 1 layer');

for (let L = 4; L <= 16; L++) {
  for (let S = 1; S <= 4; S++) {
    const row = kvShare(L, S);
    let n = 0;
    for (let i = 0; i < L; i++) if (writesAt(i, S)) n++;
    ok(n === row.writers && row.writers + row.readers === L, `drawn writers match the sentence at ${L} by ${S}`);
  }
}

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS cross-layer-kv');
