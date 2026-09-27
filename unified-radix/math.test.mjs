import { unifiedReuse } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const warm = unifiedReuse(12, 4, true);
ok(warm.full === 12, 'full-attention KV covers the whole match');
ok(warm.windowed === 4, 'the window keeps only its own length of the match');
ok(warm.windowed === Math.min(warm.match, warm.window), 'window tokens are the min of the printed match and the printed window');
ok(warm.state === 1, 'a non-empty match reuses one recurrent checkpoint');

const short = unifiedReuse(3, 4, true);
ok(short.windowed === 3, 'a match shorter than the window cannot invent tokens outside the match');
ok(short.state === 1, 'a short match still has one checkpoint');

const cold = unifiedReuse(0, 4, true);
ok(cold.full === 0 && cold.windowed === 0 && cold.state === 0, 'an empty match reuses nothing, including the recurrent state');

const off = unifiedReuse(12, 4, false);
ok(off.full === 12 && off.windowed === 4 && off.state === 0, 'with no recurrent layer the checkpoint count is 0');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS unified-radix');
