import { readFileSync } from 'node:fs';
import { build, stages, kvpTradeClause, kvpClosing, cardBlurb } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const base = {
  gpus: 4, d: 4096, layers: 48, experts: 64, topk: 6, reqs: 48, seq: 8192,
  skew: 0.4, micro: 1, link: 400, budget: 80, overhead: 3, kv: 'gqa', moe: 'ep',
};

const tp = build({ ...base, attn: 'tp' });
const pp = build({ ...base, attn: 'pp' });
const dp = build({ ...base, attn: 'dp' });
const kvp = build({ ...base, attn: 'kvp' });
ok(tp.attn === 'tp' && pp.attn === 'pp' && dp.attn === 'dp' && kvp.attn === 'kvp', 'old and new attention modes still run');
ok(dp.attnWire === 0, 'data-parallel attention crosses nothing');
ok(pp.attnWire === 0, 'pipeline attention is not an activation all-reduce');
ok(tp.attnWire !== kvp.attnWire, 'KV-parallel wire differs from tensor-parallel wire');
ok(kvp.moeWire === tp.moeWire, 'MoE wire is unchanged when only attention changes');
ok(kvp.attnWire !== kvp.moeWire, 'KV-parallel attention and expert-parallel FFN differ on the wire');
ok(build({ ...base, attn: 'kvp', moe: 'tp' }).moe === 'tp', 'tensor-parallel FFN still runs beside KV-parallel attention');
ok(build({ ...base, attn: 'tp', moe: 'pp' }).moeWire === 0, 'pipeline MoE still contributes no MoE all-reduce');

const kinds = stages(kvp).map((s) => s.kind);
ok(kinds.includes('kvgather'), 'KV-parallel draws a cache gather');
ok(!kinds.includes('alltoall') || stages(kvp).filter((s) => s.kind === 'alltoall').every((s) => s.part === 'moe'), 'the gather is not the expert token dispatch');
ok(stages(tp).some((s) => s.kind === 'allreduce'), 'tensor-parallel attention still all-reduces');
ok(stages(dp).some((s) => s.label.indexOf('NOTHING crosses') >= 0), 'data-parallel attention still names an empty wire');

const ns = [2, 4, 8];
const seqs = [512, 8192, 65536];
const ds = [1024, 4096, 8192];
const kvs = ['gqa', 'latent'];
const moes = ['ep', 'tp'];
for (const gpus of ns) for (const seq of seqs) for (const d of ds) for (const kv of kvs) for (const moe of moes) {
  const row = build({ ...base, gpus, seq, d, kv, moe, attn: 'kvp' });
  const twin = build({ ...base, gpus, seq, d, kv, moe, attn: 'dp' });
  ok(row.attnWire !== row.moeWire, `kvp wire differs from ${moe} at N=${gpus} S=${seq} d=${d} ${kv}`);
  ok(twin.attnWire === 0, `dp attnWire stays 0 at N=${gpus}`);
  ok(row.moeWire === build({ ...base, gpus, seq, d, kv, moe, attn: 'tp' }).moeWire, `moe wire ignores attn at N=${gpus} ${moe}`);
}

const one = build({ ...base, gpus: 1, attn: 'kvp' });
ok(one.attnWire === 0, 'one GPU KV-parallel gather is zero');
ok(kvpTradeClause(one) === '', 'one GPU trade clause does not claim a cache move');
ok(kvpClosing(one).includes('gathers no cache bytes'), 'one GPU closing says the gather is zero');
ok(!/gathers cache bytes/.test(kvpClosing(one)), 'one GPU closing does not claim a gather');
const many = build({ ...base, gpus: 4, attn: 'kvp' });
ok(many.attnWire > 0, 'several GPUs gather cache bytes');
ok(kvpTradeClause(many).includes('moves cache bytes'), 'trade clause names the move when the gather is non-zero');
ok(/gathers cache bytes/.test(kvpClosing(many)), 'closing names the gather when it is non-zero');
const tpOne = build({ ...base, gpus: 1, attn: 'tp' });
ok(kvpClosing(tpOne).includes('only when'), 'another attention mode does not say the current view gathers');
ok(cardBlurb.includes('With one GPU the gather is zero'), 'blurb stays true on a one-GPU strip');

const page = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(page.includes("from './math.js'"), 'page imports the shipped math');
ok(page.includes('kvpTradeClause(') && page.includes('kvpClosing('), 'page uses the shipped gather sentences');
ok(!page.includes('when that choice is selected, shards the sequence and gathers cache bytes'), 'page does not claim the gather unconditionally');
ok(!page.includes('not drawn here') && !page.includes('not shown here'), 'page does not say the sequence split is absent');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS parallelism');
