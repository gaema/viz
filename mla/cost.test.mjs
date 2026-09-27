import { readFileSync } from 'node:fs';
import { byteCompare, derive, macPrice } from './cost.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const st = { heads: 128, hdim: 128, dc: 512, dR: 64, hidden: 7168, layers: 61, kvdtype: 'fp16', ctxk: 8 };
const v = derive(st);
const rope = v.d * v.heads * v.dR;
const without = 2 * v.d * v.heads * v.dc + v.d * (v.dc + v.dR);
ok(v.macMLA === without + rope, 'projection cost includes the per-head RoPE query');
ok(v.macPct > 201, 'the default ratio is higher than the old 201% figure (' + v.macPct.toFixed(1) + ')');
ok(Math.abs(v.cachePct - 100 * (512 + 64) / (2 * 128 * 128)) < 1e-9, 'cache ratio is unchanged');

const wide = derive({ heads: 4, hdim: 16, dc: 96, dR: 224, hidden: 512, layers: 1, kvdtype: 'fp16', ctxk: 1 });
const price = macPrice(wide.macMHA, wide.macMLA);
const parts = price.text.match(/^(.*) → (.*) = (\d+)% of MHA$/);
const quot = Math.round(100 * Number(parts[2].replace(/,/g, '')) / Number(parts[1].replace(/,/g, '')));
ok(String(quot) === parts[3], 'the printed percent is the quotient of the printed MAC counts (' + price.text + ')');
const oldMla = (wide.macMLA / 1e6).toFixed(1) + 'M';
const oldPct = wide.macPct.toFixed(0);
ok(Math.round(100 * (Number(oldMla.slice(0, -1)) * 1e6) / wide.macMHA) !== Number(oldPct), '1.0M is not the true percent of the MHA count');
ok(!price.text.includes(oldMla + ' = ' + oldPct + '%'), 'the sentence does not pair 1.0M with the true percent');

function parseBytes(s) {
  const t = String(s).trim();
  if (t.endsWith('GB')) return Number(t.slice(0, -2)) * 1073741824;
  if (t.endsWith('MB')) return Number(t.slice(0, -2)) * 1048576;
  if (t.endsWith('KB')) return Number(t.slice(0, -2)) * 1024;
  if (t.endsWith('B')) return Number(t.slice(0, -1));
  return Number(t);
}
const narrow = derive({ heads: 4, hdim: 16, dc: 32, dR: 0, hidden: 64, layers: 15, kvdtype: 'fp8', ctxk: 1 });
const bytes = byteCompare(narrow.mlaTok, narrow.mhaTok);
const byteQuot = (100 * parseBytes(bytes.mla) / parseBytes(bytes.mha)).toFixed(2);
const byteTrue = (100 * narrow.mlaTok / narrow.mhaTok).toFixed(2);
ok(bytes.pct === byteQuot && bytes.text === bytes.mla + ' vs ' + bytes.mha + ' /token — ' + bytes.pct + '% of MHA', 'the printed byte percent divides the printed byte strings (' + bytes.text + ')');
ok(byteQuot !== byteTrue && !bytes.text.includes(byteTrue + '%'), 'rejects ' + bytes.mla + ' vs ' + bytes.mha + ' /token — ' + byteTrue + '% of MHA');
const pageSrc = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(pageSrc.includes('byteCompare('), 'the cache bar calls byteCompare');
ok(!pageSrc.includes('cachePct.toFixed(2)}% of MHA (lower=better)'), 'the cache bar does not append the true cache percent');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS mla');
