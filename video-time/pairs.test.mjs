import { readFileSync } from 'node:fs';
import { attentionPairs, fmtPct, latentAxes, parseN, shareOf } from './pairs.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const c = attentionPairs({ N: 100, S: 10, Tt: 10, win: 2 });
ok(c.full === 100 * 100, 'full pairs are N squared');
ok(c.factorised === 100 * (10 + 10), 'factorised pairs are N times (S+T)');
ok(c.windowed === 100 * c.k, 'windowed pairs are N times k');
ok(c.windowed < c.full, 'a window attends fewer pairs than dense attention (' + c.windowed + ' < ' + c.full + ')');
ok(c.k === Math.min(10, 4) * Math.min(10, 2), 'the window count matches the page formula');
const wide = attentionPairs({ N: 100, S: 10, Tt: 10, win: 100 });
ok(wide.windowed === wide.full, 'a window that covers every token is the dense count');
const axes = latentAxes(320, 184, 16);
ok(axes.latW === 20 && axes.latH === 11, 'a side that is not a multiple of the compression floors (' + axes.latH + ')');
ok(axes.latH !== 184 / 16, 'the unfloored division is not the latent height');

function counts(width, fps, seconds, sw, tc, lp, win) {
  const H0 = Math.round((width * 9 / 16) / 8) * 8;
  const { latW, latH } = latentAxes(width, H0, sw);
  const S = Math.max(1, Math.floor(latW / lp)) * Math.max(1, Math.floor(latH / lp));
  const F = Math.max(1, Math.round(fps * seconds));
  const Tt = 1 + Math.floor((F - 1) / tc);
  const N = S * Tt;
  return { N, pairs: attentionPairs({ N, S, Tt, win }) };
}
const TEXT = 128000;
function quotPct(part, whole) {
  const got = shareOf(part, whole);
  const fromPrinted = fmtPct(whole > 0 ? 100 * parseN(got.a) / parseN(got.b) : 0);
  const fromTrue = fmtPct(whole > 0 ? 100 * part / whole : 0);
  return { got, fromPrinted, fromTrue };
}
const tokens = quotPct(counts(192, 4, 0.5, 4, 2, 1, 6).N, TEXT);
ok(tokens.got.pct === tokens.fromPrinted && tokens.got.text === tokens.got.a + ' vs ' + tokens.got.b + ' = ' + tokens.got.pct, 'the token percent divides the printed counts (' + tokens.got.text + ')');
ok(tokens.fromPrinted !== tokens.fromTrue && tokens.got.text !== tokens.got.a + ' vs ' + tokens.got.b + ' = ' + tokens.fromTrue, 'rejects ' + tokens.got.a + ' vs ' + tokens.got.b + ' = ' + tokens.fromTrue);
ok(tokens.got.text !== '1.34 k vs 128 k = 1.1%', 'rejects 1.34 k vs 128 k = 1.1%');
const win = counts(192, 4, 3.5, 4, 1, 1, 14);
const attn = quotPct(win.pairs.windowed, TEXT * TEXT);
ok(attn.got.pct === attn.fromPrinted && attn.got.text === attn.got.a + ' vs ' + attn.got.b + ' = ' + attn.got.pct, 'the attention percent divides the printed counts (' + attn.got.text + ')');
ok(attn.fromPrinted !== attn.fromTrue && attn.got.text !== '51.6 M vs 16.4 G = 0.32%', 'rejects 51.6 M vs 16.4 G = 0.32%');
const pageSrc = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(pageSrc.includes('shareOf('), 'the page calls shareOf');
ok(!pageSrc.includes('pctOf('), 'the page does not round the unrounded ratio on its own');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS pairs');
