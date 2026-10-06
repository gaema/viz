import { readFileSync } from 'node:fs';
import {
  buildTree, longestSuffixMatch, jacobiPropose, verifyPrefix, jacobiBill,
  simulate, shownTpf, cardBlurb,
} from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const windowed = jacobiPropose(4, ['aa', 'bb', 'cc', 'dd', 'ee']);
ok(windowed.proposal.length === 4, 'a Jacobi window proposes the trajectory prefix');
ok(windowed.ngrams.length === 10, 'the window pools every n-gram inside the proposal');
const empty = buildTree(['zz', 'qq'], 4);
ok(longestSuffixMatch(empty, ['aa', 'bb'], 4).len === 0, 'that trajectory is not a suffix-tree hit');

const kept = verifyPrefix(['x', 'y'], ['x', 'z']);
ok(kept.length === 1 && kept[0] === 'x', 'verification keeps only the agreeing prefix');
ok(verifyPrefix([{ tok: 'x' }, { tok: 'z' }], ['x', 'z']).join(',') === 'x,z', 'a full agree is kept');

const bill = jacobiBill(2, ['x', 'y', 'q'], ['x', 'z']);
ok(bill.sentence === `${bill.proposed} - ${bill.rejected} = ${bill.acceptedN}`, 'bill prints its own difference');
ok(Number(bill.proposed) - Number(bill.rejected) === Number(bill.acceptedN), 'bill identity');
ok(bill.acceptedN === '1', 'the mismatched second token is not accepted');

const base = { depth: 4, plen: 3, rounds: 2, seed: 3, corpus: 6, extra: '' };
const suffix = simulate({ ...base, spec: 'suffix' }, 0.4);
const unset = simulate(base, 0.4);
ok(suffix.mean === unset.mean && suffix.emitted === unset.emitted, 'unset spec keeps the suffix-tree path');
ok(suffix.spec === 'suffix' && suffix.rounds.every((rd) => rd.prop.every((p) => !p.jacobi)), 'suffix proposals are tree proposals');

const jacobi = simulate({ ...base, spec: 'jacobi' }, 0);
ok(jacobi.spec === 'jacobi', 'jacobi mode is marked');
ok(jacobi.rounds.length > 0 && jacobi.rounds[0].prop.length > 0, 'jacobi emits a proposal with no reliance on rho');
ok(jacobi.rounds.every((rd) => rd.prop.every((p) => p.jacobi && p.sibs && p.sibs.length >= 1)), 'jacobi proposals keep the hover shape');
for (const rd of jacobi.rounds) {
  ok(rd.bill && rd.bill.sentence === `${rd.bill.proposed} - ${rd.bill.rejected} = ${rd.bill.acceptedN}`, 'round bill identity');
  for (let i = 0; i < rd.acc; i++) ok(rd.prop[i].tok === rd.truth[i], 'an accepted Jacobi token is one the reference emits');
  if (rd.acc < rd.prop.length) ok(rd.prop[rd.acc].tok !== rd.truth[rd.acc], 'the first rejected Jacobi token is not the reference');
}

const tpf = shownTpf(1.2);
ok(tpf.tpf === (Number(tpf.aS) + 1).toFixed(2), 'shownTpf identity');

const jLo = simulate({ depth: 4, plen: 6, rounds: 4, corpus: 8, seed: 0, extra: '', spec: 'jacobi' }, 0);
const jHi = simulate({ depth: 4, plen: 6, rounds: 4, corpus: 8, seed: 0, extra: '', spec: 'jacobi' }, 1);
const propToks = (row) => row.rounds.flatMap((rd) => rd.prop.map((p) => p.tok)).join(',');
ok(propToks(jLo) !== propToks(jHi), 'rho changes the Jacobi proposal tokens');
ok(jLo.mean === jHi.mean, 'the accepted-length curve stays put while those tokens change');
const last0 = jLo.rounds[0].ctx[jLo.rounds[0].ctx.length - 1];
ok(jLo.rounds[0].prop.every((p, i) => p.tok === last0 + String(i)), 'the window is the last context token plus a position index');

const windowSentence = 'The Jacobi window is the last context token plus a position index, not a copy of a repeated n-gram.';
const page = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
const readme = readFileSync(new URL('./README.md', import.meta.url), 'utf8');
const invariant = /does not feed the Jacobi|does not change what it writes/i;
ok(page.includes("from './math.js'"), 'page imports the shipped math');
ok(page.includes('simulate('), 'page calls simulate');
ok(!page.includes('gets BETTER with repetition'), 'page does not claim Jacobi improves with repetition');
ok(cardBlurb.includes(windowSentence), 'blurb names the Jacobi window');
ok(page.includes('the Jacobi window is the last context token plus a position index; it does not copy a repeated n-gram'), 'page names the Jacobi window');
ok(readme.includes(windowSentence), 'readme names the Jacobi window');
ok(!invariant.test(cardBlurb) && !invariant.test(page) && !invariant.test(readme), 'served card does not say the Jacobi window is invariant');
ok(/Jacobi mode does not move with that/.test(readme), 'the accepted-length claim stays about the curve handle');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS suffix-decoding');
