import { fimLabel, fimSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const open = fimLabel(3, 2, 2, false);
ok(open.one === 7, 'one example is prefix plus suffix plus middle');
ok(open.suffixSeen === open.suffix, 'the first middle token sees every suffix token');
ok(open.cross === open.one * open.one, 'an open mask lets the second example see the square of one example');
ok(fimSentence(open) === 'prefix 3, suffix 2, middle 2. the first middle token sees 2 of 2 suffix tokens. cross pairs 49 = 7 × 7.', 'the open sentence is the printed product');

const shut = fimLabel(3, 2, 2, true);
ok(shut.cross === 0, 'isolating the examples removes the cross pairs');
ok(shut.suffixSeen === 2, 'isolation does not hide an example from its own suffix');
ok(fimSentence(shut).endsWith('cross pairs 0.'), 'the blocked sentence prints 0 and not a false product');

function cells(P, S, M, isolate) {
  const row = fimLabel(P, S, M, isolate);
  const n = row.one * 2;
  let crossCells = 0;
  const qFirst = row.prefix + row.suffix;
  let firstSuffix = 0;
  let everyMiddle = true;
  for (let ex = 0; ex < 2; ex++) {
    for (let m = 0; m < row.middle; m++) {
      const q = ex * row.one + row.prefix + row.suffix + m;
      let seen = 0;
      for (let s = 0; s < row.suffix; s++) {
        const k = ex * row.one + row.prefix + s;
        if (k <= q && Math.floor(q / row.one) === Math.floor(k / row.one)) seen++;
      }
      if (seen !== row.suffix) everyMiddle = false;
    }
  }
  for (let q = 0; q < n; q++) {
    for (let k = 0; k < n; k++) {
      if (k > q) continue;
      const same = Math.floor(q / row.one) === Math.floor(k / row.one);
      if (!same) crossCells++;
      const local = k % row.one;
      const suffix = local >= row.prefix && local < row.prefix + row.suffix;
      if (q === qFirst && same && suffix) firstSuffix++;
    }
  }
  return { row, crossCells, firstSuffix, everyMiddle };
}

for (let P = 1; P <= 4; P++) {
  for (let S = 1; S <= 3; S++) {
    for (let M = 1; M <= 3; M++) {
      const openCells = cells(P, S, M, false);
      const shutCells = cells(P, S, M, true);
      const product = fimSentence(openCells.row).match(/cross pairs (\d+) = (\d+) × (\d+)\./);
      ok(openCells.everyMiddle && shutCells.everyMiddle, `every middle token sees its suffix at ${P},${S},${M}`);
      ok(openCells.firstSuffix === S && shutCells.firstSuffix === S, `the first middle token sees ${S} suffix tokens`);
      ok(product && Number(product[1]) === Number(product[2]) * Number(product[3]), `the open product is printed at ${P},${S},${M}`);
      ok(Number(product[2]) === openCells.row.one && Number(product[1]) === openCells.crossCells, `the printed product is the cross-example cells at ${P},${S},${M}`);
      ok(shutCells.crossCells === openCells.crossCells && shutCells.row.cross === 0, `isolate blocks those cells and prints 0 at ${P},${S},${M}`);
    }
  }
}

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS fill-in-the-middle');
