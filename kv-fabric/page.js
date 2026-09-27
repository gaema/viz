// Private caches hit only on the machine that stored the prefix.
// A shared pool hits when any machine has stored it. The printed percent
// is 100 times the printed hits over the printed request count.
import { mount } from '../framework/layout.js';
import { T, inkOn } from '../framework/theme.js';
import { hitPct, poolHits } from './math.js';

const LETTER = 'ABCDEFGH';

mount({
  mount: 'body',
  title: 'kv fabric — one pool across machines',
  blurb: 'A private cache hits only when the machine assigned this request has seen the prefix. A shared pool hits when any machine has. Repeating a prefix on a different machine raises the shared hit count and leaves the private one behind.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('prefixes', { label: 'distinct prefixes', min: 2, max: 4, step: 1, value: 4 });
    c.stepper('machines', { label: 'machines', min: 1, max: 4, step: 1, value: 4 });
    c.stepper('repeats', { label: 'arrivals of each prefix', min: 1, max: 3, step: 1, value: 2 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const bill = poolHits(page.state.prefixes, page.state.machines, page.state.repeats);
    const priv = hitPct(bill.privateHits, bill.requests);
    const shared = hitPct(bill.sharedHits, bill.requests);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);

    const waves = [];
    for (const row of bill.rows) {
      if (!waves[row.wave]) waves[row.wave] = [];
      if (!waves[row.wave][row.machine]) waves[row.wave][row.machine] = [];
      waves[row.wave][row.machine].push(row);
    }
    const R = waves.length;
    const M = page.state.machines | 0;
    const pad = W * 0.03;
    const gap = W * 0.02;
    const panelW = (W - pad * 2 - gap) / 2;
    const gridTop = H * 0.1;
    const gridH = H * 0.58;
    const panels = [
      { title: 'private', x: pad, hit: (row) => row.privateHit },
      { title: 'shared pool', x: pad + panelW + gap, hit: (row) => row.sharedHit },
    ];
    ctx.font = `${Math.max(11, H * 0.032)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    for (const panel of panels) {
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'left';
      ctx.fillText(panel.title, panel.x, H * 0.045);
      const cw = panelW / Math.max(1, M);
      const rh = gridH / Math.max(1, R);
      for (let r = 0; r < R; r++) {
        for (let m = 0; m < M; m++) {
          const cell = (waves[r] && waves[r][m]) || [];
          const x = panel.x + m * cw;
          const y = gridTop + r * rh;
          ctx.fillStyle = T.n3;
          ctx.fillRect(x + 2, y + 2, cw - 4, rh - 4);
          if (!cell.length) continue;
          const sub = (cw - 8) / cell.length;
          cell.forEach((row, i) => {
            const fill = panel.hit(row) ? T.teal : T.accent;
            ctx.fillStyle = fill;
            ctx.fillRect(x + 4 + i * sub, y + 6, Math.max(4, sub - 3), rh - 12);
            ctx.fillStyle = inkOn(fill);
            ctx.textAlign = 'center';
            ctx.fillText(LETTER[row.prefix] || '?', x + 4 + i * sub + Math.max(4, sub - 3) / 2, y + rh / 2);
          });
        }
      }
    }

    const barY = H * 0.78;
    const barH = H * 0.1;
    const bars = [
      ['private', priv, T.teal],
      ['shared pool', shared, T.teal],
    ];
    bars.forEach((item, i) => {
      const x = pad + i * (panelW + gap);
      const frac = Number(item[1].requests) === 0 ? 0 : Number(item[1].hits) / Number(item[1].requests);
      ctx.fillStyle = T.n3;
      ctx.fillRect(x, barY, panelW, barH);
      ctx.fillStyle = item[2];
      ctx.fillRect(x, barY, panelW * frac, barH);
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'left';
      ctx.fillText(`${item[0]} ${item[1].hits} of ${item[1].requests} = ${item[1].pct}%`, x, barY + barH + H * 0.045);
    });

    page.setReadout(
      `private ${priv.hits} of ${priv.requests} = ${priv.pct}%. `
      + `shared pool ${shared.hits} of ${shared.requests} = ${shared.pct}%.`,
    );
    page.probe = { privateHits: bill.privateHits, sharedHits: bill.sharedHits, requests: bill.requests, priv, shared };
  },
});
