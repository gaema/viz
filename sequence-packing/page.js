// sequence-packing concept page -- pack several documents into one row and
// watch which mask keeps them independent. Uses the shared framework:
// layout.mount() + a mode select, with the mask math in ./math.js.
//
// Three documents of lengths [2, 3, 1] share ONE row of six tokens. The
// block-diagonal mask lets each token attend only inside its own document;
// the "full" mask (true everywhere) still trains -- and every off-block cell
// is a token reading across a document boundary. Hover any cell for the
// per-pair verdict; the readout carries the cross-document pair count that
// math.js computes.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';
import { T } from '../framework/theme.js';
import { packMask, crossDocumentPairs } from './math.js';

const LENGTHS = [2, 3, 1];
const TOKS = ['r0', 'r1', 'm0', 'm1', 'm2', 'o0'];

mount({
  mount: 'body',
  title: 'sequence-packing — one row, many documents',
  blurb: 'Short sequences waste a fixed-width training row, so several documents are PACKED into one row and the attention mask does the separating. Here three documents of lengths [2, 3, 1] share a single row of six tokens. Switch the mask mode: the block-diagonal mask keeps every token inside its own document (zero cross-document attention pairs); the full mask is true everywhere, it still trains, and tokens quietly read across the document boundary. Hover any cell for the per-pair verdict — the readout counts the cross-document pairs for the selected mode.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  challenges: [
    { goal: 'Find the mask mode where a token attends to a token from ANOTHER document.', hint: 'select mode "full" — then hover an off-block cell.', check: (api) => ({ solved: (api.probe.cross ?? 0) > 0, detail: `cross-document pairs: ${api.probe.cross ?? 0} (mode: ${api.probe.mode ?? 'block'})` }) },
  ],
  controls: (c) => {
    c.select('mode', {
      label: 'mask mode',
      options: [
        { value: 'block', label: 'block-diagonal (per document)' },
        { value: 'full', label: 'full (true everywhere)' },
      ],
      value: 'block',
    });
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    const mode = page.state.mode === 'full' ? 'full' : 'block';
    const { mask, doc } = packMask(LENGTHS, mode);
    const cross = crossDocumentPairs(mask, doc);
    page.probe = { mode, cross };

    r.clear(T.n0);
    const N = LENGTHS.reduce((a, b) => a + b, 0);
    const pad = 18, leftW = 60, headerH = 44;
    const gridX = pad + leftW, gridY = headerH;
    const cell = Math.max(24, Math.min((page.W - gridX - pad - 200) / N, (page.H - gridY - pad - 8) / N));
    const gridRect = { x: gridX, y: gridY, w: N * cell, h: N * cell };

    r.label('keys →', gridX, headerH - 22, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label('queries ↓', pad, gridY + 6, { color: T.n11, font: '11px ui-monospace, monospace' });

    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    for (let j = 0; j < N; j++) { ctx.fillStyle = T.n14; ctx.fillText(TOKS[j], gridX + j * cell + cell / 2, gridY - 6); }
    ctx.restore();
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) { ctx.fillStyle = T.n14; ctx.fillText(TOKS[i], gridX - 8, gridY + i * cell + cell / 2); }
    ctx.restore();

    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = gridX + j * cell, y = gridY + i * cell;
      ctx.fillStyle = mask[i][j] ? T.accent : T.n3;
      ctx.fillRect(x, y, cell - 1.5, cell - 1.5);
      if (mask[i][j] && doc[i] !== doc[j]) {
        ctx.save(); ctx.strokeStyle = T.bad; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.strokeRect(x + 1.5, y + 1.5, cell - 4.5, cell - 4.5); ctx.restore();
      }
    }
    // document block outlines -- the packing structure the mask must preserve
    let off = 0;
    ctx.save(); ctx.strokeStyle = T.n14; ctx.lineWidth = 2;
    for (const len of LENGTHS) { ctx.strokeRect(gridX + off * cell - 1, gridY + off * cell - 1, len * cell, len * cell); off += len; }
    ctx.restore();

    // legend (right of grid)
    const lx = gridX + N * cell + 22; let ly = gridY + 4;
    const sw = (col, txt) => { ctx.save(); ctx.fillStyle = col; ctx.fillRect(lx, ly - 9, 13, 13); ctx.restore(); r.label(txt, lx + 19, ly + 2, { color: T.n12, font: '11px ui-monospace, monospace' }); ly += 24; };
    sw(T.accent, 'allowed — same document');
    if (mode === 'block') {
      sw(T.n3, 'blocked — other document');
    } else {
      ctx.save();
      ctx.fillStyle = T.accent; ctx.fillRect(lx, ly - 9, 13, 13);
      ctx.strokeStyle = T.bad; ctx.lineWidth = 2; ctx.setLineDash([3, 2]);
      ctx.strokeRect(lx + 1, ly - 8, 11, 11);
      ctx.restore();
      r.label('allowed across a boundary', lx + 19, ly + 2, { color: T.n12, font: '11px ui-monospace, monospace' });
      ly += 24;
    }
    ly += 30;
    r.label('lengths [2, 3, 1] → one row', lx, ly + 2, { color: T.n11, font: '10px ui-monospace, monospace' });
    r.label('of 6 tokens, 3 documents', lx, ly + 16, { color: T.n11, font: '10px ui-monospace, monospace' });

    if (page.pointer.over) {
      const p = page.pointer;
      const mh = cellAt(gridRect, N, N, p.x, p.y);
      if (mh) {
        const i = mh.r, j = mh.c;
        let tip;
        if (!mask[i][j]) tip = `query ${i} "${TOKS[i]}" (doc ${doc[i]}) → key ${j} "${TOKS[j]}" (doc ${doc[j]}): BLOCKED\nmasked out, so the two documents cannot read each other`;
        else if (doc[i] === doc[j]) tip = `query ${i} "${TOKS[i]}" → key ${j} "${TOKS[j]}": allowed (same doc ${doc[i]})\nnormal in-row attention`;
        else tip = `query ${i} "${TOKS[i]}" (doc ${doc[i]}) → key ${j} "${TOKS[j]}" (doc ${doc[j]}): CROSS-DOCUMENT pair\nallowed only in "full" mode — the packed row stops being independent documents`;
        page.setTip(tip);
      }
    }

    let o = `packed row: [2, 3, 1] → 6 tokens from 3 documents · mask mode: ${mode === 'block' ? 'block-diagonal (attend inside your own document)' : 'full (true everywhere)'}\n`;
    o += `cross-document attention pairs: ${cross}` +
      (cross > 0 ? ' — each one is a token reading across a document boundary' : ' — zero, so each document trains as if it were alone in the row');
    o += `    tier:${r.name}`;
    page.setReadout(o);
  },
});
