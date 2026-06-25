// depthwise-separable concept page -- MobileNet factorizes a standard conv into
// a DEPTHWISE conv (one k×k filter per input channel, NO channel mixing) followed
// by a POINTWISE 1×1 conv (mixes channels, NO spatial extent). A full conv does
// spatial + channel jointly; the factorization does them separately, which is
// why the MAC count drops by ~k²·Cout/(k²+Cout) ≈ 8-9× for 3×3.
//   full  MACs = H·W·Cin·Cout·k²
//   dwsep MACs = H·W·Cin·k²  (depthwise)  +  H·W·Cin·Cout  (pointwise)
//   ratio = full/dwsep = (k²·Cout)/(k² + Cout) = 1 / (1/Cout + 1/k²)
// Drag the input/output channel stacks (or use the sliders) and watch the bar.
import { mount } from '../framework/layout.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df';
const PAL = ['#1f6feb', '#d2691e', '#2ca02c', '#8250df', '#d1242f', '#0a9396', '#bc6c25', '#5a189a'];
const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + 'G' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : '' + Math.round(n);

let stdInRect = null, stdOutRect = null, dragMode = '', lastY = 0;

// a vertical column of n channel squares; returns square center coords.
function col(ctx, cx, yc, n, sz, glow) {
  const gap = sz + 3, h = n * gap - 3, y0 = yc - h / 2, cs = [];
  for (let i = 0; i < n; i++) {
    const y = y0 + i * gap;
    ctx.fillStyle = PAL[i % PAL.length]; ctx.globalAlpha = glow ? 1 : 0.85; ctx.fillRect(cx - sz / 2, y, sz, sz); ctx.globalAlpha = 1;
    cs.push([cx, y + sz / 2]);
  }
  return { cs, x: cx - sz / 2, y: y0, w: sz, h };
}

mount({
  mount: 'body',
  title: 'depthwise-separable — MobileNet conv, the FLOP cut',
  blurb: 'A standard convolution mixes space AND channels in one shot: every output channel reads a k×k patch of every input channel, costing H·W·Cin·Cout·k² multiply-adds. MobileNet splits that into two cheap steps. DEPTHWISE: one k×k filter per input channel, applied independently — spatial filtering with NO channel mixing (Cin·k² · H·W). POINTWISE: a 1×1 conv that mixes channels — channel mixing with NO spatial extent (Cin·Cout · H·W). Same input→output shape, but the cost drops from a PRODUCT (Cin·Cout·k²) to a SUM (Cin·k² + Cin·Cout). The ratio is (k²·Cout)/(k²+Cout) ≈ 8–9× fewer MACs for a 3×3 conv into many channels — the trick behind on-phone CNNs. Drag the input or output channel stack (or use the sliders) to change Cin/Cout, k, and the feature-map size, and watch the MAC bar and the ratio update. The two stages animate in turn.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  controls: (c, page) => {
    c.slider('k', { label: 'kernel k', min: 1, max: 7, step: 2, value: 3 });
    c.slider('Cin', { label: 'in channels', min: 3, max: 64, step: 1, value: 16 });
    c.slider('Cout', { label: 'out channels', min: 4, max: 128, step: 4, value: 32 });
    c.slider('HW', { label: 'feature map H=W', min: 8, max: 112, step: 8, value: 56 });
  },
  onPointer: (page, ev) => {
    const inside = (rc, x, y) => rc && x >= rc.x - 14 && x <= rc.x + rc.w + 14 && y >= rc.y - 10 && y <= rc.y + rc.h + 10;
    if (ev.type === 'down') { dragMode = inside(stdInRect, ev.x, ev.y) ? 'Cin' : inside(stdOutRect, ev.x, ev.y) ? 'Cout' : ''; lastY = ev.y; }
    else if (ev.type === 'up' || ev.type === 'leave') dragMode = '';
    else if (ev.type === 'move' && dragMode && page.pointer.down) {
      const d = Math.round((lastY - ev.y) / 6);
      if (d !== 0) {
        if (dragMode === 'Cin') page.controls.set('Cin', Math.max(3, Math.min(64, page.state.Cin + d)));
        else page.controls.set('Cout', Math.max(4, Math.min(128, page.state.Cout + 4 * d)));
        lastY = ev.y; page.redraw();
      }
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear('#ffffff');
    const k = st.k | 0, Cin = st.Cin | 0, Cout = st.Cout | 0, hw = st.HW | 0;
    const full = hw * hw * Cin * Cout * k * k, dw = hw * hw * Cin * k * k, pw = hw * hw * Cin * Cout, dwsep = dw + pw;
    const ratio = full / dwsep;
    const nIn = Math.min(Cin, 7), nOut = Math.min(Cout, 7), sz = 10, yc = 100;
    const stage = ((page.t || 0) % 3) < 1.5 ? 'dw' : 'pw';  // animate the two stages in turn

    // ===== standard conv (left) =====
    r.label('standard convolution', 20, 34, { color: INK, font: '12px ui-monospace, monospace' });
    const a = col(ctx, 56, yc, nIn, sz, true); stdInRect = a;
    const b = col(ctx, 196, yc, nOut, sz, true); stdOutRect = b;
    ctx.save(); ctx.strokeStyle = 'rgba(130,80,223,0.22)'; ctx.lineWidth = 0.7; // dense connections: every in→every out
    for (const p of a.cs) for (const q of b.cs) { ctx.beginPath(); ctx.moveTo(p[0] + sz / 2, p[1]); ctx.lineTo(q[0] - sz / 2, q[1]); ctx.stroke(); } ctx.restore();
    r.label('Cin (drag ↕)', 30, yc + a.h / 2 + 16, { color: '#586069', font: '9px ui-monospace, monospace' });
    r.label('Cout (drag ↕)', 168, yc + b.h / 2 + 16, { color: '#586069', font: '9px ui-monospace, monospace' });
    // k×k spatial badge on the link
    drawKbadge(ctx, r, 126, yc - 42, k, PURPLE, `${k}×${k} + all channels`);
    r.label(`MACs = H·W·Cin·Cout·k² = ${fmt(full)}  (Cin=${Cin}→Cout=${Cout}, k=${k})`, 20, yc + a.h / 2 + 32, { color: PURPLE, font: '10px ui-monospace, monospace' });

    // ===== depthwise separable (right) =====
    const x0 = 360;
    r.label('depthwise separable', x0, 34, { color: INK, font: '12px ui-monospace, monospace' });
    const di = col(ctx, x0 + 36, yc, nIn, sz, stage === 'dw'); const dm = col(ctx, x0 + 150, yc, nIn, sz, true); const dou = col(ctx, x0 + 264, yc, nOut, sz, stage === 'pw');
    // depthwise: channel i -> channel i only (no mixing), k×k spatial
    ctx.save(); ctx.lineWidth = stage === 'dw' ? 1.6 : 1; for (let i = 0; i < nIn; i++) { ctx.strokeStyle = stage === 'dw' ? GREEN : 'rgba(44,160,44,0.4)'; ctx.beginPath(); ctx.moveTo(di.cs[i][0] + sz / 2, di.cs[i][1]); ctx.lineTo(dm.cs[i][0] - sz / 2, dm.cs[i][1]); ctx.stroke(); } ctx.restore();
    // pointwise: every mid -> every out, 1×1 (channel mix, no space)
    ctx.save(); ctx.lineWidth = 0.7; for (const p of dm.cs) for (const q of dou.cs) { ctx.strokeStyle = stage === 'pw' ? 'rgba(210,105,30,0.5)' : 'rgba(210,105,30,0.18)'; ctx.beginPath(); ctx.moveTo(p[0] + sz / 2, p[1]); ctx.lineTo(q[0] - sz / 2, q[1]); ctx.stroke(); } ctx.restore();
    drawKbadge(ctx, r, x0 + 93, yc - 42, k, GREEN, `${k}×${k} per-channel`);
    drawKbadge(ctx, r, x0 + 207, yc - 42, 1, ORANGE, `1×1 mix`);
    r.label('depthwise', x0 + 18, yc + di.h / 2 + 16, { color: GREEN, font: '9px ui-monospace, monospace' });
    r.label('pointwise', x0 + 200, yc + dou.h / 2 + 16, { color: ORANGE, font: '9px ui-monospace, monospace' });
    r.label(`depthwise ${fmt(dw)}  +  pointwise ${fmt(pw)}  =  ${fmt(dwsep)} MACs`, x0, yc + di.h / 2 + 32, { color: INK, font: '10px ui-monospace, monospace' });

    // ===== MAC comparison bars (bottom) =====
    const by = 206, bx = 20, bw = W - 40, maxv = full;
    r.label('multiply-add (MAC) cost — drag a channel stack to rescale', bx, by - 8, { color: INK, font: '11px ui-monospace, monospace' });
    const barH = 20;
    // full
    ctx.fillStyle = PURPLE; ctx.fillRect(bx, by, bw * (full / maxv), barH);
    r.label(`full conv: ${fmt(full)}`, bx + 6, by + 14, { color: '#fff', font: '10px ui-monospace, monospace' });
    // dwsep (dw + pw segments)
    const y2 = by + barH + 8, wdw = bw * (dw / maxv), wpw = bw * (pw / maxv);
    ctx.fillStyle = GREEN; ctx.fillRect(bx, y2, wdw, barH); ctx.fillStyle = ORANGE; ctx.fillRect(bx + wdw, y2, wpw, barH);
    r.label(`depthwise+pointwise: ${fmt(dwsep)}`, bx + 6, y2 + 14, { color: '#fff', font: '10px ui-monospace, monospace' });
    // ratio call-out
    ctx.save(); ctx.fillStyle = INK; ctx.font = 'bold 15px ui-monospace, monospace'; ctx.textAlign = 'left';
    ctx.fillText(`${ratio.toFixed(1)}× fewer`, bx + Math.max(wdw + wpw + 12, 160), y2 + 15); ctx.restore();
    r.label(`ratio = (k²·Cout)/(k²+Cout) = (${k * k}·${Cout})/(${k * k}+${Cout}) = ${ratio.toFixed(2)}×   ·   params: ${fmt(Cin * Cout * k * k)} → ${fmt(Cin * k * k + Cin * Cout)} (${(Cin * Cout * k * k / (Cin * k * k + Cin * Cout)).toFixed(1)}×)`, bx, y2 + barH + 18, { color: '#586069', font: '10px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && !dragMode) {
      const p = page.pointer;
      if (stdInRect && p.x >= stdInRect.x - 14 && p.x <= stdInRect.x + stdInRect.w + 14 && p.y >= stdInRect.y && p.y <= stdInRect.y + stdInRect.h) page.setTip(`${Cin} input channels\ndrag ↕ to change Cin`);
      else if (stdOutRect && p.x >= stdOutRect.x - 14 && p.x <= stdOutRect.x + stdOutRect.w + 14 && p.y >= stdOutRect.y && p.y <= stdOutRect.y + stdOutRect.h) page.setTip(`${Cout} output channels\ndrag ↕ to change Cout`);
      else if (p.y >= by && p.y <= y2 + barH) page.setTip(`full conv: ${fmt(full)} MACs\ndepthwise: ${fmt(dw)}\npointwise: ${fmt(pw)}\ndw-sep total: ${fmt(dwsep)}  →  ${ratio.toFixed(1)}× cheaper`);
    }

    let o = `depthwise-separable conv: split a full conv (spatial+channel jointly) into depthwise (k×k per channel) + pointwise (1×1 mix).   tier:${r.name}\n`;
    o += `H=W=${hw}, Cin=${Cin}, Cout=${Cout}, k=${k}:  full = ${fmt(full)} MACs vs depthwise ${fmt(dw)} + pointwise ${fmt(pw)} = ${fmt(dwsep)}  →  ${ratio.toFixed(1)}× fewer (cost goes from a PRODUCT Cin·Cout·k² to a SUM Cin·k²+Cin·Cout).`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__dsPage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['k', 'Cin', 'Cout', 'HW']) if (q.has(key)) page.controls.set(key, +q.get(key));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});

function drawKbadge(ctx, r, cx, cy, k, color, label) {
  const n = Math.min(k, 5), c = 4, w = n * c;
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 0.8;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) ctx.strokeRect(cx - w / 2 + j * c, cy - w / 2 + i * c, c, c);
  ctx.restore();
  r.label(label, cx - 30, cy - w / 2 - 4, { color, font: '8px ui-monospace, monospace' });
}
