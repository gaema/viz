// multimodal-inject concept page -- how a vision-language / audio-language model
// feeds a NON-TEXT input into a text LLM. There is no separate "image port": the
// image is encoded (ViT) into patch features, a PROJECTOR (small MLP, D_enc->D)
// maps those into the LLM's token-embedding space, and the resulting "soft" image
// tokens are SPLICED into the text token sequence at a <image> placeholder. The
// merged sequence -- text embeddings and image/audio tokens, all D-dim -- is then
// processed by ordinary self-attention, which attends across modalities uniformly.
// Same trick for audio (encoder -> projector -> audio tokens). Drag the injected
// block to move where it lands; toggle modality + token count.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df', RED = '#d1242f';
const D = 16, SRC = { text: BLUE, image: ORANGE, audio: GREEN };
const PROMPTS = {
  'caption': ['Describe', 'the', '<m>', 'in', 'detail', '.'],
  'vqa': ['What', 'color', 'is', 'the', '<m>', '?'],
  'qa': ['Given', '<m>', 'answer', 'the', 'question', '.'],
};
const sign = (v, d) => { const t = Math.max(-1, Math.min(1, v / (d || 1))), m = Math.abs(t); return t >= 0 ? `rgb(255,${Math.round(255 - m * 150)},${Math.round(255 - m * 165)})` : `rgb(${Math.round(255 - m * 165)},${Math.round(255 - m * 120)},255)`; };
const hash = (s) => { let h = 7; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0; return h % 9973; };

let cur = null, bsig = '', injectAt = 0, seqRect = null, dragging = false;

function buildImage(seed, S, gN) {
  const cv = document.createElement('canvas'); cv.width = S; cv.height = S; const c = cv.getContext('2d'), im = c.createImageData(S, S), d = im.data;
  const rnd = seededRandn(seed, 12, { std: 1 }), blobs = [[230, 80, 80], [70, 130, 230], [70, 200, 130]].map((col, b) => ({ cx: (rnd[b * 2] * 0.5 + 0.5) * S, cy: (rnd[b * 2 + 1] * 0.5 + 0.5) * S, r: S * 0.3, col }));
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { let r = 235, g = 235, b = 235; for (const bl of blobs) { const w = Math.exp(-((x - bl.cx) ** 2 + (y - bl.cy) ** 2) / (2 * bl.r * bl.r)); r = r * (1 - w) + bl.col[0] * w; g = g * (1 - w) + bl.col[1] * w; b = b * (1 - w) + bl.col[2] * w; } const i = (y * S + x) * 4; d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255; }
  c.putImageData(im, 0, 0);
  // per-patch mean colour feature
  const feat = []; const P = S / gN;
  for (let pr = 0; pr < gN; pr++) for (let pc = 0; pc < gN; pc++) { let R = 0, G = 0, B = 0; for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const si = ((pr * P + y) * S + (pc * P + x)) * 4; R += d[si]; G += d[si + 1]; B += d[si + 2]; } const n = P * P; feat.push([R / n / 255 - 0.5, G / n / 255 - 0.5, B / n / 255 - 0.5]); }
  return { cv, feat };
}
function buildAudio(seed, N) {
  const L = 256, s = new Float32Array(L), rnd = seededRandn(seed, L, { std: 1 });
  for (let i = 0; i < L; i++) s[i] = 0.6 * Math.sin(i * 0.13) + 0.3 * Math.sin(i * 0.41 + 1) + 0.15 * rnd[i] * (0.5 + 0.5 * Math.sin(i * 0.02));
  const feat = [], fl = L / N;
  for (let f = 0; f < N; f++) { let rms = 0, zc = 0; for (let i = 0; i < fl; i++) { const v = s[f * fl + i]; rms += v * v; if (i > 0 && (v >= 0) !== (s[f * fl + i - 1] >= 0)) zc++; } feat.push([Math.sqrt(rms / fl) - 0.3, zc / fl - 0.3, (f / N) - 0.5]); }
  return { s, feat };
}
function project(feat3, Wp, seed, idx) { const out = new Float32Array(D), sd = seededRandn(seed + idx * 7, 5, { std: 0.6 }); const f = [feat3[0], feat3[1], feat3[2], sd[0], sd[1], sd[2], sd[3], sd[4]]; for (let o = 0; o < D; o++) { let v = 0; for (let j = 0; j < 8; j++) v += Wp[o * 8 + j] * f[j]; out[o] = v; } return out; }

function build(st) {
  const N = +st.N, gN = Math.round(Math.sqrt(N)), seed = st.seed | 0;
  const Wp = seededRandn(seed + 99, [D, 8], { std: 0.5 }).data;
  const words = PROMPTS[st.prompt], textTokens = [], placeholderPos = words.indexOf('<m>');
  words.forEach((w, i) => { if (w !== '<m>') textTokens.push({ src: 'text', label: w, emb: seededRandn(hash(w), D, { std: 0.8 }) }); });
  const phIdxInText = words.slice(0, placeholderPos).filter((w) => w !== '<m>').length;  // text index where media goes
  const img = (st.modality !== 'audio') ? buildImage(seed, 48, gN) : null;
  const aud = (st.modality !== 'image') ? buildAudio(seed + 5, N) : null;
  const imgTokens = img ? img.feat.map((f, i) => ({ src: 'image', label: 'img', emb: project(f, Wp, seed, i) })) : [];
  const audTokens = aud ? aud.feat.map((f, i) => ({ src: 'audio', label: 'aud', emb: project(f, Wp, seed + 1000, i) })) : [];
  const media = st.modality === 'image' ? imgTokens : st.modality === 'audio' ? audTokens : imgTokens.concat(audTokens);
  return { N, gN, textTokens, media, img, aud, phIdxInText };
}

mount({
  mount: 'body',
  title: 'multimodal-inject — vision/audio tokens in the text stream',
  blurb: 'How does an image get into a text LLM? Not through a separate input — it is turned into TOKENS that live in the same embedding space as words. An encoder (a ViT for images, an audio encoder for sound) turns the input into feature vectors; a PROJECTOR (a small MLP, encoder-dim → the LLM’s embedding dim D) maps them into the LLM’s token space; and those "soft" media tokens are SPLICED into the text token sequence at a <image>/<audio> placeholder. The merged sequence — text embeddings and image/audio tokens, every column the same D-dim vector — is then run through ordinary self-attention, which attends across text and media uniformly (the model never "knows" which columns came from pixels). This is exactly how LLaVA, Qwen-VL, and audio-LMs work. Drag the orange/green media block to move where it is injected; switch the modality (image / audio / both) and the token count; hover a column to see its source.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  animate: true,
  controls: (c, page) => {
    c.select('modality', { label: 'modality', options: ['image', 'audio', 'both'], value: 'image', rebuild: true });
    c.select('N', { label: 'media tokens', options: ['4', '9', '16'], value: '9', rebuild: true });
    c.select('prompt', { label: 'prompt', options: Object.keys(PROMPTS), value: 'vqa', rebuild: true });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !seqRect) return;
    const nT = cur.textTokens.length, slots = nT + 1, slotW = seqRect.w / (nT + cur.media.length);
    const toSlot = (x) => Math.max(0, Math.min(nT, Math.round((x - seqRect.x) / slotW - cur.media.length / 2)));
    // only start drag if grabbing the media block
    const mediaStart = seqRect.x + injectAt * slotW, mediaEnd = mediaStart + cur.media.length * slotW;
    if (ev.type === 'down') { dragging = (ev.x >= mediaStart && ev.x <= mediaEnd && ev.y >= seqRect.y && ev.y <= seqRect.y + seqRect.h); }
    else if (ev.type === 'up' || ev.type === 'leave') dragging = false;
    else if (ev.type === 'move' && dragging && page.pointer.down) { injectAt = toSlot(ev.x); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    const sig = `${st.modality}|${st.N}|${st.prompt}|${st.seed}`;
    if (sig !== bsig) { cur = build(st); injectAt = cur.phIdxInText; bsig = sig; }
    r.clear('#ffffff');
    const { textTokens, media, gN } = cur, nT = textTokens.length, nM = media.length;
    // merged sequence (text with media spliced at injectAt)
    const merged = textTokens.slice(0, injectAt).concat(media, textTokens.slice(injectAt));

    // ===== encoder + projector (top-left) =====
    let ex = 20; const ey = 50;
    r.label(st.modality === 'audio' ? 'audio' : 'image', ex, ey - 8, { color: INK, font: '11px ui-monospace, monospace' });
    if (cur.img) { const isz = st.modality === 'both' ? 52 : 70; ctx.drawImage(cur.img.cv, ex, ey, isz, isz); ctx.save(); ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 0.6; for (let i = 0; i <= gN; i++) { ctx.beginPath(); ctx.moveTo(ex + i * isz / gN, ey); ctx.lineTo(ex + i * isz / gN, ey + isz); ctx.stroke(); ctx.beginPath(); ctx.moveTo(ex, ey + i * isz / gN); ctx.lineTo(ex + isz, ey + i * isz / gN); ctx.stroke(); } ctx.restore(); ex += isz + 8; }
    if (cur.aud) { const aw = 80, ah = st.modality === 'both' ? 28 : 50, ay = st.modality === 'both' ? ey + 56 : ey; ctx.save(); ctx.strokeStyle = '#e6e8ea'; ctx.strokeRect(st.modality === 'both' ? 20 : ex, ay, aw, ah); ctx.strokeStyle = GREEN; ctx.lineWidth = 1; ctx.beginPath(); const ax0 = (st.modality === 'both' ? 20 : ex); for (let i = 0; i < cur.aud.s.length; i++) { const px = ax0 + i / cur.aud.s.length * aw, py = ay + ah / 2 - cur.aud.s[i] * ah * 0.4; if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); } ctx.stroke(); ctx.restore(); if (st.modality !== 'both') ex += aw + 8; }
    // encoder -> projector box
    const bx = (st.modality === 'both') ? 110 : ex, boxY = ey + 8;
    ctx.save(); ctx.fillStyle = 'rgba(130,80,223,0.08)'; ctx.fillRect(bx, boxY, 124, 52); ctx.strokeStyle = PURPLE; ctx.lineWidth = 1.3; ctx.strokeRect(bx, boxY, 124, 52); ctx.fillStyle = INK; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(st.modality === 'audio' ? 'audio encoder' : 'ViT encoder', bx + 62, boxY + 16); ctx.fillStyle = PURPLE; ctx.fillText('projector MLP', bx + 62, boxY + 32); ctx.fillStyle = '#444'; ctx.fillText('D_enc → D=' + D, bx + 62, boxY + 46); ctx.restore();
    r.label(`→ ${nM} ${st.modality === 'both' ? 'media' : st.modality} tokens (D-dim)`, bx, boxY + 66, { color: ORANGE, font: '10px ui-monospace, monospace' });

    // ===== prompt (top-right) =====
    const prx = bx + 150, pry = ey - 2; let wx = prx;
    r.label('text prompt  (<m> = media placeholder)', prx, pry - 8, { color: INK, font: '10px ui-monospace, monospace' });
    const words = PROMPTS[st.prompt];
    ctx.save(); ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
    for (const w of words) { const isPh = w === '<m>', lab = isPh ? (st.modality === 'audio' ? '<aud>' : '<img>') : w, tw = ctx.measureText(lab).width + 8; ctx.fillStyle = isPh ? 'rgba(210,105,30,0.15)' : '#f2f4f6'; ctx.fillRect(wx, pry + 8, tw, 16); ctx.strokeStyle = isPh ? ORANGE : '#d0d7de'; ctx.strokeRect(wx, pry + 8, tw, 16); ctx.fillStyle = isPh ? ORANGE : INK; ctx.fillText(lab, wx + 4, pry + 20); wx += tw + 4; if (wx > W - 90) { wx = prx; pry += 22; } }
    ctx.restore();
    r.label('the placeholder expands into the media tokens →', prx, pry + 40, { color: '#586069', font: '9px ui-monospace, monospace' });

    // ===== merged sequence heatmap =====
    const sx = 20, sy = 168, sw = W - 40, slotW = Math.min(30, sw / merged.length), seqW = slotW * merged.length, ch = Math.min(8, 120 / D), seqH = ch * D;
    seqRect = { x: sx, y: sy, w: seqW, h: seqH };
    r.label('merged token sequence — text + media, all D-dim, fed to the LLM  (drag the media block ↔)', sx, sy - 8, { color: INK, font: '11px ui-monospace, monospace' });
    ctx.save();
    merged.forEach((tok, i) => {
      const cx0 = sx + i * slotW;
      for (let o = 0; o < D; o++) { ctx.fillStyle = sign(tok.emb[o], 1.4); ctx.fillRect(cx0, sy + o * ch, slotW - 0.6, ch - 0.3); }
      ctx.strokeStyle = SRC[tok.src]; ctx.lineWidth = tok.src === 'text' ? 0.8 : 1.6; ctx.strokeRect(cx0 - 0.5, sy - 0.5, slotW, seqH + 1);
      // source tick + label above
      ctx.fillStyle = SRC[tok.src]; ctx.fillRect(cx0, sy - 5, slotW - 0.6, 3);
      if (tok.src === 'text') { ctx.save(); ctx.fillStyle = INK; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(tok.label.length > 5 ? tok.label.slice(0, 5) : tok.label, cx0 + slotW / 2, sy + seqH + 11); ctx.restore(); }
    });
    // media block bracket
    const mb0 = sx + injectAt * slotW, mbW = nM * slotW;
    ctx.strokeStyle = media[0] ? SRC[media[0].src] : ORANGE; ctx.lineWidth = 2; ctx.strokeRect(mb0 - 1.5, sy - 7, mbW + 3, seqH + 9);
    ctx.fillStyle = media[0] ? SRC[media[0].src] : ORANGE; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(`${nM} ${st.modality === 'both' ? 'media' : st.modality} tokens injected here`, mb0 + mbW / 2, sy + seqH + 24);
    ctx.restore();
    // connector projector -> media block (animated)
    ctx.save(); ctx.strokeStyle = ORANGE; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(bx + 62, boxY + 52); ctx.bezierCurveTo(bx + 62, sy - 30, mb0 + mbW / 2, sy - 40, mb0 + mbW / 2, sy - 8); ctx.stroke(); ctx.setLineDash([]);
    const fp = (page.t || 0) % 1.5 / 1.5, sxp = bx + 62, syp = boxY + 52, exp2 = mb0 + mbW / 2, fx = sxp + (exp2 - sxp) * fp, fy = syp + (sy - 8 - syp) * fp; ctx.fillStyle = ORANGE; ctx.beginPath(); ctx.arc(fx, fy, 2.6, 0, 7); ctx.fill(); ctx.restore();

    // ===== LLM bar =====
    const ly = sy + seqH + 36;
    ctx.save(); ctx.fillStyle = 'rgba(31,111,235,0.08)'; ctx.fillRect(sx, ly, seqW, 24); ctx.strokeStyle = BLUE; ctx.lineWidth = 1.3; ctx.strokeRect(sx, ly, seqW, 24); ctx.fillStyle = BLUE; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('LLM — self-attention over ALL tokens (text + media), uniformly', sx + seqW / 2, ly + 16); ctx.restore();
    // arrows from each token down into the LLM
    ctx.save(); ctx.strokeStyle = 'rgba(150,160,170,0.5)'; ctx.lineWidth = 0.6; for (let i = 0; i < merged.length; i++) { const cx0 = sx + i * slotW + slotW / 2; ctx.beginPath(); ctx.moveTo(cx0, sy + seqH + 26); ctx.lineTo(cx0, ly - 1); ctx.stroke(); } ctx.restore();

    // ===== self-attention grid over the merged tokens (causal) =====
    const L = merged.length, scal = 1 / Math.sqrt(D), att = [];
    for (let i = 0; i < L; i++) { const row = new Float32Array(L); let mx = -1e9; for (let j = 0; j <= i; j++) { let s = 0; for (let o = 0; o < D; o++) s += merged[i].emb[o] * merged[j].emb[o]; row[j] = s * scal; if (row[j] > mx) mx = row[j]; } let sum = 0; for (let j = 0; j <= i; j++) { row[j] = Math.exp(row[j] - mx); sum += row[j]; } for (let j = 0; j <= i; j++) row[j] /= sum; att.push(row); }
    const gy = ly + 44, gSide = Math.min(140, H - (ly + 58)), ac = gSide / L, gx = 40;
    const qRow = (page.pointer.over && page.pointer.x >= gx && page.pointer.x <= gx + gSide && page.pointer.y >= gy && page.pointer.y <= gy + gSide) ? Math.floor((page.pointer.y - gy) / ac) : L - 1;  // hovered row, else the last token
    r.label('self-attention weights (causal): row i attends to columns j ≤ i — text rows weight image/audio columns too', gx, gy - 10, { color: INK, font: '11px ui-monospace, monospace' });
    page._attRect = { x: gx, y: gy, w: gSide, h: gSide, ac, L };
    ctx.save();
    // source ticks: left (rows) + top (cols)
    for (let i = 0; i < L; i++) { ctx.fillStyle = SRC[merged[i].src]; ctx.fillRect(gx - 5, gy + i * ac, 3, ac - 0.4); ctx.fillRect(gx + i * ac, gy - 5, ac - 0.4, 3); }
    // cells
    for (let i = 0; i < L; i++) for (let j = 0; j <= i; j++) { const w = att[i][j]; ctx.fillStyle = `rgba(31,111,235,${Math.min(1, Math.sqrt(w) * 1.05)})`; ctx.fillRect(gx + j * ac, gy + i * ac, ac - 0.4, ac - 0.4); }
    ctx.strokeStyle = '#e6e8ea'; ctx.lineWidth = 1; ctx.strokeRect(gx, gy, gSide, gSide);
    // highlight the query row
    if (qRow >= 0 && qRow < L) { ctx.strokeStyle = RED; ctx.lineWidth = 1.6; ctx.strokeRect(gx - 0.5, gy + qRow * ac - 0.5, gSide + 1, ac + 1); }
    ctx.restore();
    r.label('rows/cols ↓→ in sequence order;  tick colour = source (text/image/audio)', gx, gy + gSide + 12, { color: '#8a939b', font: '9px ui-monospace, monospace' });
    // query-row breakdown (how much of row qRow's attention lands on each modality)
    if (qRow >= 0 && qRow < L) {
      let wt = 0, wi = 0, wa = 0; for (let j = 0; j <= qRow; j++) { const s = merged[j].src; if (s === 'image') wi += att[qRow][j]; else if (s === 'audio') wa += att[qRow][j]; else wt += att[qRow][j]; }
      const lx = gx + gSide + 28, lyy = gy + 4;
      const qtok = merged[qRow];
      r.label(`query row = token ${qRow}: ${qtok.src}${qtok.src === 'text' ? ' "' + qtok.label + '"' : ''}`, lx, lyy, { color: RED, font: '11px ui-monospace, monospace' });
      r.label('its attention splits across modalities:', lx, lyy + 20, { color: INK, font: '10px ui-monospace, monospace' });
      const bars = [['text', wt, BLUE], ['image', wi, ORANGE], ['audio', wa, GREEN]].filter((b) => b[1] > 0.0001);
      bars.forEach((b, k) => { const yy = lyy + 38 + k * 20, bw = 150 * b[1]; ctx.save(); ctx.fillStyle = b[2]; ctx.fillRect(lx, yy - 9, bw, 12); ctx.restore(); r.label(`${b[0]} ${(b[1] * 100).toFixed(0)}%`, lx + Math.max(bw + 4, 4), yy, { color: INK, font: '10px ui-monospace, monospace' }); });
      r.label(qtok.src === 'text' && (wi + wa) > 0.05 ? '→ a text token reading across image/audio tokens: cross-modal attention.' : 'hover a different row to inspect its attention', lx, lyy + 38 + bars.length * 20 + 8, { color: '#586069', font: '9px ui-monospace, monospace' });
    }

    // hover
    if (page.pointer.over && !dragging && seqRect) {
      const p = page.pointer;
      if (p.x >= sx && p.x <= sx + seqW && p.y >= sy && p.y <= sy + seqH) { const i = Math.floor((p.x - sx) / slotW); if (i >= 0 && i < merged.length) { const t = merged[i]; page.setTip(`token ${i}: ${t.src}${t.src === 'text' ? ' "' + t.label + '"' : ' #' + (i - injectAt)}\nD=${D} embedding (same space as text)\n${t.src === 'text' ? 'lookup from the embedding table' : 'encoder → projector MLP → this vector'}`); } }
      else if (page._attRect) { const a = page._attRect; if (p.x >= a.x && p.x <= a.x + a.w && p.y >= a.y && p.y <= a.y + a.h) { const i = Math.floor((p.y - a.y) / a.ac), j = Math.floor((p.x - a.x) / a.ac); if (i >= 0 && i < a.L && j >= 0 && j < a.L) page.setTip(j > i ? `(masked: token ${i} cannot attend to future token ${j})` : `attn[${i}→${j}] = ${(att[i][j] * 100).toFixed(1)}%\n${merged[i].src} token ${i} attends to ${merged[j].src} token ${j}`); } }
    }

    let o = `multimodal injection: ${st.modality} encoder → projector MLP (D_enc→${D}) → ${nM} media tokens spliced into the text sequence at the placeholder.   tier:${r.name}\n`;
    o += `merged sequence = ${nT} text + ${nM} ${st.modality === 'both' ? 'media' : st.modality} tokens = ${merged.length} tokens, all ${D}-dim. Self-attention then mixes them uniformly — the LLM never sees a "modality type", only token vectors. The causal attention grid (below) shows each row's weights spread over text AND media columns; hover a row to see its modality split. Drag the ${st.modality === 'both' ? 'media' : st.modality} block to move the injection point (now after text token ${injectAt}).`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__mmPage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['modality', 'N', 'prompt']) if (q.has(key)) page.controls.set(key, q.get(key));
  if (q.has('seed')) page.controls.set('seed', +q.get('seed'));
  if (q.has('inject') && cur) injectAt = Math.max(0, Math.min(cur.textTokens.length, +q.get('inject')));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
