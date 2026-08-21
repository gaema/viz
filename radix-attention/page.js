// radix-attention concept page -- sharing KV cache ACROSS requests with a
// radix tree of token prefixes.
//
// The idea a serving runtime exploits: two requests that begin with the same
// tokens have, layer for layer, the same keys and values for that shared span.
// So instead of one cache per request, keep ONE tree keyed by token sequence:
// a shared prefix is a single path, a divergence is a branch, and every node
// owns the KV for its own token span. A new request walks the tree, REUSES the
// longest prefix it matches, and only computes the unmatched suffix.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: each step admits the next request. Its matched prefix lights up
//    as "reused" and its suffix as "computed"; the tree grows a branch. Auto-
//    plays and loops.
//  - DIRECT MANIPULATION: drag the ▽ divergence handle on any request row (or
//    use the two steppers) to move where that request stops agreeing with the
//    one above it. The tree re-shapes and the reuse counter moves under your
//    hand -- drag a handle right and watch computed tokens turn into reused
//    ones.
//  - HOVER: a tree node reports its token span, how many requests share it, its
//    KV footprint and when it was last touched; a request row reports the path
//    it takes through the tree and its reused/computed split.
//  - EVICTION: the tree is finite. When the token count exceeds capacity the
//    least-recently-used LEAF is dropped -- never an interior node that still
//    has live children beneath it, because those children's KV is only
//    addressable through it.
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

// Token vocabulary. Symbols are integers minted in creation order; two tokens
// are the same token iff they carry the same symbol, so a shared prefix is
// literally an equal integer run. The words only make the picture readable.
const WORDS = [
  '<s>', 'You', 'are', 'helpful', 'assistant', '.', 'Always', 'answer',
  'briefly', 'Example', ':', '2+2', '=', '4', 'Q', 'how', 'do', 'I', 'sort',
  'list', '?', 'A', 'use', 'sorted', '()', 'reverse', 'string', 'join', 'dict',
  'keys', 'map', 'filter', 'lambda', 'print', 'len', 'range', 'sum', 'max',
  'min', 'append', 'pop', 'slice', 'tuple', 'set', 'loop', 'while', 'for',
  'if', 'else', 'return', 'def', 'class', 'import', 'json', 'open', 'read',
  'write', 'path', 'file', 'line', 'split', 'strip', 'lower', 'upper', 'find',
  'index', 'count', 'copy', 'deep', 'fast', 'slow', 'why', 'when', 'where',
  'what', 'which', 'who',
];
const word = (s) => (s < WORDS.length ? WORDS[s] : 't' + s);

// Where request r+1 stops agreeing with request r, as a fraction of the prompt
// length. Only the DEFAULT shape -- every one of these is draggable.
const DEFAULT_FRAC = [0.6, 0.45, 0.6, 0.3, 0.7];

// Fixed attention shape behind the KV-size readout (the per-layer cache holds a
// key and a value for every token, for every KV head).
const KV_HEADS = 8, HEAD_DIM = 128, DTYPE_BYTES = 2;
const bytesPerToken = (layers) => 2 * layers * KV_HEADS * HEAD_DIM * DTYPE_BYTES;

function fmtBytes(b) {
  if (b >= 1024 * 1024 * 1024) return (b / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KB';
  return b + ' B';
}

function defaultShares(R, L) {
  const a = [];
  for (let i = 0; i < R - 1; i++) a.push(Math.round(L * DEFAULT_FRAC[i % DEFAULT_FRAC.length]));
  return a;
}

// state.shares is a compact "6,4,6" string -- one entry per request after the
// first, giving how many leading tokens it copies from the request above it.
// Kept as a string so the framework's deep-link sync carries it in the URL for
// free (?shares=6,4,6), which is also the headless stand-in for a handle drag.
function parseShares(st) {
  const R = st.reqs, L = st.plen;
  const raw = String(st.shares == null ? '' : st.shares).split(',').map((x) => parseInt(x, 10));
  const def = defaultShares(R, L);
  const out = [];
  for (let i = 0; i < R - 1; i++) {
    const v = Number.isFinite(raw[i]) ? raw[i] : def[i];
    out.push(Math.max(0, Math.min(L, v)));
  }
  return out;
}

// Requests as token-symbol arrays. Request 0 mints L fresh symbols; request r
// copies the first shares[r-1] symbols of request r-1 and mints the rest, so
// prefix agreement is exact and transitive.
function buildRequests(st) {
  const R = st.reqs, L = st.plen, sh = parseShares(st), out = [];
  let next = 0;
  for (let r = 0; r < R; r++) {
    const toks = [];
    const k = r === 0 ? 0 : sh[r - 1];
    for (let p = 0; p < L; p++) toks.push(r > 0 && p < k ? out[r - 1][p] : next++);
    out.push(toks);
  }
  return out;
}

// ---- the radix tree -------------------------------------------------------
// A node owns a SPAN of tokens (path compression: an unbranched run is one
// node) plus the KV for exactly those tokens. `refs` counts the requests whose
// path crosses it; `last` is when it was last touched (the LRU key).
function newTree() {
  return { nodes: [{ id: 0, span: [], parent: -1, children: [], refs: 0, last: -1, depth: 0 }], nextId: 1 };
}

function insert(tree, toks, stepIdx) {
  const N = tree.nodes;
  let cur = N[0], i = 0;
  const path = [];
  cur.refs++; cur.last = stepIdx;
  while (i < toks.length) {
    let child = null;
    for (const cid of cur.children) if (N[cid].span[0] === toks[i]) { child = N[cid]; break; }
    if (!child) break;
    let j = 0;
    while (j < child.span.length && i + j < toks.length && child.span[j] === toks[i + j]) j++;
    if (j === 0) break;
    if (j < child.span.length) {
      // Partial match inside a span -> SPLIT: the head keeps the agreed tokens,
      // a new tail node inherits the rest (and the subtree, and the ref count).
      const tail = { id: tree.nextId++, span: child.span.slice(j), parent: child.id, children: child.children.slice(), refs: child.refs, last: child.last, depth: child.depth + j };
      N[tail.id] = tail;
      for (const gc of tail.children) N[gc].parent = tail.id;
      child.children = [tail.id];
      child.span = child.span.slice(0, j);
    }
    child.refs++; child.last = stepIdx;
    path.push(child.id);
    i += j; cur = child;
  }
  let newId = -1;
  if (i < toks.length) {
    const n = { id: tree.nextId++, span: toks.slice(i), parent: cur.id, children: [], refs: 1, last: stepIdx, depth: i };
    N[n.id] = n; cur.children.push(n.id); newId = n.id;
  }
  return { matched: i, computed: toks.length - i, path, newId };
}

const liveTokens = (N) => N.reduce((s, n) => s + (n.dead ? 0 : n.span.length), 0);

// LRU eviction, leaves only. An interior node with live children can never go:
// its children's KV is addressable only through the path that runs over it, so
// dropping it would strand everything below. The span just written by the
// arriving request is also off limits -- it is in use.
function evictToCapacity(tree, cap, stepIdx) {
  const N = tree.nodes, dead = [];
  let guard = 0;
  while (liveTokens(N) > cap && guard++ < 128) {
    let best = null;
    for (const n of N) {
      if (n.dead || n.id === 0) continue;
      if (n.children.length) continue;          // interior: protected
      if (n.last === stepIdx) continue;         // in use right now
      if (!best || n.last < best.last || (n.last === best.last && n.id > best.id)) best = n;
    }
    if (!best) break;
    best.dead = true; dead.push(best.id);
    const p = N[best.parent];
    if (p) p.children = p.children.filter((c) => c !== best.id);
  }
  return dead;
}

const snap = (tree) => tree.nodes.map((n) => ({ id: n.id, span: n.span.slice(), parent: n.parent, children: n.children.slice(), refs: n.refs, last: n.last, depth: n.depth, dead: !!n.dead }));

// One transport step per arriving request.
function buildSteps(st) {
  const toks = buildRequests(st), tree = newTree(), steps = [];
  let cumR = 0, cumC = 0;
  for (let r = 0; r < st.reqs; r++) {
    const res = insert(tree, toks[r], r);
    const dead = evictToCapacity(tree, st.capacity, r);
    cumR += res.matched; cumC += res.computed;
    steps.push({
      r, matched: res.matched, computed: res.computed, path: res.path, newId: res.newId,
      evicted: dead, nodes: snap(tree), cumR, cumC, total: liveTokens(tree.nodes),
      label: `request ${r} arrives — ${res.matched} token${res.matched === 1 ? '' : 's'} reused, ${res.computed} computed`
        + (dead.length ? ` · evicted ${dead.length} LRU leaf node${dead.length === 1 ? '' : 's'}` : ''),
    });
  }
  return steps;
}

// Vertical placement: every live leaf takes the next row, every parent sits at
// the mean of its children, so a shared prefix draws as one trunk. Nodes
// evicted on THIS step get ghost rows underneath.
function layoutRows(nodes, ghosts) {
  const rows = {};
  let row = 0;
  const visit = (id) => {
    const n = nodes[id];
    const kids = n.children.filter((c) => nodes[c] && !nodes[c].dead);
    if (!kids.length) { rows[id] = row++; return rows[id]; }
    const rs = kids.map(visit);
    rows[id] = rs.reduce((a, b) => a + b, 0) / rs.length;
    return rows[id];
  };
  visit(0);
  for (const id of ghosts) rows[id] = row++;
  return rows;
}

function fit(ctx, s, maxW) {
  s = String(s);
  if (ctx.measureText(s).width <= maxW) return s;
  while (s.length > 1 && ctx.measureText(s + '…').width > maxW) s = s.slice(0, -1);
  return s + '…';
}

function roundRect(ctx, x, y, w, h, rr) {
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// Hit-test geometry captured each draw(), reused by onPointer + the hover tips.
let geom = null;
let dragIdx = -1;   // which divergence handle is grabbed (index into shares)

// Write one divergence point and re-run the transport, so the tree, the branch
// structure and the counters all follow the handle.
function setShare(page, idx, val) {
  const st = page.state, L = st.plen;
  const sh = parseShares(st);
  if (idx < 0 || idx >= sh.length) return;
  sh[idx] = Math.max(0, Math.min(L, Math.round(val)));
  page.controls.set('shares', sh.join(','), { rebuild: true, silent: true });
  if (st.req === idx + 1) page.controls.set('diverge', sh[idx], { silent: true });
  page.redraw();
}

mount({
  mount: 'body',
  title: 'radix-attention — one KV cache shared across requests',
  blurb: 'Requests that start with the same tokens have the same keys and values for that span, so a serving runtime stores prompts in a radix tree instead of one cache per request: a shared prefix is ONE path, a divergence is a branch, and every node owns the KV for its own token span. Step the transport to admit requests one at a time — the matched prefix lights up as reused, the rest is computed. Drag a ▽ divergence handle to move where a request stops agreeing with the one above it and watch the tree re-shape and the reuse counter move. When the tree exceeds capacity, the least-recently-used LEAF is evicted; an interior node with live children never is.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  autoplay: true,
  compare: { stepA: 'first', stepB: 'last', labelA: 'first request — cold tree, everything computed', labelB: 'last request — warm tree, prefix reused' },
  onPointer: (page, ev) => {
    if (!geom) return;
    if (ev.type === 'down') {
      dragIdx = -1;
      for (const h of geom.handles) {
        if (Math.abs(ev.x - h.x) <= 10 && ev.y >= h.y - 4 && ev.y <= h.y + h.h + 4) { dragIdx = h.idx; break; }
      }
      if (dragIdx >= 0) setShare(page, dragIdx, (ev.x - geom.gridX) / geom.tw);
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragIdx = -1;
    } else if (ev.type === 'move' && dragIdx >= 0 && page.pointer.down) {
      setShare(page, dragIdx, (ev.x - geom.gridX) / geom.tw);
    }
  },
  challenges: [
    { goal: 'Make one request a total cache hit — 0 tokens computed for it.', hint: 'drag a ▽ handle all the way to the right end of its row, then step to that request.', check: (api) => ({ solved: !!api.probe.r && api.probe.computedNow === 0, detail: `request ${api.probe.r ?? '–'} computed ${api.probe.computedNow ?? '–'} token(s) — needs 0` }) },
    { goal: 'Serve more than half of all tokens from the cache.', hint: 'push the divergence points right: the further two requests agree, the longer the shared trunk.', check: (api) => ({ solved: (api.probe.reusePct ?? 0) > 50, detail: `${(api.probe.reusePct ?? 0).toFixed(1)}% of tokens reused — needs > 50%` }) },
  ],
  controls: (c, page) => {
    c.state.shares = '';                     // filled from ?shares= or the defaults
    c.stepper('reqs', { label: 'requests', min: 2, max: 6, value: 4 });
    c.stepper('plen', { label: 'tokens per request', min: 4, max: 14, value: 10 });
    c.stepper('req', {
      label: 'edit divergence of request', min: 1, max: 5, value: 1, rebuild: false,
      onInput: (v, st) => { const sh = parseShares(st); const i = Math.min(v, st.reqs - 1) - 1; if (sh[i] != null) page.controls.set('diverge', sh[i], { silent: true }); },
    });
    c.stepper('diverge', {
      label: '…diverges at token', min: 0, max: 14, value: 6, rebuild: true,
      onInput: (v, st) => setShare(page, Math.min(st.req, st.reqs - 1) - 1, v),
    });
    c.slider('capacity', { label: 'cache capacity (tokens)', min: 8, max: 96, step: 2, value: 20, rebuild: true });
    c.stepper('layers', { label: 'layers (KV size)', min: 4, max: 80, value: 32, rebuild: false });
    c.transport({ compute: () => buildSteps(page.state), speed: 0.8, loop: true });
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    const R = st.reqs, L = st.plen, s = page.step();
    const toks = buildRequests(st), sh = parseShares(st);
    const bpt = bytesPerToken(st.layers);

    const pad = 14, gridX = pad + 54, rightPad = 96;
    const tw = Math.max(16, Math.min(46, (W - gridX - pad - rightPad) / L));
    const rowH = 22, rowGap = 4, reqTop = 58;
    const handles = [], nodeBoxes = [], rowBoxes = [];

    // ---- header -----------------------------------------------------------
    r.label(s ? `step ${s.r + 1} / ${R}  —  ${s.label}` : `${R} requests waiting — press ▶ (or step) to admit the first one`,
      pad, 26, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label('incoming requests (each row is one prompt, left to right in token order)', pad, 44, { color: T.n11, font: '11px ui-monospace, monospace' });

    // ---- request strips ---------------------------------------------------
    ctx.save();
    ctx.font = '9.5px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let q = 0; q < R; q++) {
      const y = reqTop + q * (rowH + rowGap);
      const admitted = s ? q <= s.r : false;
      const isCur = s ? q === s.r : false;
      rowBoxes.push({ q, x: pad, y, w: gridX + L * tw - pad, h: rowH });
      ctx.textAlign = 'left';
      ctx.fillStyle = isCur ? T.n14 : admitted ? T.n11 : T.n9;
      ctx.font = (isCur ? 'bold ' : '') + '11px ui-monospace, monospace';
      ctx.fillText('req ' + q, pad, y + rowH / 2);
      ctx.textAlign = 'center';
      ctx.font = '9.5px ui-monospace, monospace';
      for (let p = 0; p < L; p++) {
        const x = gridX + p * tw;
        let fill, ink, edge;
        if (isCur) {
          const reused = p < s.matched;
          fill = alphaOf(reused ? T.ok : T.warn, 0.32);
          edge = reused ? T.ok : T.warn;
          ink = T.n14;
        } else if (admitted) {
          fill = alphaOf(T.n9, 0.16); edge = T.n6; ink = T.n11;
        } else {
          fill = rgbaToken('n9', 0.06); edge = T.n4; ink = T.n9;
        }
        ctx.fillStyle = fill; ctx.fillRect(x + 1, y, tw - 2, rowH);
        ctx.strokeStyle = edge; ctx.lineWidth = 1; ctx.strokeRect(x + 1.5, y + 0.5, tw - 3, rowH - 1);
        ctx.fillStyle = ink;
        ctx.fillText(fit(ctx, word(toks[q][p]), tw - 6), x + tw / 2, y + rowH / 2 + 0.5);
      }
      if (isCur) {
        ctx.textAlign = 'left';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillStyle = T.ok; ctx.fillText(`↺ ${s.matched}`, gridX + L * tw + 8, y + rowH / 2 - 5);
        ctx.fillStyle = T.warn; ctx.fillText(`⚙ ${s.computed}`, gridX + L * tw + 8, y + rowH / 2 + 7);
        ctx.textAlign = 'center';
      }
      // divergence handle: where this request stops copying the one above it
      if (q > 0) {
        const k = sh[q - 1], hx = gridX + k * tw;
        const grabbed = dragIdx === q - 1;
        handles.push({ idx: q - 1, x: hx, y, h: rowH, k, q });
        ctx.save();
        ctx.strokeStyle = grabbed ? T.violet : alphaOf(T.violet, 0.75);
        ctx.lineWidth = grabbed ? 2.5 : 1.5;
        ctx.beginPath(); ctx.moveTo(hx, y - 3); ctx.lineTo(hx, y + rowH + 3); ctx.stroke();
        ctx.fillStyle = grabbed ? T.violet : alphaOf(T.violet, 0.85);
        ctx.beginPath(); ctx.moveTo(hx - 5, y - 9); ctx.lineTo(hx + 5, y - 9); ctx.lineTo(hx, y - 2); ctx.closePath(); ctx.fill();
        ctx.restore();
      }
    }
    ctx.restore();
    const reqBottom = reqTop + R * (rowH + rowGap);
    r.label('↔ drag a ▽ handle: it marks where that request diverges from the one above — everything left of it is a shared prefix',
      pad, reqBottom + 13, { color: T.violet, font: '10px ui-monospace, monospace' });

    // ---- the radix tree ---------------------------------------------------
    const treeTop = reqBottom + 46;
    r.label('radix tree of prompt prefixes — one node per token span, holding that span\'s KV', pad, treeTop - 8, { color: T.n11, font: '11px ui-monospace, monospace' });
    const nodes = s ? s.nodes : null;
    const nh = 22;
    const treeBottom = H - 62;
    if (nodes) {
      const ghosts = s.evicted;
      const rows = layoutRows(nodes, ghosts);
      // Row pitch stretches to fill the band the tree has, so a 3-leaf tree does
      // not huddle at the top of an empty panel.
      const maxRow = Math.max(0, ...Object.values(rows));
      const pitch = Math.max(nh + 6, Math.min(46, (treeBottom - treeTop - 18) / (maxRow + 1)));
      const yOf = (id) => treeTop + 8 + rows[id] * pitch;
      const onPath = new Set(s.path);
      // edges first, so boxes sit on top
      ctx.save();
      ctx.strokeStyle = T.n7; ctx.lineWidth = 1.2;
      for (const n of nodes) {
        if (n.id === 0 || n.dead || rows[n.id] == null) continue;
        const p = nodes[n.parent];
        if (!p || rows[p.id] == null) continue;
        const x = gridX + n.depth * tw, yp = yOf(p.id) + nh / 2, yc = yOf(n.id) + nh / 2;
        ctx.beginPath(); ctx.moveTo(x - 7, yp); ctx.lineTo(x - 7, yc); ctx.lineTo(x, yc); ctx.stroke();
      }
      ctx.restore();
      // root marker
      ctx.save();
      ctx.fillStyle = T.n9;
      ctx.beginPath(); ctx.arc(gridX - 7, yOf(0) + nh / 2, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('root', gridX - 14, yOf(0) + nh / 2);
      ctx.restore();

      ctx.save();
      ctx.textBaseline = 'middle';
      for (const n of nodes) {
        if (n.id === 0 || rows[n.id] == null) continue;
        const ghost = n.dead;
        if (ghost && !ghosts.includes(n.id)) continue;
        const x = gridX + n.depth * tw, y = yOf(n.id), w = Math.max(10, n.span.length * tw - 4);
        nodeBoxes.push({ id: n.id, x, y, w, h: nh, node: n });
        ctx.save();
        roundRect(ctx, x, y, w, nh, 5);
        if (ghost) {
          ctx.fillStyle = rgbaToken('n9', 0.10); ctx.fill();
          ctx.setLineDash([4, 3]); ctx.strokeStyle = T.bad; ctx.lineWidth = 1.2; ctx.stroke();
        } else {
          // deeper fill = shared by more requests
          ctx.fillStyle = alphaOf(T.accent, 0.12 + 0.5 * Math.min(1, (n.refs - 1) / Math.max(1, R - 1)));
          ctx.fill();
          const fresh = n.id === s.newId;
          ctx.strokeStyle = fresh ? T.warn : onPath.has(n.id) ? T.ok : T.accentLine;
          ctx.lineWidth = fresh || onPath.has(n.id) ? 2 : 1;
          ctx.stroke();
        }
        ctx.restore();
        ctx.font = '9.5px ui-monospace, monospace';
        ctx.textAlign = 'left';
        ctx.fillStyle = ghost ? T.bad : T.n14;
        const txt = ghost ? '✕ ' + n.span.map(word).join(' ') : n.span.map(word).join(' ');
        ctx.fillText(fit(ctx, txt, w - 24), x + 5, y + nh / 2);
        if (!ghost) {
          ctx.textAlign = 'right';
          ctx.fillStyle = T.n10; ctx.font = '9px ui-monospace, monospace';
          ctx.fillText('×' + n.refs, x + w - 4, y + nh / 2);
        }
      }
      ctx.restore();
      if (ghosts.length) r.label('✕ evicted: least-recently-used LEAF (an interior node with live children is never evictable)', pad, Math.min(treeBottom + 8, treeTop + 8 + (maxRow + 1) * pitch + 6), { color: T.bad, font: '10px ui-monospace, monospace' });
    } else {
      r.label('(empty — the tree is built as requests arrive)', gridX, treeTop + 20, { color: T.n9, font: '11px ui-monospace, monospace' });
    }

    // ---- counters ---------------------------------------------------------
    const cumR = s ? s.cumR : 0, cumC = s ? s.cumC : 0, tot = cumR + cumC;
    const pct = tot ? (100 * cumR) / tot : 0;
    const barY = H - 42, barX = gridX, barW = W - gridX - pad - 8;
    ctx.save();
    ctx.fillStyle = alphaOf(T.ok, 0.75); ctx.fillRect(barX, barY, barW * (tot ? cumR / tot : 0), 14);
    ctx.fillStyle = alphaOf(T.warn, 0.75); ctx.fillRect(barX + barW * (tot ? cumR / tot : 0), barY, barW * (tot ? cumC / tot : 1), 14);
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(barX, barY, barW, 14);
    ctx.restore();
    r.label('tokens', pad, barY + 11, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label(`↺ ${cumR} reused from cache   ⚙ ${cumC} computed   —   ${pct.toFixed(1)}% of all prompt tokens served from the tree`,
      barX, barY + 30, { color: T.n13, font: '11.5px ui-monospace, monospace' });
    const held = s ? s.total : 0;
    r.label(`cache ${held}/${st.capacity} tok · ${fmtBytes(held * bpt)}`, barX + barW, barY - 6, { color: held > st.capacity - 2 ? T.warn : T.n11, font: '10px ui-monospace, monospace', align: 'right' });

    geom = { gridX, tw, rowH, reqTop, rowGap, handles, nodeBoxes, rowBoxes, barX, barY, barW };

    // ---- hover-to-inspect -------------------------------------------------
    if (page.pointer.over && dragIdx < 0) {
      const p = page.pointer;
      let tip = null;
      for (const h of handles) {
        if (Math.abs(p.x - h.x) <= 10 && p.y >= h.y - 10 && p.y <= h.y + h.h + 4) {
          tip = `divergence of request ${h.q}\nit copies the first ${h.k} of ${L} tokens from request ${h.q - 1}\n↔ drag to move it: further right = longer shared trunk = fewer tokens computed`;
          break;
        }
      }
      for (const b of nodeBoxes) {
        if (tip) break;
        if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
          const n = b.node;
          tip = n.dead
            ? `node ${n.id} — EVICTED this step\nspan "${n.span.map(word).join(' ')}" (${n.span.length} tok)\nit was a LEAF and the least recently used (last touched at request ${n.last})\nfreed ${fmtBytes(n.span.length * bpt)}; the next request needing it must recompute`
            : `node ${n.id}  ·  tokens ${n.depth}..${n.depth + n.span.length - 1}\nspan "${n.span.map(word).join(' ')}" (${n.span.length} tok)\nshared by ${n.refs} request${n.refs === 1 ? '' : 's'} — stored ONCE, read by all of them\nKV = 2 (K,V) × ${st.layers} layers × ${KV_HEADS} kv heads × ${HEAD_DIM} dim × ${DTYPE_BYTES} B\n   = ${fmtBytes(bpt)}/token × ${n.span.length} = ${fmtBytes(n.span.length * bpt)}\nlast used at request ${n.last} (the LRU key)`;
          break;
        }
      }
      for (const b of rowBoxes) {
        if (tip) break;
        if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
          if (!s || b.q > s.r) { tip = `request ${b.q} — not admitted yet\n${L} tokens: ${toks[b.q].map(word).join(' ')}`; break; }
          const path = b.q === s.r && nodes
            ? 'root → ' + s.path.map((id) => `n${id}[${nodes[id].span.map(word).join(' ')}]`).join(' → ') + (s.newId >= 0 ? ` → NEW n${s.newId}[${nodes[s.newId].span.map(word).join(' ')}]` : '')
            : '(step to this request to trace its path)';
          const m = b.q === s.r ? s.matched : null;
          tip = `request ${b.q}: ${L} tokens\n${toks[b.q].map(word).join(' ')}\n`
            + (m == null ? '' : `longest matched prefix = ${m} tok reused · ${L - m} tok computed\n`)
            + `path: ${path}`;
          break;
        }
      }
      if (!tip && p.x >= barX - 6 && p.x <= barX + barW && p.y >= barY - 8 && p.y <= barY + 24) {
        tip = `${cumR} reused + ${cumC} computed = ${tot} prompt tokens so far\nwithout prefix sharing all ${tot} would be computed\nsaved ${fmtBytes(cumR * bpt)} of recompute at ${fmtBytes(bpt)}/token`;
      }
      if (tip) page.setTip(tip);
    }

    page.probe = { r: s ? s.r : null, computedNow: s ? s.computed : null, reusePct: pct, held };

    let o = s
      ? `request ${s.r}: walk the tree, longest matched prefix = ${s.matched}/${L} tokens → REUSE their KV; compute only the ${s.computed}-token suffix and hang it off a new node.\n`
      : 'no request admitted yet — the tree is empty, so the first request matches nothing and computes every token.\n';
    o += `running total: ${cumR} reused · ${cumC} computed · ${pct.toFixed(1)}% of prompt tokens served from cache · tree holds ${held}/${st.capacity} tokens (${fmtBytes(held * bpt)} at ${fmtBytes(bpt)}/token).\n`;
    o += 'A node = one token span + its KV. Insert splits a node when a new request agrees with only part of its span; a full match is a free prefill. Eviction takes the least-recently-used LEAF only — an interior node still carries the path to its children.';
    page.setReadout(o);
  },
}).then((page) => {
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?shares=6,4,6 sets the divergence points (headless stand-in for dragging the
  // ▽ handles, since a screenshot has no pointer). One entry per request after
  // the first; each is clamped to [0, tokens-per-request].
  if (q.has('shares')) page.controls.set('shares', q.get('shares'), { rebuild: true, silent: true });
  const sh0 = parseShares(page.state);
  page.controls.set('diverge', sh0[Math.min(page.state.req, page.state.reqs - 1) - 1] ?? 0, { silent: true });
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover).
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: any explicit hook pauses autoplay first.
  if (q.has('step') || q.has('hover') || q.has('shares')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
