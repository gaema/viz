// MLA cache and projection arithmetic. The absorbed query and output run at
// latent width. The per-head RoPE query is a separate projection of width
// d_R per head, d · n_h · d_R, and the two down-projections produce the
// cached latent and the shared RoPE key.

const DT = { fp16: 2, fp8: 1 };

export function derive(st) {
  const heads = st.heads | 0, hdim = st.hdim | 0, dc = st.dc | 0, dR = st.dR | 0;
  const d = st.hidden | 0, L = st.layers | 0, B = DT[st.kvdtype] || 2;
  const ctx = (st.ctxk | 0) * 1024;

  const mhaElem = 2 * heads * hdim;
  const mlaElem = dc + dR;

  const mhaTok = mhaElem * L * B;
  const mlaTok = mlaElem * L * B;
  const mhaAll = mhaTok * ctx;
  const mlaAll = mlaTok * ctx;

  const macMHA = 4 * d * heads * hdim;
  const macRope = d * heads * dR;
  const macMLA = 2 * d * heads * dc + macRope + d * (dc + dR);

  return {
    heads, hdim, dc, dR, d, L, B, ctx, mhaElem, mlaElem, mhaTok, mlaTok, mhaAll, mlaAll,
    macMHA, macMLA, macRope,
    shrink: mhaElem / mlaElem,
    cachePct: 100 * mlaElem / mhaElem,
    macPct: 100 * macMLA / macMHA,
  };
}

function macText(m) {
  return Math.round(m).toLocaleString('en-US');
}

function macValue(s) {
  return Number(String(s).replace(/,/g, ''));
}

// The percent is the quotient of the two counts in the sentence. A rounded
// "1.0M" is not that count, so it is not what the percent divides.
export function fmtBytes(b) {
  if (b >= 1073741824) return (b / 1073741824).toFixed(2) + ' GB';
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
  return Math.round(b) + ' B';
}

function parseBytes(s) {
  const t = String(s).trim();
  if (t.endsWith('GB')) return Number(t.slice(0, -2)) * 1073741824;
  if (t.endsWith('MB')) return Number(t.slice(0, -2)) * 1048576;
  if (t.endsWith('KB')) return Number(t.slice(0, -2)) * 1024;
  if (t.endsWith('B')) return Number(t.slice(0, -1));
  return Number(t);
}

// The percent divides the byte strings in the sentence. 480 B / 1.9 KB is
// not the true 480/1920.
export function byteCompare(mlaBytes, mhaBytes) {
  const mla = fmtBytes(mlaBytes);
  const mha = fmtBytes(mhaBytes);
  const pct = (100 * parseBytes(mla) / parseBytes(mha)).toFixed(2);
  return { mla, mha, pct, text: `${mla} vs ${mha} /token — ${pct}% of MHA` };
}

export function macPrice(macMHA, macMLA) {
  const mha = macText(macMHA);
  const mla = macText(macMLA);
  const pct = Math.round(100 * macValue(mla) / macValue(mha));
  return { mha, mla, pct, text: `${mha} → ${mla} = ${pct}% of MHA` };
}
