// LoRA: ΔW = B A with rank r. A full din×dout update is not representable
// once r is smaller than both sides. Merging BA into W costs nothing extra
// at serve time; leaving it unmerged adds one thin GEMM per token.

export function loraParams(din, dout, r) {
  return r * (din + dout);
}

export function fullParams(din, dout) {
  return din * dout;
}

// B is dout×r, A is r×din. Product is dout×din.
export function loraProduct(B, A) {
  const dout = B.length, r = A.length, din = A[0].length;
  const W = Array.from({ length: dout }, () => Array(din).fill(0));
  for (let i = 0; i < dout; i++) {
    for (let k = 0; k < r; k++) {
      const b = B[i][k];
      if (b === 0) continue;
      for (let j = 0; j < din; j++) W[i][j] += b * A[k][j];
    }
  }
  return W;
}

export function rankOf(M, tol = 1e-8) {
  const A = M.map((row) => row.slice());
  const rows = A.length, cols = A[0].length;
  let rank = 0, col = 0;
  for (let r = 0; r < rows && col < cols;) {
    let piv = r;
    for (let i = r + 1; i < rows; i++) if (Math.abs(A[i][col]) > Math.abs(A[piv][col])) piv = i;
    if (Math.abs(A[piv][col]) <= tol) { col++; continue; }
    if (piv !== r) { const tmp = A[r]; A[r] = A[piv]; A[piv] = tmp; }
    const pv = A[r][col];
    for (let i = r + 1; i < rows; i++) {
      const f = A[i][col] / pv;
      for (let j = col; j < cols; j++) A[i][j] -= f * A[r][j];
    }
    rank++; r++; col++;
  }
  return rank;
}

// Extra MACs per the serving choice, on top of the frozen base matmul.
// Merged: the sum is already inside W, so the extra is 0.
// Unmerged: y = x W + (x A) B, which is r*(din + dout) MACs per token.
export function extraMacs(din, dout, r, tokens, merged) {
  if (merged) return 0;
  return tokens * r * (din + dout);
}
