// Muon orthogonalises a 2-D gradient with the quintic Newton–Schulz iteration
// (coefficients 3.4445, -4.7750, 2.0315) so its singular values move toward 1.
// The iteration is a matrix method: embeddings, the output head, and
// per-channel gains stay on a second optimizer.

const A_COEF = 3.4445, B_COEF = -4.7750, C_COEF = 2.0315;

const MATRIX_KINDS = new Set(['matrix', 'weight']);

export function appliesTo(kind) {
  return MATRIX_KINDS.has(kind);
}

function matmul(X, Y) {
  const n = X.length, m = Y[0].length, k = Y.length;
  const O = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) for (let t = 0; t < k; t++) {
    const x = X[i][t];
    for (let j = 0; j < m; j++) O[i][j] += x * Y[t][j];
  }
  return O;
}

function transpose(X) {
  return Array.from({ length: X[0].length }, (_, j) => X.map((row) => row[j]));
}

function frobenius(X) {
  let s = 0;
  for (const row of X) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

function scale(X, s) {
  return X.map((row) => row.map((v) => v * s));
}

function add(X, Y) {
  return X.map((row, i) => row.map((v, j) => v + Y[i][j]));
}

// One polar iteration. X is tall-or-square; the caller transposes a wide matrix.
function newtonSchulzSquare(X, steps) {
  let M = scale(X, 1 / (frobenius(X) + 1e-7));
  for (let s = 0; s < steps; s++) {
    const A = matmul(M, transpose(M));
    const A2 = matmul(A, A);
    const poly = add(scale(A, B_COEF), scale(A2, C_COEF));
    M = add(scale(M, A_COEF), matmul(poly, M));
  }
  return M;
}

export function newtonSchulz(G, steps = 5) {
  const tall = G.length >= G[0].length;
  const X = tall ? G : transpose(G);
  const Y = newtonSchulzSquare(X, steps);
  return tall ? Y : transpose(Y);
}

function jacobiEigenvalues(A0) {
  const n = A0.length;
  const A = A0.map((row) => row.slice());
  for (let iter = 0; iter < 48; iter++) {
    let p = 0, q = 1, max = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const v = Math.abs(A[i][j]);
      if (v > max) { max = v; p = i; q = j; }
    }
    if (max < 1e-14) break;
    const app = A[p][p], aqq = A[q][q], apq = A[p][q];
    const tau = (aqq - app) / (2 * apq);
    const t = Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t), s = t * c;
    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const aik = A[k][p], aiq = A[k][q];
      const np = c * aik - s * aiq, nq = s * aik + c * aiq;
      A[k][p] = A[p][k] = np;
      A[k][q] = A[q][k] = nq;
    }
    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[p][q] = A[q][p] = 0;
  }
  return A.map((row, i) => row[i]);
}

export function singularValues(M) {
  const cols = M[0].length, rows = M.length;
  const G = Array.from({ length: cols }, () => Array(cols).fill(0));
  for (let i = 0; i < cols; i++) for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let r = 0; r < rows; r++) s += M[r][i] * M[r][j];
    G[i][j] = s;
  }
  return jacobiEigenvalues(G).map((e) => Math.sqrt(Math.max(0, e))).sort((a, b) => b - a);
}
