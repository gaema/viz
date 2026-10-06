// Forward value is the high-precision weight rounded onto the grid and scaled back.
// The backward step moves that same high-precision weight by grad * lr, as if
// rounding had derivative 1. A flat bin keeps the forward code until a boundary.

export function fmt2(x) {
  const n = +x;
  return (Number.isFinite(n) ? n : 0).toFixed(2);
}

export function qatForward(weight, scale, bits) {
  const weightText = fmt2(weight);
  const scaleText = fmt2(Math.max(0, +scale));
  const b = Math.max(2, Math.min(8, Math.round(+bits) || 8));
  const w = Number(weightText);
  const s = Number(scaleText);
  const qmax = (1 << (b - 1)) - 1;
  const qmin = -qmax;
  const code = s === 0 ? 0 : Math.max(qmin, Math.min(qmax, Math.round(w / s)));
  const forward = fmt2(code * s);
  return { weight: weightText, scale: scaleText, bits: String(b), qmin, qmax, code, forward };
}

export function qatSentence(row) {
  return `${row.code} * ${row.scale} = ${row.forward}`;
}

export function steStep(weight, grad, lr) {
  const weightText = fmt2(weight);
  const gradText = fmt2(Math.max(0, +grad));
  const lrText = fmt2(Math.max(0, +lr));
  const step = fmt2(Number(gradText) * Number(lrText));
  const next = fmt2(Number(weightText) - Number(step));
  return { weight: weightText, grad: gradText, lr: lrText, step, next };
}

export function steSentence(row) {
  return `${row.grad} * ${row.lr} = ${row.step}`;
}

export function steNextSentence(row) {
  return `${row.weight} - ${row.step} = ${row.next}`;
}
