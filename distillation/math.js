// Temperature softmax of a teacher. When the teacher's argmax is the wrong
// class, the soft target still puts more mass there than on the truth, and a
// student matching that target copies the error. A hard label does not.

export function softmax(logits, T = 1) {
  const t = T > 1e-6 ? T : 1e-6;
  let m = -Infinity;
  for (const z of logits) m = Math.max(m, z / t);
  const e = logits.map((z) => Math.exp(z / t - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function argmax(p) {
  let k = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
  return k;
}

// True when the soft target the student is asked to match prefers the
// teacher's wrong class over the labelled truth.
export function copiesTeacherError(teacherLogits, truth, T = 1) {
  const p = softmax(teacherLogits, T);
  const guess = argmax(p);
  return guess !== truth && p[guess] > p[truth];
}

export function softMass(logits, index, T = 1) {
  return softmax(logits, T)[index];
}
