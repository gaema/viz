// Teacher forcing adds one step of error at every frame, so frame n carries
// n times the one-step error. Self-forcing trains on the model's own frames,
// so every frame stays at that one-step error. The printed teacher bar is
// the printed step times the frame number.

export function frameErrors(frames, step) {
  const n = Math.max(1, frames | 0);
  const stepText = Math.max(0, +step).toFixed(3);
  const s = Number(stepText);
  const teacher = [];
  const self = [];
  for (let i = 0; i < n; i++) {
    teacher.push((s * (i + 1)).toFixed(3));
    self.push(stepText);
  }
  return { frames: n, step: stepText, teacher, self };
}
