// TeaCache skips a sampler step whose residual magnitude is under the
// threshold. The drift is the sum of the residuals that were not applied.
// A higher threshold skips more, so the drift rises. The same skipped count
// is a larger share of a few-step schedule than of a long one.

export function skipped(residuals, threshold) {
  return residuals.filter((r) => Math.abs(r) < threshold);
}

export function drift(residuals, threshold) {
  return skipped(residuals, threshold).reduce((s, r) => s + Math.abs(r), 0);
}

// Share of the trajectory dropped. One skip out of four steps is a quarter
// of a few-step schedule and a much smaller share of a long one.
export function scheduleShare(skippedCount, steps) {
  return skippedCount / steps;
}
