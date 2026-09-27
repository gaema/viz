// A block-diffusion drafter proposes `width` tokens in one parallel step.
// Position 0 is the draft the model is most sure of. Each later position is
// accepted with probability quality^(index+1), so the tail of a wider block
// is accepted less often than the tail of a short one.

export function positionAccept(index, quality) {
  const q = Math.min(1, Math.max(0, quality));
  return q ** (index + 1);
}

export function tailAccept(width, quality) {
  return positionAccept(Math.max(1, width | 0) - 1, quality);
}
