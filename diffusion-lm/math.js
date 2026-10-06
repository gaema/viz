// Each step scores every still-masked position from the whole sequence, including
// later positions, and unmasks the highest score. The next step receives the
// updated tokens and mask only: no causal KV state. While any mask remains, the
// masked count falls by one. An empty mask stays empty.

export function scoreMasked(tokens, mask) {
  const scores = tokens.map(() => 0);
  for (let i = 0; i < tokens.length; i++) {
    if (!mask[i]) continue;
    let s = 0;
    for (let j = 0; j < tokens.length; j++) s += tokens[j] * (j + 1);
    scores[i] = s + i;
  }
  return scores;
}

export function diffusionStep(tokens, mask) {
  const scores = scoreMasked(tokens, mask);
  const maskedIdx = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) maskedIdx.push(i);
  const masked = maskedIdx.length;
  if (masked === 0) {
    return { tokens: tokens.slice(), mask: mask.slice(), scores, unmasked: -1, masked, next: 0, kv: null };
  }
  let best = maskedIdx[0];
  for (const i of maskedIdx) {
    if (scores[i] > scores[best] || (scores[i] === scores[best] && i < best)) best = i;
  }
  const nextTokens = tokens.slice();
  const nextMask = mask.slice();
  let pred = 0;
  for (let j = 0; j < tokens.length; j++) pred += tokens[j];
  nextTokens[best] = pred;
  nextMask[best] = false;
  return { tokens: nextTokens, mask: nextMask, scores, unmasked: best, masked, next: masked - 1, kv: null };
}

export function maskSentence(step) {
  const masked = String(step.masked);
  const sub = step.masked === 0 ? '0' : '1';
  const next = String(Number(masked) - Number(sub));
  return `${masked} - ${sub} = ${next}`;
}

export const DEMO_TOKENS = [1, 2, 3, 4];

export function diffusionAt(steps) {
  let tokens = DEMO_TOKENS.slice();
  let mask = tokens.map(() => true);
  const want = Math.max(0, steps | 0);
  let last = null;
  for (let i = 0; i < want && mask.some(Boolean); i++) {
    last = diffusionStep(tokens, mask);
    tokens = last.tokens;
    mask = last.mask;
  }
  const pending = diffusionStep(tokens, mask);
  return { tokens, mask, pending, ran: last };
}
