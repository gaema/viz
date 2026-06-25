// Curriculum order -- mirrors the family/slug order in ../index.html. The
// shared mount() framework (layout.js) imports this to build the in-demo
// prev/next navigation strip. KEEP IN SYNC with index.html when pages are
// added or reordered (the catalogue is the source of truth for the card grid;
// this is the flat ordered list the per-page nav walks).
export const ORDER = [
  // A -- Foundational primitives
  { slug: 'matmul', family: 'A' },
  { slug: 'dot-product', family: 'A' },
  { slug: 'dtype-bits', family: 'A' },
  { slug: 'softmax', family: 'A' },
  // B -- Attention
  { slug: 'qkv-projection', family: 'B' },
  { slug: 'scaled-dot-attention', family: 'B' },
  { slug: 'multi-head', family: 'B' },
  { slug: 'gqa-mqa', family: 'B' },
  { slug: 'causal-mask', family: 'B' },
  { slug: 'attention-patterns', family: 'B' },
  { slug: 'kv-cache', family: 'B' },
  { slug: 'rope', family: 'B' },
  { slug: 'flash-attention', family: 'B' },
  // C -- Transformer block
  { slug: 'tokenization', family: 'C' },
  { slug: 'embedding', family: 'C' },
  { slug: 'mlp-gated', family: 'C' },
  { slug: 'activations', family: 'C' },
  { slug: 'normalization', family: 'C' },
  { slug: 'residual-stream', family: 'C' },
  { slug: 'lm-head', family: 'C' },
  { slug: 'transformer-block', family: 'C' },
  { slug: 'prefill-vs-decode', family: 'C' },
  { slug: 'sampling', family: 'C' },
  // D -- Mixture of Experts
  { slug: 'moe-routing', family: 'D' },
  { slug: 'moe-balance', family: 'D' },
  // E -- Non-transformer sequence (SSM)
  { slug: 'ssm-scan', family: 'E' },
  { slug: 'mamba-block', family: 'E' },
  { slug: 'gated-deltanet', family: 'E' },
  { slug: 'hybrid-by-layer', family: 'E' },
  // F -- CNN / vision
  { slug: 'convolution', family: 'F' },
  { slug: 'receptive-field', family: 'F' },
  { slug: 'pooling', family: 'F' },
  { slug: 'feature-maps', family: 'F' },
  { slug: 'batchnorm', family: 'F' },
  { slug: 'residual-block', family: 'F' },
  { slug: 'depthwise-separable', family: 'F' },
  { slug: 'patch-embedding', family: 'F' },
  // G -- Quantization / multimodal
  { slug: 'quantization', family: 'G' },
  { slug: 'vram-budget', family: 'G' },
  { slug: 'multimodal-inject', family: 'G' },
  // H -- End-to-end (capstone) + training
  { slug: 'forward-pass', family: 'H' },
  { slug: 'backprop', family: 'H' },
  // I -- Real-model grounding: real weights, fetched at runtime
  { slug: 'real-embeddings', family: 'I' },
  { slug: 'real-attention', family: 'I' },
  { slug: 'real-logits', family: 'I' },
  { slug: 'logit-lens', family: 'I' },
  { slug: 'real-quant', family: 'I' },
  { slug: 'real-generate', family: 'I' },
];

// Resolve the current page's slug from an explicit override or the URL path
// (…/<slug>/index.html or …/<slug>/). Returns '' when not on a demo page.
export function currentSlug(override) {
  if (override) return override;
  if (typeof location === 'undefined') return '';
  const parts = location.pathname.split('/').filter(Boolean);
  if (!parts.length) return '';
  const last = parts[parts.length - 1];
  return /\.html?$/.test(last) ? (parts[parts.length - 2] || '') : last;
}

// neighbours for the nav strip: { index, total, prev, next, family }
export function neighbours(slug) {
  const i = ORDER.findIndex((e) => e.slug === slug);
  if (i < 0) return { index: -1, total: ORDER.length, prev: null, next: null, family: '' };
  return { index: i, total: ORDER.length, prev: ORDER[i - 1] || null, next: ORDER[i + 1] || null, family: ORDER[i].family };
}
