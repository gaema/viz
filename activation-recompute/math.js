// Activation checkpointing. `segment` is how many layers are recomputed
// between stored boundaries.
//   memory        = number of stored boundaries (the open segment is recomputed)
//   extraPerLayer = the average recomputed prefix inside a segment
// Store-all is segment 1 (a boundary at every layer, no extra forward).
// Store-none is segment = layers (the single input, then a full extra forward).
// The sum of the two has a minimum in between, near sqrt(layers).

export function recomputeCost(layers, segment) {
  const s = Math.max(1, Math.min(layers, segment | 0));
  // Stored boundaries only. The open segment is recomputed, not kept.
  // segment 1 stores every layer. segment = layers stores the single input.
  const memory = Math.ceil(layers / s);
  const extraPerLayer = (s - 1) / 2;
  return { segment: s, memory, extraPerLayer, objective: memory + extraPerLayer };
}
