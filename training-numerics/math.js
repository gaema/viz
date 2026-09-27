// FP8 E4M3 training range. A loss scale lifts gradients into that range.
// Too small and they flush to zero. Too large and they overflow. The master
// weights that the optimizer actually steps stay in fp32, which is the
// memory price of training in a format this narrow.

export const FP8_E4M3_MAX = 448;
export const FP8_E4M3_MIN_NORMAL = 2 ** -6;

export function scaledFate(grad, scale) {
  const mag = Math.abs(grad * scale);
  if (mag > FP8_E4M3_MAX) return 'overflow';
  if (mag < FP8_E4M3_MIN_NORMAL) return 'flush';
  return 'kept';
}

export function masterBytes(params) { return params * 4; }
export function fp8Bytes(params) { return params; }
