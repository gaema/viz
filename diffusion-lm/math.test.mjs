import { scoreMasked, diffusionStep, maskSentence, diffusionAt, DEMO_TOKENS } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(step) {
  const masked = String(step.masked);
  const sub = step.masked === 0 ? '0' : '1';
  ok(String(step.next) === String(Number(masked) - Number(sub)), 'next count');
  ok(maskSentence(step) === `${masked} - ${sub} = ${step.next}`, 'mask sentence');
  ok(step.kv == null, 'no causal KV on the step');
  ok(step.mask.filter(Boolean).length === step.next, 'the returned mask has the next count');
}

const tokens = DEMO_TOKENS.slice();
const mask = tokens.map(() => true);
const base = scoreMasked(tokens, mask);
const flipped = tokens.slice();
flipped[3] = 9;
const later = scoreMasked(flipped, mask);
ok(base[0] !== later[0], 'a later position changes the score of an earlier masked index');

const step = diffusionStep(tokens, mask);
identity(step);
ok(step.masked === 4 && step.next === 3, 'a full mask falls by one');
ok(step.unmasked >= 0 && step.mask[step.unmasked] === false, 'one position is unmasked');

const follow = diffusionStep(step.tokens, step.mask);
identity(follow);
ok(follow.masked === step.next, 'the next step consumes the updated mask and no KV argument');

const empty = diffusionStep([1, 2, 3, 4], [false, false, false, false]);
identity(empty);
ok(empty.masked === 0 && empty.next === 0 && empty.unmasked === -1, 'an empty mask stays empty');

const at0 = diffusionAt(0);
identity(at0.pending);
ok(at0.pending.masked === 4, 'before any step the mask is full');
const atEnd = diffusionAt(8);
identity(atEnd.pending);
ok(atEnd.pending.masked === 0, 'past the last mask the pending step changes nothing');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS diffusion-lm');
