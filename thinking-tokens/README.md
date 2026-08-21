# thinking-tokens -- the reasoning trace is a decoder setting

> **▶ [Open this demo](index.html)** · [all demos →](../index.html) · opens with no download; the optional real model needs an http server: `python3 -m http.server 8155 `

Interactive page: **there is no reasoning ARCHITECTURE**. A reasoning model
emits its chain of thought between two *literal tokens*, `<think>`
`</think>`, and everything between them is ordinary autoregressive generation --
the same forward pass, the same softmax, the same sampler as any other token.
The trace exists because of **training**, not because of a new block: a **format
reward** (put the thinking inside the tags) beside an **accuracy reward**
([DeepSeek-R1, arXiv 2501.12948](https://arxiv.org/abs/2501.12948)).

Because the boundary is just a token, **test-time compute is controllable from
the decoder**, with no retraining ([s1 / budget forcing, arXiv
2501.19393](https://arxiv.org/abs/2501.19393); reference implementation:
[simplescaling/s1](https://github.com/simplescaling/s1)):

| Lever | What the decoder does | What you see |
|---|---|---|
| 🟢 suppress + inject | clamp the boundary token's logit toward −∞ and append `Wait` | generation carries on, and often **self-corrects** |
| 🟡 force early | emit the boundary token at a token budget | the model must **answer now** from a truncated thought |

Both act on **one logit and one comparison per step**. Nothing else changes.

## Honesty: which option this page took

The collection can run a real GPT-2 in the browser, and this page takes
**option (a) -- the real runtime, on a boundary token GPT-2 actually has.**

**GPT-2 is not a reasoning model.** It was never trained with a format reward
it has **no `<think>` / `</think>` tokens**, so the real mode does not pretend to
show a reasoning trace. What it shows is the **decoder mechanism itself**:
pick a boundary token GPT-2 really has (`"."`, `"\n"`, or `<|endoftext|>`),
clamp *that* token's real logit on real logits, inject a real ` Wait`, and the
real continuation changes under your hand. That is the same clamp and the same
injection a reasoning model is *trained* to make useful -- only the boundary
token differs. Reasoning models are trained to use this exact lever with a
`<think>` boundary; the page says so on-canvas, in both modes.

The default **synthetic stand-in** (no download) shows what the lever looks like
on a model that *was* trained for it: a scripted trace of that shape, labelled
as a stand-in in the banner.

**The real model never downloads on open.** The page opens synthetic and fetches
weights only on the explicit, size-labelled **"load real GPT-2 (~548 MB)"**
button (or `?real=1` / `?autoload=1` for tooling).

## The trade, and why it is not a straight line

- **Cost is exactly linear.** Every forced token is a token generated and a step
 of latency. The dashed line in the right panel is a straight line by
 construction.
- **Benefit turns over.** Past the natural stopping point you buy a redundant
 confirmation, then doubt with no evidence, then a repetition loop -- and the
 answer can end up **worse** than the one you already had. The stand-in walks
 exactly that arc: `$6` (a slip) → `$8` (self-corrected) → `$8` (re-checked,
 nothing gained) → `$12` (invented discount) → `$24` (degenerate).

⚠️ **The accuracy curve is SCHEMATIC** -- it draws the reported *shape* (rise,
plateau, turnover), not measured numbers, and is labelled as such on the canvas.
No measurement is claimed anywhere on this page.

The bottom panel folds in the **trained** form of the same lever: instead of
clamping at decode time, train the length in with a **length penalty tied to
difficulty** -- `reward = accuracy − λ(difficulty) · length`, so easy problems
get a short leash and hard ones a long one
([arXiv 2506.05256](https://arxiv.org/abs/2506.05256)).

## Interactions

| | Interaction |
|---|---|
| 1 | **Transport** -- step / play / scrub the generation, autoplay + loop; `?step=N` |
| 2 | **Direct manipulation** -- drag the stop-token logit bar to clamp it (down to −∞), and watch the trace extend under your hand |
| 3 | **Inject a continuation** -- one click suppresses the boundary the model just reached and appends `Wait` there |
| 4 | **Hover any token** -- the candidate logits, the bias arithmetic, and `p(stop) = exp(l_stop) / Σ exp(l_i)` for that step |
| 5 | **Forced length** -- a hard token budget (stepper, or click/drag in the curve panel) that emits the boundary token mid-thought |
| 6 | **A/B compare** -- bias 0 against bias −∞, same seed, side by side |
| 7 | **Challenges** (`?ch=N`) -- self-correct; answer early; push past the turnover into a worse answer |

## URL hooks (one URL replays one run)

`?step=N` · `?bias=` · `?inj=` · `?force=` · `?seed=` · `?stopTok=period|newline|eot`
· `?hover=x,y` · `?play=1` · `?compare=1` · `?prompt=` (real mode) ·
`?real=1` / `?autoload=1` (opt in to the download). Everything is seeded
deterministic, so a link reproduces the exact frame.

## Render tier

T1 (Canvas2D). The real mode's "compute tier" is the GPT-2 forward in JS, one
full forward per generated token (no KV cache), reusing the verified forward in
[`../real-attention/gpt2.js`](../real-attention/gpt2.js).

## Wiring

`layout.mount` + `controls` (`bias` / `inj` / `force` / `seed` / `stopTok`,
two action buttons, the load button) + `controls.transport({loop:true})`,
`onPointer` for the two drag targets, `setTip` for hover, `probe` for the
challenge checks. Source: [`page.js`](page.js).
