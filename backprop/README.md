# backprop -- the backward pass (how a model learns)

> **▶ [Open this demo](index.html)**  ·  [all demos →](../index.html)  ·  needs an http server (ES modules): `python3 -m http.server 8099 --directory viz`

Interactive page: **backpropagation / autograd** -- the *backward* pass, the
training counterpart to the [forward-pass](../forward-pass/README.md) capstone (and
the only page about how a model *learns*, not just runs). **Anchor**: H (end to end
/ training). Family H (see `../plan/curriculum.md`).

## What it shows

A tiny computation graph -- one neuron with a squared loss:

```
m1 = w1·x1     m2 = w2·x2     z = m1 + m2 + b     a = tanh(z)     L = (a − y)²
```

runs **forward** (values flow →) to a loss `L`, then **gradients flow back** (←) by
the **chain rule**: each node takes the gradient arriving from above and multiplies
by its own local derivative, handing the result to its inputs:

```
dL/da = 2(a − y)      dL/dz = dL/da · (1 − a²)      dL/dw1 = dL/dz · x1   ...
```

Every node shows its **value** and its **gradient** (`g=`, in red); the red arrows
are the backward flow. The gradient on each weight, `dL/dw`, is exactly the **slope
of the loss** in that weight's direction.

Flip on **descend** and the page repeatedly applies `w ← w − lr·dL/dw` and you watch
`L` fall toward 0 on the **loss curve** while the output `a` converges to the target
`y` -- that loop, over billions of parameters, *is* training.

**Step** the reveal (0–4) to watch gradients propagate layer by layer; **drag** any
leaf node (inputs `x`, weights `w`, bias `b`, target `y`) to change it and see every
value and gradient update; toggle **descend** and tune the **learning rate**.

## Render tier

T1 (Canvas2D: the computation-graph nodes with value + gradient, the backward-flow
arrows, and the loss-vs-step curve).

## Wiring

`layout.mount()` + a 5-step `transport` (the backward reveal) + controls (`lr`,
`train`, `seed`) + `animate` (the descent loop + loss curve) + `onPointer`
drag-a-leaf-node + hover. `?lr` / `?train` / `?seed` / `?step` hooks. Source:
[`page.js`](page.js).
