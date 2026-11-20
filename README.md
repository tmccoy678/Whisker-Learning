# Gray Matter + Whiskers

Can a cell-free conductive medium learn which optical-whisker actions carry an
electrical signal with less local stress?

`run.m` is one normalized MATLAB screen:

- `70 × 46` conductive lattice;
- conductivity contrast `1 : 10^-6`;
- `40 Hz`, `0.50–1.00 V` heartbeat;
- upper, balanced, lower, or dark whisker action;
- consequence-driven action values;
- one lesion at trial `70`;
- `200` total trials.

Run with MATLAB R2026a:

```matlab
run("run.m")
```

[`learning.gif`](learning.gif) is the recorded screening animation. This is a
computational hypothesis, not wet-lab evidence or a biological-neural claim.

MIT licensed.

## Three retained benchmark findings

Status: **sidebared after the first public benchmark screen**.

1. The medium representation solved XOR at `100%` accuracy; the frozen linear
   SVM control scored `50%`.
2. Across five fixed seeds and five-fold held-out Iris testing, the medium
   averaged `96.1%`; the raw linear control averaged `95.3%`.
3. On Wine, a human-standard raw-input skip connection raised the medium from
   `93.7%` to `95.8%` and retained `92.1%` after an untrained lesion.

These are normalized computational screens, not wet-lab performance claims.
