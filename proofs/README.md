# proofs

Machine-checked versions of every formal result in the paper, in Lean 4 with Mathlib (v4.15.0).

```bash
lake exe cache get      # prebuilt Mathlib, a few GB
lake build              # builds WamProofs and PaperResults
lake env lean AxiomCheck.lean
```

`PaperResults.lean` states each proposition and the corollary of the paper and proves it, using lemmas
from `WamProofs.lean`. `AxiomCheck.lean` prints the axioms every theorem depends on. Each one reports
only Lean's three standard axioms, `propext`, `Classical.choice` and `Quot.sound`. There is no `sorry`.

| result in the paper | Lean theorems |
|---|---|
| Proposition 1 | `response_transformed`, `prop1_size_probe_invariant`, `prop1_reflection_direction` |
| Proposition 2 | `task1_ratio_linear` |
| Proposition 3 | `prop3_identity`, `prop3_scaled`, `prop3_large_response` |
| Proposition 4 | `offSupport_value`, `offSupport_risk`, `prop4_response`, `offSupport_deriv`, `offSupport_unbounded` |
| Corollary 5 | `cor5_data_grounded`, `cor5_direction_limits` |
| Proposition 6 | `prop6_scaled_swap`, `prop6_i`, `prop6_ii`, `prop6_iii`, `prop6_example` |
| Proposition 7 | `prop7_item`, `prop7_mean`, `prop7_ordering`, `prop7_zero_example` |
| Proposition 8 | `prop8_affine_step`, `prop8`, `prop8_nonlinear` |
| Appendix D | `appD_reflection_cov`, `appD_rotation_det`, `appD_consistency` |
