# Diagnosis and v11 patch notes

The probes show that the v10 rescue target was nearly unreachable:

* `bit_labels` were the failed-BP residual mask.
* The sampled residual weight was about 88 bits on average, with median around 97.
* The v10 `standard_reachable` and `expanded_reachable` labels were only about 2.6% and 3.3%.
* The oracle table showed only about 3.4% reachability at the actual expanded setting `P=32,W=14`.

This means the large neural network mostly learned that the configured search cannot rescue the state.
The deeper issue is the target basis. GRAND is a noise-guessing decoder; it should primarily guess
the channel noise/correction relative to the channel hard decision. The v10 package instead guessed
the residual between a failed iterative LDPC decoder's final hard decision and the true codeword.
After BP has fallen into a trapping/pseudocodeword state, that residual is often high-weight and
spread across the Tanner graph.

v11 changes the basis:

```text
old label/search mask: BP_final_hard XOR true_codeword
new label/search mask: GRAND_base_hard XOR true_codeword
```

where `GRAND_base_hard` is channel hard on transmitted positions and BP posterior hard on punctured
positions. Candidate masks are also applied to this base during evaluation.

Other fixes:

* Corrects the AI rank-prior indexing bug.
* Performs random nonzero syndrome validation for Sionna internal 5G codeword reconstruction.
* Fills punctured systematic positions in `encode_internal()` from the input message.
* Adds Tanner-syndrome OSD repair by solving `H[:, support] e = syndrome(base)` on low-cost supports.
* Disables aggressive AI skip gating by default after BP failure.
