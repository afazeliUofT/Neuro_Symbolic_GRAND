# Baseline-design references for the hybrid BP + AI-guided GRAND rescue package

This package is designed around a **fixed standard Sionna 5G NR LDPC code** and compares:

- standalone BP / SPA and NMS baselines,
- classical post-processing via weighted bit flipping,
- BP followed by reference-inspired GRAND rescue variants,
- the proposed hybrid BP + AI-guided GRAND rescue receiver.

The benchmark philosophy is to keep the **same encoder, same channel realizations, and same LLRs** across decoders, so the comparison isolates the **receiver/decoder design**.

## References used to design the baselines

1. **Sionna / 5G NR LDPC support**
   - NVIDIA Sionna official documentation and module index.
   - Main idea used here: instantiate the code from Sionna's built-in 5G NR LDPC support rather than a custom ad hoc short code.

2. **ORBGRAND / ordered-reliability GRAND**
   - K. R. Duffy, W. An, and M. Médard, "Ordered Reliability Bits Guessing Random Additive Noise Decoding," 2022.
   - Main idea used here: rank candidate flips by reliability order and search likely error patterns first.

3. **Segmented GRAND / segmented ORBGRAND**
   - M. Rowshan et al., segmented GRAND / ORBGRAND complexity-reduction line.
   - Main idea used here: partition the candidate space and prioritize syndrome-consistent/localized sub-patterns.

4. **Fine-tuned ORBGRAND with a few soft values**
   - L. Wan, H. Yin, and W. Zhang, "Fine-tuning ORBGRAND with Very Few Channel Soft Values," 2025.
   - Main idea used here: lightly refine ORB-style ordering using exact soft-value information.

5. **LDPC trapping sets / error-floor behavior under iterative decoding**
   - T. Richardson, "Error floors of LDPC codes," 2003.
   - Main idea used here: iterative BP failures are often concentrated in structured residual configurations / trapping sets.

6. **Neural post-decoders specialized for failed LDPC words**
   - Recent neural-post-decoding / error-floor mitigation literature.
   - Main idea used here: train a second-stage network specialized on the subset of words that survive the first decoder.

## Important note

The GRAND-family rescue baselines in this package are **reference-inspired software baselines**, not official third-party author implementations. They are intended to provide a **fair same-code same-channel benchmark** inside this scaffold.
