# V4 update notes

The v4 package is the follow-up to the committed v3 full run.

## Why v4 exists

The v3 analysis showed that:

- point-level BLER, query count, and latency were already very strong against the current baseline
- however, most remaining average query cost came from a small number of **post-search fallback** cases
- the worst branch was **expanded-overflow AI search followed by fallback**, which had low estimated fallback success relative to its cost

## Main algorithmic change

The v4 default config keeps:

- presearch hopeless-case skipping
- expanded-overflow AI search
- standard post-search fallback

but disables:

- **post-search fallback after expanded-overflow AI failure**

This is intended to cut the heaviest remaining query tail while only slightly perturbing BLER.

## Main analysis additions

The v4 report exports:

- `overview_nontrivial_by_profile_snr.csv`
- `nsgrand_action_contribution_summary.csv`
- `postsearch_outcome_summary.csv`

These make it easier to see whether the next iteration should improve:

- skip gating
- expanded AI search
- post-search fallback policy
- confidence-label design
