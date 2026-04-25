# Hybrid BP + PUSCH-aligned CRC-aware AI/Tanner-GRAND evaluation

| profile | snr_db | decoder | BLER | avg_latency_ms | avg_queries | rescue_rate | crc_fail_rate | avg_crc_valid_candidates | avg_parity_valid_candidates |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | bp_nms_20 | 1 | 68.65 | 0 | 0 | 1 | 0 | 0 |
| A | 0 | bp_nms_50 | 0.99 | 170.4 | 0 | 0 | 0.99 | 0 | 0 |
| A | 0 | hybrid_bp_nsg | 1 | 919.6 | 2400 | 0 | 1 | 0 | 0 |
| A | 2 | bp_nms_20 | 0.220974 | 47.01 | 0 | 0 | 0.221 | 0 | 0 |
| A | 2 | bp_nms_50 | 0.093633 | 59.39 | 0 | 0 | 0.09363 | 0 | 0 |
| A | 2 | hybrid_bp_nsg | 0.17603 | 216.2 | 476.4 | 0.04494 | 0.176 | 0.824 | 0.824 |
| A | 4 | bp_nms_20 | 0 | 18.34 | 0 | 0 | 0 | 0 | 0 |
| A | 4 | bp_nms_50 | 0 | 18.21 | 0 | 0 | 0 | 0 | 0 |
| A | 4 | hybrid_bp_nsg | 0 | 18.26 | 0 | 0 | 0 | 1 | 1 |
