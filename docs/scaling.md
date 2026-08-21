# Scaling

## MINDlarge Preprocessing

The interaction preprocessing pipeline was benchmarked on both MINDsmall
and MINDlarge using 25,000-row Pandas chunks.

| Dataset   | Impressions | Candidate Interactions | Runtime | Throughput | Peak RSS |
| --------- | ----------: | ---------------------: | ------: | ---------: | -------: |
| MINDsmall |     156,965 |                  5.84M |  6.34 s |  920,980/s |   573 MB |
| MINDlarge |   2,232,748 |                 83.51M | 93.42 s |  893,859/s |   588 MB |

MINDlarge contained roughly 14× more interactions, while throughput remained
near 0.9M interactions/sec and peak process memory remained below 600 MB.

The pipeline therefore scaled approximately linearly with input size while
maintaining bounded memory usage.

PySpark was intentionally not introduced because chunked local preprocessing
processed the full 83.5M-interaction dataset in approximately 93 seconds.
Distributed processing would add operational complexity without addressing
a measured bottleneck.
