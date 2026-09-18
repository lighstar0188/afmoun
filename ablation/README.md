# AF-Muon reviewer ablations

This folder is intentionally isolated from the main training scripts. It reuses the
T5-small shared-vocabulary sentinel model, but exposes tied-table and 1D auxiliary
optimizer choices independently.

## Factorial tied-table vs 1D auxiliary ablation

Reviewer question: does the gain come from the finite-cap tied vocabulary table, or
from replacing all 1D auxiliary parameters?

The four arms are:

| Arm | Shared vocabulary table | 1D auxiliary parameters |
| --- | --- | --- |
| `adamw_adamw` | AdamW | AdamW |
| `cap_adamw` | finite-cap tied LMO | AdamW |
| `adamw_rms` | AdamW | RMS-LMO |
| `cap_rms` | finite-cap tied LMO | RMS-LMO |

Default cap settings are `c=3`, `s=0.5`, with `rho_output/rho_hidden=1` if
`--rho-output 50 --rho-hidden 50` is used.

Example command:

```bash
cd <repo>

AFMOUN_WORKER_PYTHON=<python> \
<python> ablation/run_tied_factorial.py \
  --gpus 0 1 2 3 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/ablation_tied_factorial_t5small_275m_seed43 \
  --seeds 43 \
  --arms adamw_adamw,cap_adamw,adamw_rms,cap_rms \
  --vocab-size 49152 \
  --train-tokens 275000000 \
  --eval-tokens 1000000 \
  --dim 512 \
  --heads 8 \
  --encoder-layers 4 \
  --decoder-layers 4 \
  --source-len 256 \
  --target-len 256 \
  --micro-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --matrix-weight-decay 0.1 \
  --tied-weight-decay 0.0 \
  --vector-weight-decay 0.0 \
  --rho-hidden 50 \
  --rho-output 50 \
  --tied-cap 3 \
  --tied-scale 0.5 \
  --eval-every-steps 2000 \
  --log-every-steps 200 \
  --diag-every-steps 2000 \
  --encoder-ablation-every-steps 2000 \
  --checkpoint-every-steps 0 \
  --bf16
```

Summarize:

```bash
<python> ablation/summarize_tied_factorial.py \
  --run-roots runs/ablation_tied_factorial_t5small_275m_seed43 \
  --csv-out outputs/ablation_tied_factorial_t5small_275m_seed43.csv
```
