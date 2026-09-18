# Three-Way Shared Vocabulary Sentinel

This folder is intentionally self-contained. It does not modify the frozen
release scripts or optimizer package.

Goal: test AF-Muon V2 on an encoder-decoder setting where one vocabulary table is
shared by three roles:

- encoder input lookup
- decoder input lookup
- output classifier / LM head

The first sentinel uses contiguous FineWeb token blocks. The encoder receives a
source prefix and the decoder predicts the following target segment. This keeps
the data path simple while exercising the three-way shared table.

Default optimizer protocol:

- all arms: hidden matrix WD = 0.1
- Hybrid Muon: shared vocabulary table and vector aux group use AdamW with WD = 0.01
- SCION-style Sign and AF-Muon: tied table and vector roles use WD = 0
- all arms: constant LR, FP32 parameters/state, optional BF16 autocast

Recommended dry run for the paper-scale shared-vocabulary setting:

```bash
cd ...

AFMOUN_WORKER_PYTHON=...\
... t5test/run_three_way_shared_vocab.py \
  --gpus 5 6 7 \
  --data-dir ... \
  --run-root runs/t5test_small_5xparams_3seeds_rho1 \
  --seeds 43 44 45 \
  --arms muon,sign,afmoun \
  --vocab-size 49152 \
  --train-tokens 275000000 \
  --eval-tokens 1000000 \
  --micro-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --rho-output-values 50 \
  --dry-run
```

If the dry run is clean, remove `--dry-run`.
