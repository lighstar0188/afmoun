# Models Folder

Put local Hugging Face model/config/tokenizer folders here.

Example:

```text
models/
  Llama-3.2-1B/
    config.json
    tokenizer.json
    ...
```

For from-scratch pretraining, the training script loads the model config and
initializes random weights:

```bash
python scripts/train_causal_lm.py --model-dir models/Llama-3.2-1B --init-from-config
```

No model weights are committed to this release folder.
