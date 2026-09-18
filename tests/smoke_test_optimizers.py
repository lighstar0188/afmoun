from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from optimizers import build_afmoun, build_hybrid_muon, build_scion_sign


class TinyTiedLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.wte.weight

    def get_input_embeddings(self):
        return self.wte

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, idx):
        x = self.wte(idx)
        x = self.proj(x)
        x = self.ln(x)
        return self.lm_head(x)


def run_one(builder, name: str) -> None:
    torch.manual_seed(123)
    model = TinyTiedLM()
    opt = builder(model)
    x = torch.randint(0, 32, (4, 12))
    y = torch.roll(x, shifts=-1, dims=1)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1))
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item(), name
    assert model.wte.weight is model.lm_head.weight, name


def main() -> None:
    run_one(build_hybrid_muon, "hybrid_muon")
    run_one(build_scion_sign, "scion_sign")
    run_one(build_afmoun, "afmoun")
    print("smoke test passed")


if __name__ == "__main__":
    main()
