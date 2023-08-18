from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .base import AdapterConfig, AdapterLayer


@dataclass
class LoraConfig(AdapterConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})

    def __post_init__(self) -> None:
        if isinstance(self.target_modules, list):
            assert len(self.target_modules) >= 1
        assert self.r > 0


class LoraLayer(AdapterLayer):
    def __init__(self, r: int = 8) -> None:
        if r <= 0:
            raise ValueError("TODO")

        self.r = r
        super().__init__()

    @classmethod
    def from_config(cls, config: AdapterConfig) -> LoraLayer:
        r = getattr(config, "r", None)
        assert isinstance(r, int)
        return cls(r=r)

    def reset_requires_grad(self, target: nn.Module) -> None:
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def post_forward(self, target, args, kwargs, output):
        if self.merged:
            return output

        (X,) = args
        lora_output = self.lora_B(self.lora_A(X))
        return output + lora_output

    def merge(self, target: nn.Module) -> None:
        if self.merged:
            return

        delta_weight = self.get_delta_weight()
        target.weight.data += delta_weight
        self.merged = True

    def unmerge(self, target: nn.Module) -> None:
        if not self.merged:
            return

        delta_weight = self.get_delta_weight()
        target.weight.data -= delta_weight
        self.merged = False

    def get_delta_weight(self) -> torch.Tensor:
        raise NotImplementedError


class LinearLoraLayer(LoraLayer):
    def reset_params(self, prefix: str, target: nn.Module) -> None:
        in_features = getattr(target, "in_features", None)
        out_features = getattr(target, "out_features", None)
        assert isinstance(in_features, int)
        assert isinstance(out_features, int)

        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        return self.lora_B.weight @ self.lora_A.weight  # type: ignore


class EmbeddingLoraLayer(LoraLayer):
    def reset_params(self, prefix: str, target: nn.Module) -> None:
        assert isinstance(target, nn.Embedding)

        self.lora_A = nn.Embedding(
            target.num_embeddings,
            self.r,
            padding_idx=target.padding_idx,
            max_norm=target.max_norm,
            norm_type=target.norm_type,
            scale_grad_by_freq=target.scale_grad_by_freq,
            sparse=target.sparse,
            device=target.weight.device,
            dtype=target.weight.dtype,
        )
        self.lora_B = nn.Linear(self.r, target.embedding_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        return self.lora_A.weight @ self.lora_B.weight.T
