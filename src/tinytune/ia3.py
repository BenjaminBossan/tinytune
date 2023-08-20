from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base import AdapterConfig, AdapterLayer


@dataclass
class IA3Config(AdapterConfig):
    pass


class LinearIA3Layer(AdapterLayer):
    @classmethod
    def from_config(cls, config: AdapterConfig) -> LinearIA3Layer:
        return cls()

    def reset_params(self, prefix: str, target: nn.Module) -> None:
        assert isinstance(target.weight, torch.Tensor)
        self.ia3_weight = nn.Parameter(torch.ones_like(target.weight[:1]))

    def reset_requires_grad(self, target: nn.Module) -> None:
        self.ia3_weight.requires_grad_(True)

    def pre_forward(self, target, args, kwargs):
        if self.merged:
            return args, kwargs

        (X,) = args
        X = X * self.ia3_weight
        return (X,), kwargs

    def merge(self, target: nn.Module) -> None:
        if self.merged:
            return

        target.weight.data *= self.ia3_weight
        self.merged = True

    def unmerge(self, target: nn.Module) -> None:
        if not self.merged:
            return

        target.weight.data /= self.ia3_weight
        self.merged = False
