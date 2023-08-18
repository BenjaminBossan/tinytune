from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
from torch import nn


@dataclass
class AdapterConfig:
    target_modules: str | list[str] | None = field(
        default=None,
        metadata={"help": "List of modules to apply the adapter on. TODO: expand"},
    )

    mimic_layers: list[str] = field(
        default_factory=list,
        metadata={
            "help": ("List of modules to replace by a copy that is trained instead")
        },
    )


class AdapterLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.merged: bool = False
        self._identity = nn.Identity()

    def reset_params(self, prefix: str, model: nn.Module) -> None:
        raise NotImplementedError

    def reset_requires_grad(self, model: nn.Module) -> None:
        raise NotImplementedError

    def merge(self, module: nn.Module) -> None:
        # should be idempotent
        raise NotImplementedError

    def unmerge(self, module: nn.Module) -> None:
        # should be idempotent
        raise NotImplementedError

    def pre_forward(self, module, args, kwargs):
        return args, kwargs

    def forward(self, *args, **kwargs) -> Any:
        # TODO maybe we don't need identity, as this should never be called?
        return self._identity(*args, **kwargs)

    def post_forward(self, module, args, kwargs, output):
        return output

    def named_adapter_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        # modify this if this module has parameters that should _not_ be stored
        # as adapter parameters
        yield from self.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def adapter_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_adapter_parameters(recurse=recurse):
            yield parameter


class Adapter(AdapterLayer):
    """Container for multiple adapter layers.

    For most adaptation methods, we expect that more than a single layer is
    adapted. This class allows to handle multiple adapter layers as if they
    were a single layer.
    """

    def __init__(self, adapter_name: str) -> None:
        super().__init__()
        self.adapter_name = adapter_name
        self.layers: dict[str, AdapterLayer] = nn.ModuleDict()  # type: ignore
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def reset_requires_grad(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        for name, layer in self.layers.items():
            target = getattr(model, name)
            layer.reset_requires_grad(target)

    def merge(self, model: nn.Module) -> None:
        for name, layer in self.layers.items():
            target = getattr(model, name)
            layer.merge(target)

    def unmerge(self, model: nn.Module) -> None:
        for name, layer in self.layers.items():
            target = getattr(model, name)
            layer.unmerge(target)

    def deactivate(self, model: nn.Module) -> None:
        for name, layer in self.layers.items():
            layer.unmerge(getattr(model, name))

        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def activate(self, model: nn.Module) -> None:
        self.deactivate(model)
        for name, layer in self.layers.items():
            target = getattr(model, name)
            handle_pre = target.register_forward_pre_hook(
                layer.pre_forward, with_kwargs=True
            )
            handle_post = target.register_forward_hook(
                layer.post_forward, with_kwargs=True
            )
            self.handles.extend([handle_pre, handle_post])

    def add_layer(self, layer_name: str, model: nn.Module, layer: AdapterLayer) -> None:
        target = getattr(model, layer_name)
        layer.reset_params(f"{self.adapter_name}_{layer_name}", target)
        self.layers[layer_name] = layer

    def named_adapter_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        adapter_names: None | str | Sequence[str] = None,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        if adapter_names is None:
            adapter_names = list(self.layers.keys())
        elif isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            adapter = self.layers[adapter_name]
            yield from adapter.named_adapter_parameters(
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
            )

    def adapter_parameters(
        self,
        recurse: bool = True,
        adapter_names: None | str | Sequence[str] = None,
    ) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_adapter_parameters(
            recurse=recurse, adapter_names=adapter_names
        ):
            yield parameter


class MimicLayer(AdapterLayer):
    def __init__(self):
        super().__init__()

    def reset_params(self, prefix: str, target: nn.Module) -> None:
        self.mimic = copy.deepcopy(target)

    def reset_requires_grad(self, target: nn.Module) -> None:
        self.mimic.requires_grad_(True)

    def pre_forward(self, target, args, kwargs):
        if self.merged:
            return args, kwargs

        self._args, self._kwargs = args, kwargs
        return args, kwargs

    def post_forward(self, target, args, kwargs, output):
        if self.merged:
            return output

        mimic_output = self.mimic(*self._args, **self._kwargs)
        return mimic_output

    def _swap_parameters(self, module0: nn.Module, module1: nn.Module) -> None:
        # swap out parameters of module0 and module1
        for name, p0 in module0.named_parameters():
            p1 = getattr(module1, name)
            p0.data, p1.data = p1.data, p0.data

    def merge(self, target: nn.Module) -> None:
        if self.merged:
            return

        self._swap_parameters(self.mimic, target)
        self.merged = True

    def unmerge(self, target: nn.Module) -> None:
        if not self.merged:
            return

        self._swap_parameters(self.mimic, target)
        self.merged = False
