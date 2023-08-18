"""Nomenclature:

In general, the whole base model is referred to as "model".

The individual sub-modules of the model that are adapted are referred to as "target"

The adapter layers that are applied to the "target" are referred to as "layer".

An "adapter" is a collection of adapter layers. A single adapter layer can
theoretically also act like an adapter.

The "wrapper" is a collection of adapters. It also has a reference to the
"model".

"""
from __future__ import annotations

from typing import Any, Iterator, Sequence

from torch import nn

from .base import AdapterConfig, Adapter, AdapterLayer, MimicLayer
from .construction import _get_adaptation_strategy, _get_selection_strategy


class AdapterWrapper(nn.Module):
    ##################
    # INITIALIZATION #
    ##################

    def __init__(self, model: nn.Module, config: AdapterConfig | None = None) -> None:
        super().__init__()
        self.model = model
        self.peft_config = config

        self.adapters: dict[str, Adapter] = nn.ModuleDict()  # type: ignore
        self.active_adapters: list[str] = []

    #####################
    # HANDLING ADAPTERS #
    #####################

    def apply_adapters(self) -> None:
        """Activates all adapters that are currently activate adapters."""
        for adapter in self.adapters.values():
            adapter.deactivate(self.model)

        for name in self.active_adapters:
            adapter = self.adapters[name]
            adapter.activate(self.model)
            adapter.reset_requires_grad(self.model)

    def add_adapter(self, adapter_name: str = "default", activate: bool = True) -> None:
        if adapter_name in self.adapters:
            raise KeyError(
                f"Adapter {adapter_name} already exists, delete it first "
                "or use a different name."
            )

        adapter = Adapter(adapter_name)
        self.adapters[adapter_name] = adapter
        if activate:
            self.active_adapters.append(adapter_name)
            self.apply_adapters()

    def delete_adapter(self, adapter_name) -> None:
        if adapter_name not in self.adapters:
            raise KeyError(f"Adapter {adapter_name} does not exist.")

        adapter = self.adapters.pop(adapter_name)
        adapter.deactivate(self.model)
        if adapter_name in self.active_adapters:
            self.active_adapters.remove(adapter_name)

    def set_active_adapters(self, adapter_names: str | Sequence[str]) -> None:
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        self.active_adapters.clear()
        self.active_adapters.extend(adapter_names)
        self.apply_adapters()

    def add_adapter_layer(
        self,
        adapter_name: str,
        layer_name: str,
        layer: AdapterLayer,
    ) -> None:
        """Add an adapter layer to the adapter with the given name."""
        if adapter_name not in self.adapters:
            raise KeyError(f"Adapter {adapter_name} does not exist.")
        if not hasattr(self.model, layer_name):
            raise AttributeError(f"Layer {layer_name} does not exist.")

        adapter = self.adapters[adapter_name]
        adapter.add_layer(layer_name, self.model, layer)
        if adapter_name in self.active_adapters:
            adapter.activate(self.model)

    def _unload_and_optionally_merge(self, merge: bool) -> nn.Module:
        for adapter_name in self.active_adapters:
            adapter = self.adapters[adapter_name]
            adapter.deactivate(self.model)
            if merge:
                adapter.merge(self.model)
            else:
                adapter.unmerge(self.model)

        return self.model

    def merge_adapters(self, adapter_names: None | str | Sequence[str] = None) -> None:
        # None means all
        if adapter_names is None:
            adapter_names = self.active_adapters
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            adapter = self.adapters[adapter_name]
            adapter.merge(self.model)

    def unmerge_adapters(self) -> None:
        for adapter in self.adapters.values():
            adapter.unmerge(self.model)

    def unload(self) -> nn.Module:
        return self._unload_and_optionally_merge(merge=False)

    def merge_and_unload(self) -> nn.Module:
        return self._unload_and_optionally_merge(merge=True)

    def swap_base_model(self, new_model: nn.Module) -> None:
        self.unmerge_adapters()
        self.unload()
        self.model = new_model
        self.apply_adapters()

    ########################
    # CREATION FROM CONFIG #
    ########################

    def add_adapter_from_config(
        self, config: AdapterConfig, adapter_name: str = "default"
    ) -> AdapterWrapper:
        self.add_adapter(adapter_name, activate=False)

        selection_strategy = _get_selection_strategy(config, self.model)
        any_match = False
        for layer_name, layer_specific_args in selection_strategy:
            any_match = True
            adaptation_strategy = _get_adaptation_strategy(config, layer_specific_args)
            new_layer = adaptation_strategy(self.model, layer_name)
            self.add_adapter_layer(adapter_name, layer_name, new_layer)

        if not any_match:
            raise ValueError("Could not find any matching layers for the given config")

        mimic_layer_names = config.mimic_layers or []
        for mimic_layer_name in mimic_layer_names:
            if not hasattr(self.model, mimic_layer_name):
                raise AttributeError(f"Layer {mimic_layer_name} does not exist.")
            self.add_adapter_layer(adapter_name, mimic_layer_name, MimicLayer())

        self.active_adapters.append(adapter_name)
        self.apply_adapters()
        return self

    @classmethod
    def from_config(
        cls, model: nn.Module, config: AdapterConfig, adapter_name: str = "default"
    ) -> AdapterWrapper:
        wrapper = cls(model, config=config)
        wrapper.add_adapter_from_config(config, adapter_name)
        return wrapper

    ##################################
    # EXPOSING METHODS OF BASE MODEL #
    ##################################

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Any:
        if not hasattr(self.model, "generate"):
            raise AttributeError("Model does not have a generate method.")
        return self.model.generate(*args, **kwargs)  # type: ignore

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        yield from self.named_adapter_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
        yield from self.model.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield from self.adapter_parameters(recurse=recurse)
        yield from self.model.parameters(recurse=recurse)

    def named_adapter_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        for name in self.active_adapters:
            adapter = self.adapters[name]
            yield from adapter.named_adapter_parameters(
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
            )

    def adapter_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_adapter_parameters(recurse=recurse):
            yield parameter
