import copy

import pytest
import torch
from torch import nn

from .ia3 import IA3Config, LinearIA3Layer
from .lora import LoraConfig, LinearLoraLayer
from .wrapper import AdapterWrapper


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 300)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(300, 1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        return X


class EmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 300)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(300, 1)

    def forward(self, X):
        X = self.emb(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X).squeeze(-1)
        return X


TEST_CASES = [
    (MLP, LoraConfig(target_modules="lin0")),
    (MLP, LoraConfig(target_modules="lin1")),
    (MLP, LoraConfig(target_modules=["lin0"])),
    (MLP, LoraConfig(target_modules=["lin0", "lin1"])),
    (MLP, LoraConfig(target_modules=["lin0"], mimic_layers=["lin1"])),
    (EmbModel, LoraConfig(target_modules="emb")),
    (EmbModel, LoraConfig(target_modules="lin1")),
    (EmbModel, LoraConfig(target_modules=["emb"])),
    (EmbModel, LoraConfig(target_modules=["emb", "lin1"])),
    (EmbModel, LoraConfig(target_modules=["emb"], mimic_layers=["lin1"])),
    (MLP, IA3Config(target_modules="lin0")),
    (MLP, IA3Config(target_modules="lin1")),
    (MLP, IA3Config(target_modules=["lin0"])),
    (MLP, IA3Config(target_modules=["lin0", "lin1"])),
    (MLP, IA3Config(target_modules=["lin0"], mimic_layers=["lin1"])),
]


def get_prefix_from_config(config):
    # used to identify if a parameter is an adapter parameter
    prefix = {
        LoraConfig: "lora",
        IA3Config: "ia3",
    }[type(config)]
    return prefix


class TestAltLora:
    device = "cpu"

    @pytest.fixture(autouse=True, params=["cpu", "cuda:0"], scope="class")
    def set_device(self, request):
        if request.param == "cuda:0":
            if not torch.cuda.is_available():
                pytest.skip(reason="device is not CUDA-capble")
        self.device = request.param

    def get_data(self, model_cls):
        if model_cls == MLP:
            X = torch.rand(9, 10)
            y = torch.rand(9, 1)
        elif model_cls == EmbModel:
            X = torch.randint(0, 10, (9, 1))
            y = torch.rand(9, 1)
        X, y = X.to(self.device), y.to(self.device)
        return X, y

    def get_tt_model(self, model_cls, config, **kwargs):
        torch.manual_seed(0)
        model = model_cls()
        tt_model = AdapterWrapper.from_config(model, config, **kwargs)
        tt_model = tt_model.to(self.device).eval()
        return tt_model

    def fit(self, model, X, y, epochs=3):
        torch.manual_seed(0)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # train at least 3 steps for all parameters to be updated (probably this
        # is required because of symmetry breaking of some layers that are
        # initialized with constants)
        for _ in range(10):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_applying_lora_does_not_change_output(self, model_cls, config):
        X, _ = self.get_data(model_cls)
        model = model_cls().to(self.device).eval()
        output_base = model(X)

        tt_model = AdapterWrapper.from_config(model, config)
        output_tt = tt_model(X)

        torch.testing.assert_close(output_base, output_tt)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_training_changes_output(self, model_cls, config):
        X, y = self.get_data(model_cls)
        tt_model = self.get_tt_model(model_cls, config)

        output_base = tt_model.model(X)
        output_before = tt_model(X)
        torch.testing.assert_close(output_base, output_before)

        self.fit(tt_model, X, y)
        output_after = tt_model(X)
        assert not torch.allclose(output_before, output_after)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_only_adapter_layers_are_updated(self, model_cls, config):
        X, y = self.get_data(model_cls)
        tt_model = self.get_tt_model(model_cls, config)
        params_before = {k: v.clone().detach() for k, v in tt_model.named_parameters()}
        self.fit(tt_model, X, y)
        params_after = dict(tt_model.named_parameters())

        assert params_before.keys() == params_after.keys()

        prefix = get_prefix_from_config(config)
        for key, param_before in params_before.items():
            param_after = params_after[key]
            is_adapter_param = prefix in key
            is_mimicked = "mimic." in key
            if is_adapter_param or is_mimicked:
                assert not torch.allclose(param_before, param_after)
                assert param_after.requires_grad
            else:
                torch.testing.assert_close(param_before, param_after)
                assert not param_after.requires_grad

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_swap_base_model(self, model_cls, config):
        X, y = self.get_data(model_cls)

        # initialize with model 0
        torch.manual_seed(0)
        model0 = model_cls()
        tt_model = AdapterWrapper.from_config(model0, config)
        tt_model = tt_model.to(self.device).eval()
        output_base0 = tt_model.model(X)

        # initialize with model 1 which is identical to model 0
        torch.manual_seed(0)
        model1 = model_cls()
        tt_model.swap_base_model(model1)
        tt_model = tt_model.to(self.device).eval()
        output_base1 = tt_model.model(X)

        torch.testing.assert_close(output_base0, output_base1)

        # initialize with model 2 which is different from model 0
        torch.manual_seed(1)  # different seed, hence different model
        model2 = model_cls()
        tt_model.swap_base_model(model2)
        tt_model = tt_model.to(self.device).eval()
        output_base2 = tt_model.model(X)

        assert not torch.allclose(output_base0, output_base2)

        # train with model 2
        self.fit(tt_model, X, y)
        output_after2 = tt_model.model(X)
        assert not torch.allclose(output_base2, output_after2)

        # swap back to model 1
        # since adapters were trained, output of model 1 should also change,
        # even though tt_model was never trained with model 1
        tt_model.swap_base_model(model1)
        output_after1 = tt_model.model(X)
        assert not torch.allclose(output_base1, output_after1)

        # swap back to model 0
        tt_model.swap_base_model(model0)
        output_after0 = tt_model.model(X)
        torch.testing.assert_close(output_after0, output_after1)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    @pytest.mark.parametrize("merge", [True, False])
    def test_unload(self, model_cls, config, merge):
        X, y = self.get_data(model_cls)
        tt_model = self.get_tt_model(model_cls, config)
        output_before = tt_model(X)
        self.fit(tt_model, X, y)

        if merge:
            tt_model.merge_adapters()

        model = tt_model.unload()
        output_unloaded = model(X)
        torch.testing.assert_close(output_before, output_unloaded)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    @pytest.mark.parametrize("merge", [True, False])
    def test_merge_and_unload(self, model_cls, config, merge):
        X, y = self.get_data(model_cls)
        tt_model = self.get_tt_model(model_cls, config)

        self.fit(tt_model, X, y)
        output_after = tt_model(X)

        if merge:
            tt_model.merge_adapters()
        model = tt_model.merge_and_unload()
        output_unloaded = model(X)
        torch.testing.assert_close(output_after, output_unloaded)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_merge_unmerge(self, model_cls, config):
        X, y = self.get_data(model_cls)
        tt_model = self.get_tt_model(model_cls, config)

        self.fit(tt_model, X, y)
        output_after = tt_model(X)

        tt_model.merge_adapters()
        output_merged = tt_model(X)
        torch.testing.assert_close(output_after, output_merged)

        tt_model.unmerge_adapter()
        output_unmerged = tt_model(X)
        torch.testing.assert_close(output_after, output_unmerged)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_multiple_adapters_only_active_one_is_updated(self, model_cls, config):
        X, y = self.get_data(model_cls)

        config_A = config
        config_B = copy.copy(config)
        tt_model = self.get_tt_model(model_cls, config_A, adapter_name="adapter-A")
        tt_model.add_adapter_from_config(config_B, "adapter-B")
        tt_model.set_active_adapters(["adapter-A"])

        params_A_before = {
            k: v.clone().detach()
            for k, v in tt_model.adapters["adapter-A"].named_adapter_parameters()
        }
        params_B_before = {
            k: v.clone().detach()
            for k, v in tt_model.adapters["adapter-B"].named_adapter_parameters()
        }
        self.fit(tt_model, X, y, epochs=3)
        params_A_after = dict(tt_model.adapters["adapter-A"].named_adapter_parameters())
        params_B_after = dict(tt_model.adapters["adapter-B"].named_adapter_parameters())

        prefix = get_prefix_from_config(config)
        # adapter layers for A should have changed
        for key, param_before in params_A_before.items():
            param_after = params_A_after[key]
            is_adapter_param = prefix in key
            is_mimicked = "mimic." in key
            if is_adapter_param or is_mimicked:
                assert param_after.requires_grad
                assert not torch.allclose(param_before, param_after)
            else:
                assert not param_after.requires_grad
                torch.testing.assert_close(param_before, param_after)

        # all B layers should be the same
        for key, param_before in params_B_before.items():
            param_after = params_B_after[key]
            torch.testing.assert_close(param_before, param_after)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_multiple_adapters_all_active_ones_are_updated(self, model_cls, config):
        if config.mimic_layers:
            # they probably shadow one another
            pytest.xfail("Fails when multiple mimic layers are applied")

        X, y = self.get_data(model_cls)

        config_A = config
        config_B = copy.copy(config)
        tt_model = self.get_tt_model(model_cls, config_A, adapter_name="adapter-A")
        tt_model.add_adapter_from_config(config_B, "adapter-B")
        assert tt_model.active_adapters == ["adapter-A", "adapter-B"]

        params_A_before = {
            k: v.clone().detach()
            for k, v in tt_model.adapters["adapter-A"].named_adapter_parameters()
        }
        params_B_before = {
            k: v.clone().detach()
            for k, v in tt_model.adapters["adapter-B"].named_adapter_parameters()
        }
        self.fit(tt_model, X, y, epochs=3)
        params_A_after = dict(tt_model.adapters["adapter-A"].named_adapter_parameters())
        params_B_after = dict(tt_model.adapters["adapter-B"].named_adapter_parameters())
        prefix = get_prefix_from_config(config)

        # adapter layers for A should have changed
        for key, param_before in params_A_before.items():
            param_after = params_A_after[key]
            is_adapter_param = prefix in key
            is_mimicked = "mimic." in key
            if is_adapter_param or is_mimicked:
                assert param_after.requires_grad
                assert not torch.allclose(param_before, param_after)
            else:
                assert not param_after.requires_grad
                torch.testing.assert_close(param_before, param_after)

        # adapter layers for B should have changed
        for key, param_before in params_B_before.items():
            param_after = params_B_after[key]
            is_adapter_param = prefix in key
            is_mimicked = "mimic." in key
            if is_adapter_param or is_mimicked:
                assert param_after.requires_grad
                assert not torch.allclose(param_before, param_after)
            else:
                assert not param_after.requires_grad
                torch.testing.assert_close(param_before, param_after)

    @pytest.mark.parametrize("model_cls, config", TEST_CASES)
    def test_multiple_adapters_output(self, model_cls, config):
        if config.mimic_layers:
            # they probably shadow one another
            pytest.xfail("Fails when multiple mimic layers are applied")

        X, y = self.get_data(model_cls)
        prefix = get_prefix_from_config(config)
        config_A = config

        # set config B
        config_B = copy.copy(config)
        tt_model = self.get_tt_model(model_cls, config_B, adapter_name="adapter-B")
        # check that only adapter parameters require grad
        grad_dict = {
            name: param.requires_grad for name, param in tt_model.named_parameters()
        }
        grad_dict_expected = {
            name: (prefix in name) or ("mimic." in name) for name in grad_dict
        }
        assert grad_dict == grad_dict_expected
        output_B_0 = tt_model(X)

        # set config A
        tt_model.add_adapter_from_config(config_A, "adapter-A")
        tt_model.set_active_adapters("adapter-A")
        # config_A is now the active adapter
        grad_dict = {
            name: param.requires_grad for name, param in tt_model.named_parameters()
        }
        assert grad_dict == grad_dict_expected
        output_A_0 = tt_model(X)
        # after initialization, adapter is noop, so outputs should be the same
        torch.testing.assert_close(output_A_0, output_B_0)

        # train B
        tt_model.set_active_adapters("adapter-B")
        grad_dict = {
            name: param.requires_grad for name, param in tt_model.named_parameters()
        }
        assert grad_dict == grad_dict_expected
        # different number of epochs to ensure different weights
        self.fit(tt_model, X, y, epochs=15)
        output_B_1 = tt_model(X)

        # first, ensure that the output differs after training
        assert not torch.allclose(output_B_0, output_B_1)

        # going back to A should produce the same A output as before
        tt_model.set_active_adapters("adapter-A")
        grad_dict = {
            name: param.requires_grad for name, param in tt_model.named_parameters()
        }
        assert grad_dict == grad_dict_expected
        output_A_1 = tt_model(X)
        torch.testing.assert_close(output_A_0, output_A_1)

        # going back to B should produce the same B output as before
        tt_model.set_active_adapters("adapter-B")
        grad_dict = {
            name: param.requires_grad for name, param in tt_model.named_parameters()
        }
        assert grad_dict == grad_dict_expected
        output_B_2 = tt_model(X)
        torch.testing.assert_close(output_B_1, output_B_2)

        # delete the currently active adapter
        tt_model.delete_adapter("adapter-B")
        tt_model.set_active_adapters("adapter-A")
        output_A_3 = tt_model(X)
        torch.testing.assert_close(output_A_1, output_A_3)

        # deleting B again raises a KeyError
        with pytest.raises(KeyError):
            tt_model.delete_adapter("adapter-B")
        # deleting A is possible
        tt_model.delete_adapter("adapter-A")

    def test_mixing_adaptations_parameters_works(self):
        X, y = self.get_data(MLP)
        model = MLP().to(self.device).eval()
        output_base = model(X)

        tt_model = AdapterWrapper(model)
        tt_model.add_adapter()

        lin_lora_0 = LinearLoraLayer(r=4)
        tt_model.add_adapter_layer("default", "lin0", lin_lora_0)
        lin_lora_1 = LinearLoraLayer(r=5)
        tt_model.add_adapter_layer("default", "lin1", lin_lora_1)

        output_tt = tt_model(X)
        torch.testing.assert_close(output_base, output_tt)

        self.fit(tt_model, X, y)

        output_after = tt_model(X)
        assert not torch.allclose(output_tt, output_after)

        lora_layers = [
            layer for layer in tt_model.modules() if isinstance(layer, LinearLoraLayer)
        ]
        assert len(lora_layers) == 2
        assert lora_layers[0].lora_A.out_features == 4
        assert lora_layers[0].lora_B.in_features == 4
        assert lora_layers[1].lora_A.out_features == 5
        assert lora_layers[1].lora_B.in_features == 5

    def test_mixing_adaptations_types_works(self):
        X, y = self.get_data(MLP)
        model = MLP().to(self.device).eval()
        output_base = model(X)

        tt_model = AdapterWrapper(model)
        tt_model.add_adapter()

        lin_lora = LinearLoraLayer(r=4)
        tt_model.add_adapter_layer("default", "lin0", lin_lora)
        lin_ia3 = LinearIA3Layer()
        tt_model.add_adapter_layer("default", "lin1", lin_ia3)

        output_tt = tt_model(X)
        torch.testing.assert_close(output_base, output_tt)

        self.fit(tt_model, X, y)

        output_after = tt_model(X)
        assert not torch.allclose(output_tt, output_after)

        lora_layers = [
            layer for layer in tt_model.modules() if isinstance(layer, LinearLoraLayer)
        ]
        assert len(lora_layers) == 1
        assert lora_layers[0].lora_A.out_features == 4
        assert lora_layers[0].lora_B.in_features == 4

        ia3_layers = [
            layer for layer in tt_model.modules() if isinstance(layer, LinearIA3Layer)
        ]
        assert len(ia3_layers) == 1
        assert ia3_layers[0].ia3_weight.shape == (1, 300)

    @pytest.mark.xfail(reason="Mixing adapter methods works")
    def test_nesting_adaptations_types_raises(self):
        X, y = self.get_data(MLP)
        model = MLP().to(self.device).eval()

        tt_model = AdapterWrapper(model)
        tt_model.add_adapter()

        # trying to apply IA³ to a Lora layer should raise an error
        lin_lora = LinearLoraLayer(r=4)
        tt_model.add_adapter_layer("default", "lin0", lin_lora)
        lin_ia3 = LinearIA3Layer()

        with pytest.raises(ValueError):
            tt_model.add_adapter_layer("default", "lin0", lin_ia3)

    @pytest.mark.skip(reason="TODO")
    def test_wrong_module_names_raises(self):
        pass
