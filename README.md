# Fine tuning lib

Implements some parameter-efficient fine-tuning techniques for neural networks in PyTorch.

Started out as a refactoring attempt for [PEFT](https://github.com/huggingface/peft), now lives in this cave.

## Initializing from a config file

This is similar to how PEFT does it.

```python
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

X, y = self.get_data()
# only train the LoRA weights for layer 'lin0'
config = LoraConfig(target_modules="lin0"))
model = AdapterWrapper.from_config(MLP(), config)
# use whatever training method or custom train function you like
train(model, X, y)
```

Train an additional layer, but without LoRA

```python
# 'lin1' is trained normally
config = LoraConfig(target_modules="lin0"), modules_to_save=["lin1"])
model = AdapterWrapper.from_config(MLP(), config)
train(model, X, y)
```

## Mixing adapters

These things are not possible in PEFT.

### Mixing LoRA layers with different settings

```python
model = AdapterWrapper(MLP())
model.add_adapter()

# pass the uninitialized layer to add_adapter
lin0_lora = partial(LinearLoraLayer, r=4)
model.add_adapter_layer("lin0", lin0_lora)
lin1_lora = partial(LinearLoraLayer, r=16)
model.add_adapter_layer("lin1", lin1_lora)
```

### Mixing different types of adapters

For instance, mix LoRA and IA³:

```python
model = AdapterWrapper(MLP())
model.add_adapter()

lin_lora = partial(LinearLoraLayer, r=4)
model.add_adapter_layer("lin0", lin_lora)
lin_ia3 = LinearIA3Layer
model.add_adapter_layer("lin1", lin_ia3)
```

### Custom adapters

```python
class MyLinearLayer(AdapterLayer):
    ...
    
model = AdapterWrapper(MLP())
model.add_adapter()
model.add_adapter_layer("lin0", MyLinearLayer)
```

## Utilities

This is the same as in PEFT:

```python
# create a model with the "default" adapter
config = ...
model = AdapterWrapper.from_config(MLP(), config, adapter_name="default")

# create a new adapter
model.add_adapter("other-adapter")

# add new adapter layer
model.add_adapter_layer("lin1", LinearLoraLayer)

# delete adapter layer
model.delete_adapter_layer("lin1")

# switch to a different adapter
model.set_adapter("default")

# merging
model.merge_adapter()

# undo the merge
model.unmerge_adapter()

# return the base mode
base_model = model.unload()

# merge adapter into base model and return it
base_model_merged = model.merge_and_unload()
```

## Status

Not seriously maintained, don't use this for prod.
