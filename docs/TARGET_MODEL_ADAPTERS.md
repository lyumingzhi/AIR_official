# Target Model Adapters

AIR treats a face-swapping model as a small PyTorch adapter. Keep third-party face-swapping repositories and checkpoints outside this code tree, then expose them through an adapter under `advDF/ensemble_test/`.

## Required Interface

An adapter must be a `torch.nn.Module` with:

```python
class YourModel(torch.nn.Module):
    def __init__(self, opt=None, device="cuda"):
        ...

    def forward(self, img_s, img_t):
        ...

    def depreprocess(self, output):
        ...
```

Inputs are normalized tensors in `[0, 1]` with shape `[B, 3, H, W]` on CUDA. `img_s` is the identity/source image and `img_t` is the attribute/target image. `forward()` should return the swapped image tensor. `depreprocess()` must convert the model output back to `[0, 1]` tensor space so the FR surrogate loss can compare outputs consistently.

Optional, for feature/style attacks:

```python
def get_inter_feats(self, img_s, img_t):
    ...
```

## Minimal Adapter Template

Copy `advDF/ensemble_test/CustomModelTemplate.py` to a new file, import your model inside it, and implement the three required methods. Do not vendor large checkpoints into AIR; document external paths in `docs/ARTIFACTS.md` or use symlinks for local experiments.

## Registering A New Model

The current main driver is `advDF/ensemble_test/import_test_1000.py`. Add a branch in `Tester.__init__` for your new `--source_model` value and add the adapter instance to `self.model`. For example:

```python
if model == "custom":
    from advDF.ensemble_test.CustomModel import CustomModel
    self.custom = CustomModel(opt)
    self.model = {"custom": self.custom}
```

For transfer experiments, you can also set `self.model` to multiple target adapters, e.g. `{"simswap": self.simswap, "custom": self.custom}`.

## Validation Checklist

Before claiming support for a new model:

```bash
/home1/mingzhi/anaconda3/envs/py310/bin/python main.py --help
/home1/mingzhi/anaconda3/envs/py310/bin/python tools/smoke_import.py
/home1/mingzhi/anaconda3/envs/py310/bin/python main.py --source_model custom --dir /path/to/images --pair_start 0 --pair_end 1 --dry_run --fail_on_missing_pairs
```

Then run a one-pair real smoke experiment and keep the resulting `run_manifest.json` as evidence. Full paper claims require completed shards plus `tools/audit_experiment.py --require_complete`.

## Current Built-in Scope

The current main driver directly supports SimSwap, FaceShifter, and MegaGAN transfer combinations. AgileGAN and InfoSwap wrappers exist as legacy/local starting points but are not registered as fully validated main-driver targets. DiffFace and DiffSwap are not wired in this cleaned release.
