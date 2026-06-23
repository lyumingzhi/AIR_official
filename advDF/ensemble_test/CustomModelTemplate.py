"""Template for adding a new face-swapping target model to AIR.

Copy this file to `YourModel.py`, implement the TODOs, and register it in
`import_test_1000.py::Tester.__init__` under a new `--source_model` value.
"""
from __future__ import annotations

import torch


class CustomModelTemplate(torch.nn.Module):
    """Minimal adapter contract expected by AIR attacks.

    Inputs to `forward` are CUDA tensors in [0, 1] with shape [B, 3, H, W].
    Return a swapped image tensor. `depreprocess` must return [0, 1] tensors.
    """

    def __init__(self, opt=None, device: str = "cuda") -> None:
        super().__init__()
        self.opt = opt
        self.device = device
        # TODO: load your third-party model and checkpoints here.
        # Keep large checkpoints outside this repository.

    def forward(self, img_s: torch.Tensor, img_t: torch.Tensor) -> torch.Tensor:
        assert torch.is_tensor(img_s) and img_s.dim() == 4 and img_s.size(1) == 3
        assert torch.is_tensor(img_t) and img_t.dim() == 4 and img_t.size(1) == 3
        # TODO: run your face-swapping model.
        # `img_s` provides identity/source; `img_t` provides target attributes.
        raise NotImplementedError("Implement the face-swapping forward pass")

    def depreprocess(self, output: torch.Tensor) -> torch.Tensor:
        # TODO: map your model output back to [0, 1] tensor space if needed.
        return output

    def get_inter_feats(self, img_s: torch.Tensor, img_t: torch.Tensor):
        # Optional: implement only if using feature/style attacks.
        raise NotImplementedError("Intermediate features are optional")
