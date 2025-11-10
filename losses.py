"""Loss functions used during training."""

from __future__ import annotations

from typing import Dict, Tuple

from torchvision import models

from torch_utils import ensure_torch_available, nn, torch

ensure_torch_available()

import torch.nn.functional as F


class VGGFeatureExtractor(nn.Module):
    """Pre-trained VGG19 truncated to specific layers for perceptual loss."""

    def __init__(self, layers: Tuple[str, ...]) -> None:
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.selected_layers = layers
        self.slices = nn.ModuleDict()

        layer_names = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 24,
        }

        prev_idx = 0
        for name in layers:
            idx = layer_names[name]
            self.slices[name] = nn.Sequential(*[vgg19[i] for i in range(prev_idx, idx + 1)])
            prev_idx = idx + 1

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover - heavy model
        outputs: Dict[str, torch.Tensor] = {}
        current = x
        for name, module in self.slices.items():
            current = module(current)
            outputs[name] = current
        return outputs


class CompositeLoss(nn.Module):
    """Combine reconstruction, perceptual, adversarial and edge losses."""

    def __init__(
        self,
        loss_weights: Dict[str, float],
        perceptual_layers: Tuple[str, ...],
        use_gan: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weights = loss_weights
        self.use_gan = use_gan
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        if loss_weights.get("perceptual", 0.0) > 0:
            self.perceptual_net = VGGFeatureExtractor(perceptual_layers)
        else:
            self.perceptual_net = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        discriminator_pred: torch.Tensor | None = None,
        discriminator_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_loss = torch.zeros(1, device=pred.device)

        if self.loss_weights.get("reconstruction", 0.0) > 0:
            recon = self.l1(pred, target)
            total_loss = total_loss + self.loss_weights["reconstruction"] * recon

        if self.perceptual_net is not None and self.loss_weights.get("perceptual", 0.0) > 0:
            pred_vgg = self.perceptual_net(pred)
            target_vgg = self.perceptual_net(target)
            perceptual_loss = sum(
                self.l1(pred_vgg[name], target_vgg[name]) for name in pred_vgg.keys()
            ) / len(pred_vgg)
            total_loss = total_loss + self.loss_weights["perceptual"] * perceptual_loss

        if self.loss_weights.get("edge", 0.0) > 0:
            pred_gray = torch.mean(pred, dim=1, keepdim=True)
            target_gray = torch.mean(target, dim=1, keepdim=True)
            pred_edges = self._sobel_edges(pred_gray)
            target_edges = self._sobel_edges(target_gray)
            edge_loss = self.l1(pred_edges, target_edges)
            total_loss = total_loss + self.loss_weights["edge"] * edge_loss

        if self.use_gan and self.loss_weights.get("adversarial", 0.0) > 0:
            if discriminator_pred is None or discriminator_target is None:
                raise ValueError("Adversarial loss enabled but discriminator outputs not provided")
            adv_loss = self.mse(discriminator_pred, torch.ones_like(discriminator_pred))
            adv_loss += self.mse(discriminator_target, torch.zeros_like(discriminator_target))
            total_loss = total_loss + self.loss_weights["adversarial"] * adv_loss

        return total_loss

    @staticmethod
    def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude for edge consistency."""

        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=x.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=x.device)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return magnitude


__all__ = ["CompositeLoss"]

