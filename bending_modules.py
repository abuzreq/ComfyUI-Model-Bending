# Shared bending module classes used by both web UI and standalone model bending nodes.
# Kept in one place to avoid duplication between nodes.py and model_bending_nodes.py.
import torch
import torch.nn as nn
from .bendutils import operations


class ApplyToRandomSubsetModule(nn.Module):
    """Apply a sub-module to a random subset of batch/channel/spatial dimensions."""
    def __init__(self, module, percentage=0.5, seed=None, dim="batch"):
        super().__init__()
        self.module = module
        self.percentage = percentage
        self.seed = seed
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        if self.percentage == 0 or self.percentage == 1.0:
            return x

        B, C, H, W = x.shape
        out = x.clone()

        if self.dim == "batch":
            n = B
            subset_size = int(n * self.percentage)
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))[:subset_size]
            out[idx] = self.module(x[idx], *args, **kwargs)
            return out
        elif self.dim == "channel":
            n = C
            subset_size = int(n * self.percentage)
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))[:subset_size]
            out[:, idx] = self.module(x[:, idx], *args, **kwargs)
            return out
        elif self.dim == "spatial":
            num_pixels = H * W
            subset_size = int(num_pixels * self.percentage)
            flat_idx = torch.randperm(num_pixels, generator=torch.Generator().manual_seed(self.seed))[:subset_size]
            rows = flat_idx // W
            cols = flat_idx % W
            mask = torch.zeros(H, W, dtype=torch.bool, device=x.device)
            mask[rows, cols] = True
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
            transformed = self.module(x, *args, **kwargs)
            out = torch.where(mask, transformed, x)
            return out
        else:
            raise ValueError(f"Unsupported dimension mode: {self.dim}")


class BendingModule(nn.Module):
    """
    Base class for all bending operations. Ensures input shape alignment and optional
    step-based application (steps_to_bend / current_step).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        num_unsqueeze_added = 0
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            num_unsqueeze_added = 2
        if x.ndim == 3:
            x = x.unsqueeze(0)
            num_unsqueeze_added = 1
        elif x.ndim != 4:
            raise ValueError(f"Input tensor must be 3D or 4D, but got ndim={x.ndim}")

        if (hasattr(self, 'current_step') and hasattr(self, 'steps_to_bend') and self.current_step is not None and self.steps_to_bend is not None):
            if self.current_step in self.steps_to_bend:
                output = self.bend(x, *args, **kwargs)
            else:
                output = x
        else:
            output = self.bend(x, *args, **kwargs)

        for i in range(num_unsqueeze_added):
            output = output.squeeze(0)
        return output

    def bend(self, x, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the bend() method.")


class AddNoiseModule(BendingModule):
    def __init__(self, noise_std=1, seed=42):
        super().__init__()
        self.noise_std = noise_std
        self.seed = seed

    def bend(self, x, *args, **kwargs):
        noise = x.new_empty(x.shape).normal_(
            mean=0, std=self.noise_std, generator=torch.Generator(device=x.device).manual_seed(self.seed))
        return x + noise


class AddScalarModule(BendingModule):
    def __init__(self, scalar=1):
        super().__init__()
        self.scalar = scalar

    def bend(self, x, *args, **kwargs):
        constant = torch.full(x.shape, self.scalar, device=x.device, dtype=x.dtype)
        return x + constant


class MultiplyScalarModule(BendingModule):
    def __init__(self, scalar=1):
        super().__init__()
        self.scalar = scalar

    def bend(self, x, *args, **kwargs):
        constant = torch.full(x.shape, self.scalar, device=x.device, dtype=x.dtype)
        return x * constant


class ThresholdModule(BendingModule):
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold

    def bend(self, x, *args, **kwargs):
        return operations["threshold"](self.threshold)(x)


class RotateModule(BendingModule):
    def __init__(self, angle_degrees=0):
        super().__init__()
        self.angle_degrees = angle_degrees

    def bend(self, x, *args, **kwargs):
        return operations["rotate_image"](self.angle_degrees)(x)


class ScaleModule(BendingModule):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def bend(self, x, *args, **kwargs):
        return operations["scale_image"](self.scale_factor)(x)


class ErosionModule(BendingModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def bend(self, x, *args, **kwargs):
        return operations["erosion"](self.kernel_size)(x)


class DilationModule(BendingModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def bend(self, x, *args, **kwargs):
        return operations["dilation"](self.kernel_size)(x)


class GradientModule(BendingModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def bend(self, x, *args, **kwargs):
        return operations["gradient"](self.kernel_size)(x)


class SobelModule(BendingModule):
    def __init__(self, normalized=True):
        super().__init__()
        self.normalized = normalized

    def bend(self, x, *args, **kwargs):
        return operations["sobel"](self.normalized)(x)
