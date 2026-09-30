# Shared bending module classes used by both web UI and standalone model bending nodes.
# Kept in one place to avoid duplication between nodes.py and model_bending_nodes.py.
import math

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
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"[model-bending] bending expects a tensor but the hooked module returned {type(x).__name__}. "
                "Transformer blocks that return tuples can be bent with 'DiT Block Bending'."
            )
        if x.ndim == 5:
            # Video tensors (B, C, T, H, W): fold time into the batch so 4-D ops act on each frame.
            b, c, t, h, w = x.shape
            folded = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            out = self.forward(folded, *args, **kwargs)
            return out.reshape(b, t, c, out.shape[-2], out.shape[-1]).permute(0, 2, 1, 3, 4)

        num_unsqueeze_added = 0
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            num_unsqueeze_added = 2
        if x.ndim == 3:
            x = x.unsqueeze(0)
            num_unsqueeze_added = 1
        elif x.ndim != 4:
            raise ValueError(f"[model-bending] input tensor must be 2D-5D, but got ndim={x.ndim}")

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


class GatedBendingModule(nn.Module):
    """
    Wraps a bending module with a diffusion-time window [t_end, t_start] (normalised t, 1 = pure noise).
    The appliers read weight(t) and blend: x + w(t) * (bend(x) - x). Outside the window w = 0.
    ramp: "hard" (w = 1 inside), "linear" or "cosine" (w rises over ramp_width at both edges).
    """
    is_gated = True

    def __init__(self, module, t_start=1.0, t_end=0.0, ramp="hard", ramp_width=0.1):
        super().__init__()
        self.inner = module
        self.t_start = max(t_start, t_end)
        self.t_end = min(t_start, t_end)
        self.ramp = ramp
        self.ramp_width = max(0.0, ramp_width)

    def weight(self, t):
        if t is None:
            return 1.0
        if t > self.t_start or t < self.t_end:
            return 0.0
        if self.ramp == "hard" or self.ramp_width <= 0:
            return 1.0
        # Ramp only at interior edges: a window that starts at pure noise (1.0) or ends at the clean image
        # (0.0) has nothing to fade in from or out to.
        distances = []
        if self.t_start < 1.0:
            distances.append(self.t_start - t)
        if self.t_end > 0.0:
            distances.append(t - self.t_end)
        if not distances:
            return 1.0
        edge = max(0.0, min(1.0, min(distances) / self.ramp_width))
        if self.ramp == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * edge)
        return edge

    def forward(self, x, *args, **kwargs):
        # Called directly only when an applier does not know about gating: bend unconditionally.
        return self.inner(x, *args, **kwargs)



class FourierAmplifyModule(BendingModule):
    """
    Amplifies specific frequency components of a tensor using FFT.
    Inherits shape alignment from BendingModule.
    """
    def __init__(self, cutoff_freq=5, amp_factor=2.0, steps_to_bend=None):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.amp_factor = amp_factor
        self.steps_to_bend = steps_to_bend
        self.current_step = None

    def bend(self, x, *args, **kwargs):
        # 1. FFT to Frequency Domain
        # We use .float() because FFT typically requires float32/64
        # 2-D FFT per sample and channel: transforming all dims would mix the cond/uncond batch and channels.
        sample_fft = torch.fft.fft2(x.float(), dim=(-2, -1))
        sample_fft_shifted = torch.fft.fftshift(sample_fft, dim=(-2, -1))

        # 2. Create the Low-Pass Mask
        b, c, h, w = x.shape
        
        # Create coordinate grids
        # Note: We use x.device to ensure mask is on the same device as input
        y_grid = torch.arange(-h // 2, h // 2, device=x.device).view(h, 1)
        x_grid = torch.arange(-w // 2, w // 2, device=x.device).view(1, w)
        
        # Calculate Euclidean distance from center (radius)
        radius = torch.sqrt(x_grid**2 + y_grid**2)

        # Create mask: 1 for low frequencies (inside circle), 0 for high (outside)
        mask = (radius < self.cutoff_freq).float()
        
        # Reshape mask for broadcasting: [1, 1, h, w]
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 3. Targeted Amplification
        # (1-mask) preserves high frequencies (noise/details) at 1.0x
        # (mask * amp_factor) scales low frequencies (structure)
        sample_fft_filtered = (sample_fft_shifted * (1 - mask)) + \
                               (sample_fft_shifted * mask * self.amp_factor)

        # 4. Inverse FFT to Spatial Domain
        sample_fft_ishifted = torch.fft.ifftshift(sample_fft_filtered, dim=(-2, -1))
        denoised_sample = torch.fft.ifft2(sample_fft_ishifted, dim=(-2, -1)).real

        # Return to original dtype (e.g., float16 if working with SDXL/Latents)
        return denoised_sample.to(x.dtype)
    
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
