# Shared bending module classes used by both web UI and standalone model bending nodes.
# Kept in one place to avoid duplication between nodes.py and model_bending_nodes.py.
import math

import kornia.geometry.transform as KT
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
        if x.ndim == 5:
            # Video (B, C, T, H, W): pick the subset per frame, the same subset in every frame.
            b, c, t, h, w = x.shape
            folded = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            out = self.forward(folded, *args, **kwargs)
            return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        if x.ndim == 3:  # (B, L, C) tokens: treat as a 1-pixel-high image so channel/batch subsets still work
            return self.forward(x.permute(0, 2, 1).unsqueeze(2), *args, **kwargs).squeeze(2).permute(0, 2, 1)
        if x.ndim != 4:
            raise ValueError(f"[model-bending] subset expects a 3-D to 5-D tensor, got ndim={x.ndim}")

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

    # Temporal modules receive video tensors (B, C, T, H, W) whole instead of frame by frame.
    temporal = False

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"[model-bending] bending expects a tensor but the hooked module returned {type(x).__name__}. "
                "Transformer blocks that return tuples can be bent with 'DiT Block Bending'."
            )
        if x.ndim == 5 and not self.temporal:
            # Video tensors (B, C, T, H, W): fold time into the batch so 4-D ops act on each frame.
            b, c, t, h, w = x.shape
            folded = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            out = self.forward(folded, *args, **kwargs)
            return out.reshape(b, t, c, out.shape[-2], out.shape[-1]).permute(0, 2, 1, 3, 4)

        if x.ndim == 5:  # temporal module: sees (B, C, T, H, W) whole
            return self.bend(x, *args, **kwargs)

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

        # Some kornia-based ops squeeze a batch of one; restore the input layout.
        if output.shape != x.shape and output.numel() == x.numel():
            output = output.reshape(x.shape)
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


PADDING_MODES = ("zeros", "border", "reflection")


def _float_op(x, fn):
    """Run fn in float32 (kornia/grid_sample are unreliable in fp16/bf16) and cast back."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return fn(x.float()).to(x.dtype)
    return fn(x)


class RotateModule(BendingModule):
    def __init__(self, angle_degrees=0, padding="zeros"):
        super().__init__()
        self.angle_degrees = angle_degrees
        self.padding = padding

    def bend(self, x, *args, **kwargs):
        if self.padding == "zeros":
            return operations["rotate_image"](self.angle_degrees)(x)
        return _float_op(x, lambda v: KT.rotate(
            v, torch.full((v.shape[0],), float(self.angle_degrees), device=v.device, dtype=v.dtype),
            padding_mode=self.padding))


class ScaleModule(BendingModule):
    def __init__(self, scale_factor=1, padding="zeros"):
        super().__init__()
        self.scale_factor = scale_factor
        self.padding = padding

    def bend(self, x, *args, **kwargs):
        if self.padding == "zeros":
            return operations["scale_image"](self.scale_factor)(x)
        s = float(self.scale_factor)
        return _float_op(x, lambda v: KT.scale(
            v, torch.tensor([[s, s]], device=v.device, dtype=v.dtype).expand(v.shape[0], 2), padding_mode=self.padding))


class TranslateModule(BendingModule):
    """Shift by (dx, dy) as a fraction of the width/height (0.5 = half the frame); positive = right/down."""

    def __init__(self, dx=0.0, dy=0.0, padding="border"):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.padding = padding

    def bend(self, x, *args, **kwargs):
        h, w = x.shape[-2:]
        shift = [float(self.dx) * w, float(self.dy) * h]
        return _float_op(x, lambda v: KT.translate(
            v, torch.tensor([shift], device=v.device, dtype=v.dtype).expand(v.shape[0], 2), padding_mode=self.padding))


class FlipModule(BendingModule):
    def __init__(self, direction="horizontal"):
        super().__init__()
        self.direction = direction

    def bend(self, x, *args, **kwargs):
        dims = {"horizontal": (-1,), "vertical": (-2,), "both": (-2, -1)}[self.direction]
        return torch.flip(x, dims)


def _gaussian_kernel1d(sigma, device):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum(), radius


def gaussian_blur_along(x, sigma, dim):
    """1-D Gaussian blur of x along `dim` with edge replication (works for any size, unlike reflect padding)."""
    if sigma <= 0 or x.shape[dim] < 2:
        return x
    k, radius = _gaussian_kernel1d(float(sigma), x.device)
    xf = x.float().movedim(dim, -1)
    shape = xf.shape
    flat = xf.reshape(-1, 1, shape[-1])
    flat = torch.nn.functional.pad(flat, (radius, radius), mode="replicate")
    flat = torch.nn.functional.conv1d(flat, k.view(1, 1, -1))
    return flat.reshape(shape).movedim(-1, dim).to(x.dtype)


def gaussian_blur2d(x, sigma):
    return gaussian_blur_along(gaussian_blur_along(x, sigma, -1), sigma, -2)


class GaussianBlurModule(BendingModule):
    """Spatial Gaussian blur (edge replication); sigma in latent/token cells."""

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def bend(self, x, *args, **kwargs):
        return gaussian_blur2d(x, self.sigma)


class SharpenModule(BendingModule):
    """Unsharp mask: x + amount * (x - blur(x))."""

    def __init__(self, amount=1.0, sigma=1.0):
        super().__init__()
        self.amount = amount
        self.sigma = sigma

    def bend(self, x, *args, **kwargs):
        return x + float(self.amount) * (x - gaussian_blur2d(x, self.sigma))


# ---------------------------------------------------------------------------
# Temporal modules: act across the frames of video tensors (B, C, T, H, W).
# A 4-D input is treated as a single frame.
# ---------------------------------------------------------------------------

def _as_video(x):
    return (x.unsqueeze(2), True) if x.ndim == 4 else (x, False)


FRAME_CURVES = ("linear", "ease_in", "ease_out", "smooth", "triangle")


def frame_weights(num_frames, w_start, w_end, curve="linear"):
    """Per-frame weights from w_start (first frame) to w_end (last frame) along a curve."""
    if num_frames <= 1:
        return [float(w_start)]
    out = []
    for i in range(num_frames):
        u = i / (num_frames - 1)
        if curve == "ease_in":
            u = u * u
        elif curve == "ease_out":
            u = 1 - (1 - u) * (1 - u)
        elif curve == "smooth":
            u = 0.5 - 0.5 * math.cos(math.pi * u)
        elif curve == "triangle":
            u = 1 - abs(2 * u - 1)
        out.append(float(w_start) + (float(w_end) - float(w_start)) * u)
    return out


class FrameRampModule(BendingModule):
    """
    Applies an inner bend with a strength that varies over the frames: frame f becomes
    x_f + w(f) * (bend(x_f) - x_f), with w going from w_start to w_end along `curve`
    ('triangle' peaks in the middle). Works with any inner op, so effects can grow, fade or pulse over time.
    """
    temporal = True

    def __init__(self, module, w_start=0.0, w_end=1.0, curve="linear"):
        super().__init__()
        self.module = module
        self.w_start = w_start
        self.w_end = w_end
        self.curve = curve

    def bend(self, x, *args, **kwargs):
        v, was_image = _as_video(x)
        bent = self.module(v, *args, **kwargs)
        w = torch.tensor(frame_weights(v.shape[2], self.w_start, self.w_end, self.curve),
                         device=v.device, dtype=torch.float32).view(1, 1, -1, 1, 1)
        out = (v.float() + w * (bent.float() - v.float())).to(v.dtype)
        return out.squeeze(2) if was_image else out


class TemporalShiftModule(BendingModule):
    """Moves content `frames` frames later in time (negative = earlier). 'border' repeats the edge frame,
    'wrap' rolls around, 'zeros' fills with zeros."""
    temporal = True

    def __init__(self, frames=1, padding="border"):
        super().__init__()
        self.frames = frames
        self.padding = padding

    def bend(self, x, *args, **kwargs):
        v, was_image = _as_video(x)
        n, t = int(self.frames), v.shape[2]
        if n == 0 or t < 2:
            return x
        if self.padding == "wrap":
            out = torch.roll(v, shifts=n, dims=2)
        else:
            idx = torch.arange(t, device=v.device) - n
            valid = (idx >= 0) & (idx < t)
            out = v.index_select(2, idx.clamp(0, t - 1))
            if self.padding == "zeros":
                out = out * valid.view(1, 1, -1, 1, 1).to(out.dtype)
        return out.squeeze(2) if was_image else out


class TemporalBlurModule(BendingModule):
    """Gaussian blur along time (sigma in latent frames); smears motion and makes content linger."""
    temporal = True

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def bend(self, x, *args, **kwargs):
        if x.ndim != 5:
            return x
        return gaussian_blur_along(x, self.sigma, 2)


class FrameReverseModule(BendingModule):
    """Reverses the order of the frames."""
    temporal = True

    def bend(self, x, *args, **kwargs):
        return torch.flip(x, (2,)) if x.ndim == 5 else x


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
