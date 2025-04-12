import torch
import torch.nn as nn
from .bendutils import operations

# ------------------- Bending classes-----------------------


class BendingModule(nn.Module):
    '''
    Base class for all bending operations, mainly to perform some pre and post processing on the results. Specifically, to ensure the input's shape aligns coming in and going out.
    Note that bending operation can be defined by directly extending nn.Module if pre/post processing is not needed. For example:

    class AddNoiseModule(nn.Module):
        def __init__(self, noise_std=1):
            super().__init__()
            self.noise_std = noise_std

        def forward(self, x, *args, **kwargs):
            noise = x.new_empty(x.shape).normal_(std=self.noise_std)
            return x + noise

        Note: the  *args, **kwargs are added to anticipate any extra parameters passed to this method when this module is called. e.g., the SpatialTransformer module under comfy/ldm/module/attention.py does that
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # Check if the input is 3D (single image), and add a batch dimension if necessary.
        batch_added = False
        if x.ndim == 3:  # Single image: (C, H, W)
            x = x.unsqueeze(0)  # Convert to (1, C, H, W)
            batch_added = True
        elif x.ndim != 4:  # Expect a 4D tensor: (B, C, H, W)
            raise ValueError(
                f"Input tensor must be 3D or 4D, but got ndim={x.ndim}")

        # Delegate the transformation to the child class using a separate method.
        output = self.bend(x, *args, **kwargs)

        # If we added a batch dimension, remove it from the output.
        if batch_added:
            output = output.squeeze(0)
        return output

    def bend(self, x, *args, **kwargs):
        # Child classes must override this method
        raise NotImplementedError(
            "Subclasses must implement the compute() method.")


class AddNoiseModule(BendingModule):
    def __init__(self, noise_std=1):
        super().__init__()
        self.noise_std = noise_std

    def bend(self, x, *args, **kwargs):
        noise = x.new_empty(x.shape).normal_(std=self.noise_std)

        return x + noise


class AddScalarModule(BendingModule):
    def __init__(self, scalar=1):
        super().__init__()
        self.scalar = scalar

    def bend(self, x, *args, **kwargs):
        constant = torch.full(x.shape, self.scalar,
                              device=x.device, dtype=x.dtype)

        return x + constant


class MultiplyScalarModule(BendingModule):
    def __init__(self, scalar=1):
        super().__init__()
        self.scalar = scalar

    def bend(self, x, *args, **kwargs):
        constant = torch.full(x.shape, self.scalar,
                              device=x.device, dtype=x.dtype)

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
