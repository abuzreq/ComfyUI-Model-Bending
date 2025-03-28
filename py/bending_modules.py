import torch
import torch.nn as nn
from .bendutils import operations

# ------------------- Bending classes-----------------------


class AddNoiseModule(nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda', noise_std=1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, image):
        noise = image.new_empty(image.shape).normal_(std=self.noise_std)

        return image + noise


class AddScalarModule(nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda', scalar=1):
        super().__init__()
        self.scalar = scalar

    def forward(self, image):
        constant = torch.full(image.shape, self.scalar,
                              device=image.device, dtype=image.dtype)

        return image + constant


class MultiplyScalarModule(nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda', scalar=1):
        super().__init__()
        self.scalar = scalar

    def forward(self, image):
        constant = torch.full(image.shape, self.scalar,
                              device=image.device, dtype=image.dtype)

        return image * constant


class ThresholdModule(nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda', threshold=0):
        super().__init__()
        self.threshold = threshold

    def forward(self, image):
        return operations["threshold"](self.threshold)(image)


class RotateModule(nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda', angle_degrees=0):
        super().__init__()
        self.angle_degrees = angle_degrees

    def forward(self, image):
        return operations["rotate_image"](self.angle_degrees)(image)


class ScaleModule(nn.Module):
    def __init__(self, dtype=torch.float32, device='cuda', scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, image):
        return operations["scale_image"](self.scale_factor)(image)


class ErosionModule(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image):
        return operations["erosion"](self.kernel_size)(image)


class DilationModule(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image):
        return operations["dilation"](self.kernel_size)(image)


class GradientModule(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image):
        return operations["gradient"](self.kernel_size)(image)


class SobelModule(nn.Module):
    def __init__(self, normalized=True):
        super().__init__()
        self.normalized = normalized

    def forward(self, image):
        return operations["sobel"](self.normalized)(image)
