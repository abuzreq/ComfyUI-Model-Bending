# modified from https://github.com/dzluke/DAFX2024/blob/main/util.py

import copy
import logging
import numpy as np
import math
import argparse
import torch
import torch.nn as nn
import scipy.linalg
import random
from typing import NamedTuple
from kornia import morphology, filters
import kornia.geometry.transform as KT
from comfy.model_base import BaseModel

SAMPLING_RATE = None


def ensure_clone_has_own_inner(original_model, cloned_model):
    """
    Ensure the cloned Comfy model has its own inner model/stream so that modifications
    (hooks, unet replacement) only affect the clone and never the original.
    ComfyUI's ModelPatcher.clone() shares the same inner .model (and .stream); assigning
    to cloned_model.model.diffusion_model or registering hooks on it would mutate the original.
    """
    if hasattr(original_model, "model") and getattr(cloned_model, "model", None) is getattr(original_model, "model", None):
        inner = original_model.model
        cloned_model.model = copy.copy(inner)
        if hasattr(inner, "_modules") and isinstance(getattr(inner, "_modules", None), dict):
            cloned_model.model._modules = type(inner._modules)(list(inner._modules.items()))
        if hasattr(inner, "diffusion_model") and getattr(inner, "diffusion_model", None) is not None:
            cloned_model.model.diffusion_model = copy.deepcopy(inner.diffusion_model)
    if hasattr(original_model, "stream") and getattr(cloned_model, "stream", None) is getattr(original_model, "stream", None):
        stream = original_model.stream
        cloned_model.stream = copy.copy(stream)
        if hasattr(stream, "_modules") and isinstance(getattr(stream, "_modules", None), dict):
            cloned_model.stream._modules = type(stream._modules)(list(stream._modules.items()))
        if hasattr(stream, "unet") and getattr(stream, "unet", None) is not None:
            cloned_model.stream.unet = copy.deepcopy(stream.unet)


def hook_module(model: nn.Module, layer_path: str, new_module: nn.Module):
    """
    Injects new_module into model at the specified layer_path.

    - If the target module supports hooks, a forward hook is registered that calls new_module,
      passing output, positional arguments, and keyword arguments.
    - If the target module is not hookable (e.g. a bare nn.ModuleList which has no proper forward),
      the target is wrapped in a new container (a Sequential) that appends new_module.

    Args:
        model (nn.Module): The model instance.
        layer_path (str): Dot-separated path to the target module
            e.g. "features.3" or "block.0"
        new_module (nn.Module): The module to inject.

    Returns:
        A hook handle if a hook is registered, otherwise None.
    """
    if not layer_path:
        return

    # Split and navigate to the parent of the target module.
    parts = layer_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last_part = parts[-1]

    # Get the target module reference.
    if last_part.isdigit():
        idx = int(last_part)
        target_module = parent[idx]
    else:
        target_module = getattr(parent, last_part)

    # Decide whether to hook or wrap:
    # In this example, if the target is an nn.ModuleList (that isn’t already a Sequential),
    # we consider it not properly hookable.
    if isinstance(target_module, nn.ModuleList):
        print("Module List detected, wrapping in Sequential.")
        # Wrap by creating a new Sequential that contains the modules from target_module,
        # and then appending the new_module.
        # We don't simply append to the ModuleList instance because there is no guarantee that the list's items will be iterated on and called during the model execution.
        modules = list(target_module)
        modules.append(new_module)
        new_seq = nn.Sequential(*modules)
        # Replace the attribute in the parent with the new Sequential.
        if last_part.isdigit():
            parent[int(last_part)] = new_seq
        else:
            setattr(parent, last_part, new_seq)
        print(
            f"Wrapped ModuleList at '{layer_path}' in a Sequential that appends the new module.")
        return None
    elif isinstance(target_module, nn.Sequential):
        target_module.append(new_module)
    else:
        # Otherwise, register a forward hook.
        def hook(module, args, kwargs, output):
            # Call the injected new_module, passing the output plus original args and kwargs.
            return new_module(output, *args, **kwargs)
        
        # The with_kwargs flag lets the hook capture both positional args and keyword args.
        handle = target_module.register_forward_hook(hook, with_kwargs=True)
        print(f"Registered hook on module at '{layer_path}'.")
        return handle


def inject_module(model: nn.Module, layer_path: str, new_module: nn.Module):
    """
    Replaces or appends a submodule in `model` at the location given by `layer_path`.

    If append is False (default), the target module is replaced.
    If append is True, the target module is expected to be an nn.ModuleList,
    and the new module is appended to it.

    Args:
        model (nn.Module): The model instance.
        layer_path (str): Dot-separated path to the target attribute,
                          e.g., "output_blocks.0" (for replacement) or "output_blocks" (for appending).
        new_module (nn.Module): The PyTorch module to inject.
        append (bool): If True, append to a ModuleList. Otherwise, replace the module.


    Note: An older version of hook_module that injects the new_module into the model.
    However, in some cases adding new modules into an existing model is intrusive and may change how the model behaves so we use hook_module instead. 

    """
    if len(layer_path) == 0:
        return
    parts = layer_path.split('.')
    # Navigate to the parent of the target attribute.
    parent = model
    for part in parts[:-1]:

        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last_part = parts[-1]

    if not last_part.isdigit():
        target_module = getattr(parent, last_part)

        if not isinstance(target_module, nn.ModuleList) and not isinstance(target_module, nn.Sequential):
            seq = nn.Sequential()
            seq.append(target_module)
            seq.append(new_module)
            setattr(parent, last_part, seq)

            # raise ValueError(f"Target module at '{layer_path}' is not a ModuleList, cannot append.")
            print(f"Appended new module and target module into new list.", seq)
        else:
            target_module.append(new_module)
            print(f"Appended new module to ModuleList at '{layer_path}'.")
    else:
        target_module = parent[int(last_part)]
        # Standard replacement logic.

        idx = int(last_part)
        parent.insert(idx + 1, new_module)
        # parent[idx] = new_module

        print(f"Injected custom module into '{layer_path}'.")
    print("Updated Model Part", parent)


def _all_subclasses(cls):
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_all_subclasses(sub))
    return out


def process_path(path, extra_skips=[]):
    # All BaseModel subclasses, not only direct ones: WAN22, WAN21_Vace, WAN21_Camera, ... subclass WAN21.
    subclasses = ['BaseModel'] + [c.__name__.split('.')[-1] for c in _all_subclasses(BaseModel)]
    skips = ["diffusion_model", "UNet2DConditionModel", "Flux", "Flux2", ]
    skips += extra_skips
    # clean up loose dots at start or end
    start = 1 if path[0] == '.' else 0
    end = -1 if path[-1] == '.' else len(path)
    path = path[start:end]

    res = [x for x in path.split('.') if x not in skips and x not in subclasses]
    return res[0] if len(res) == 1 else '.'.join(res)


def get_model_tree(module):
    """
    Recursively builds a nested dictionary representing the module's structure.

    Returns a dictionary with the module type and any children modules.
    """
    tree = {"type": module.__class__.__name__}
    children = dict(module.named_children())
    if children:
        tree["children"] = {name: get_model_tree(
            child) for name, child in children.items()}

    return tree

def get_conv2d_leaf_paths(module: nn.Module, prefix=''):
    conv_paths = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and len(list(child.children())) == 0:
            conv_paths.append(path)
        else:
            conv_paths.extend(get_conv2d_leaf_paths(child, path))
    return conv_paths

def get_leaf_paths(module, prefix=''):
    paths = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        # If the child has no children, it's a leaf
        if len(list(child.children())) == 0:
            paths.append(path)
        else:
            paths.extend(get_leaf_paths(child, path))
    return paths

def get_conv2d_leaf_paths_with_types(module: nn.Module, prefix='', type_prefix=''):
    named_paths = []
    type_paths = []
    
    for name, child in module.named_children():
        name_path = f"{prefix}.{name}" if prefix else name
        type_name = child.__class__.__name__
        type_path = f"{type_prefix}.{type_name}" if type_prefix else type_name

        if isinstance(child, nn.Conv2d) and len(list(child.children())) == 0:
            named_paths.append(name_path)
            type_paths.append(type_path)
        else:
            child_named, child_types = get_conv2d_leaf_paths_with_types(child, name_path, type_path)
            named_paths.extend(child_named)
            type_paths.extend(child_types)
    
    return named_paths, type_paths

def set_sampling_rate(sr):
    global SAMPLING_RATE
    SAMPLING_RATE = sr


def clear_dir(p):
    """
    Delete the contents of the directory at p
    """
    if not p.is_dir():
        return
    for f in p.iterdir():
        if f.is_file():
            f.unlink()
        else:
            clear_dir(f)
            f.rmdir()


def format_time(seconds):
    """

    :param seconds:
    :return: a dictionary with the keys 'h', 'm', 's', that is the amount of hours, minutes, seconds equal to 'seconds'
    """
    hms = [seconds // 3600, (seconds // 60) % 60, seconds % 60]
    hms = [int(t) for t in hms]
    labels = ['h', 'm', 's']
    return {labels[i]: hms[i] for i in range(len(hms))}


def time_string(seconds):
    """
    Returns a string with the format "0h 0m 0s" that represents the amount of time provided
    :param seconds:
    :return: string
    """
    t = format_time(seconds)
    return "{}h {}m {}s".format(t['h'], t['m'], t['s'])


def rms(audio, sr):
    return np.sqrt(np.mean(audio**2))


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def spectrum(audio, sr):
    """
    Return the magnitude spectrum and the frequency bins

    :returns amps, freqs: where the ith element of amps is the amplitude of the ith frequency in freqs
    """
    # TODO: Should this be the normalized magnitude spectrum?
    # calculate amplitude spectrum
    N = next_power_of_2(audio.size)
    fft = np.fft.rfft(audio, N)
    amplitudes = abs(fft)

    # get frequency bins
    frequencies = np.fft.rfftfreq(N, d=1. / sr)

    return amplitudes, frequencies


def centroid(audio, sr):
    """
    Compute the spectral centroid of the given audio.
    Spectral centroid is the weighted average of the frequencies, where each frequency is weighted by its amplitude

    the centroid is the sum across each frequency f and amplitude a: f * a / sum(a)

    :param audio: audio as a numpy array
    :param sr: the sampling rate of the audio
    """

    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0

    return sum(amps * freqs) / sum(amps)


def spread(audio, sr):
    """
    Compute the spectral spread of the given audio
    Spectral spread is the average each frequency component weighted by its amplitude and subtracted by the spectral centroid

    spread = sqrt(sum(amp(k) * (freq(k) - centroid)^2) / sum(amp))
    """
    amps, freqs = spectrum(audio, sr)
    cent = centroid(audio, sr)

    if sum(amps) == 0:
        return 0

    return math.sqrt(sum(amps * (freqs - cent)**2) / sum(amps))


def skewness(audio, sr):
    """
    Compute the spectral skewness

    Skewness is the sum of (freq - centroid)^3 * amps divided by (spread^3 times the sum of the amps)

    """
    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0

    cent = centroid(audio, sr)
    spr = spread(audio, sr)

    return sum(amps * (freqs - cent)**3) / (spr**3 * sum(amps))


def kurtosis(audio, sr):
    """
    Compute the spectral kurtosis

    Kurtosis is the sum of (freq - centroid)^4 * amp divided by (the spread^4 times the sum of the amps)

    """
    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0

    cent = centroid(audio, sr)
    spr = spread(audio, sr)

    return sum(amps * (freqs - cent)**4) / (spr**4 * sum(amps))


def moments(audio, sr):
    """
    Return the four statistical moments that make up the spectral shape: centroid, spread, skewness, kurtosis
    """
    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0, 0, 0, 0

    cent = sum(amps * freqs) / sum(amps)
    spr = math.sqrt(sum(amps * (freqs - cent)**2) / sum(amps))
    skew = sum(amps * (freqs - cent)**3) / (spr**3 * sum(amps))
    kurt = sum(amps * (freqs - cent)**4) / (spr**4 * sum(amps))

    return cent, spr, skew, kurt


def flux(spectrum1, spectrum2):
    """
    Sum of absolute value of current amp minus prev frame's amp
    @param spectrum1: an np array of amplitudes, which is spectrum(audio, sr)[0]
    @param spectrum2: an np array of amplitudes, which is spectrum(audio, sr)[0]
    @return: spectral flux
    """
    if spectrum2 is None or spectrum1 is None:
        return 0
    spectral_flux = 0
    assert spectrum1.size == spectrum2.size
    for i in range(spectrum1.size):
        diff = abs(spectrum1[i] - spectrum2[i])
        spectral_flux += diff
    return spectral_flux


def add_scalar(x, a):
    """

    @param x:
    @param a:
    @return:
    """
    return x + (torch.ones_like(x) * a)


def add_rms(x, audio):
    """
    Add the rms of audio to the latent tensor x
    @param x: latent tensor
    @param audio:
    @return: same shape as x
    """
    return x + (torch.ones_like(x) * rms(audio))


def add_gaussian_rms(x, audio):
    """
    Add a matrix made of random samples from a normal gaussian
    @param x: latent tensor
    @param audio:
    @return:
    """
    return x + torch.normal(torch.zeros_like(x), torch.ones_like(x)) * rms(audio)


def add_centroid(x, audio):
    """
    Add a tensor full of the spectral centroid to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./1000
    return x + (torch.ones_like(x) * centroid(audio, SAMPLING_RATE) * scale)


def add_spread(x, audio):
    """
    Add a tensor full of the spectral spread to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./10
    return x + (torch.ones_like(x) * spread(audio, SAMPLING_RATE) * scale)


def add_skewness(x, audio):
    """
    Add a tensor full of the spectral skewness to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./10
    return x + (torch.ones_like(x) * skewness(audio, SAMPLING_RATE) * scale)


def add_kurtosis(x, audio):
    """
    Add a tensor full of the spectral kurtosis to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./10
    return x + (torch.ones_like(x) * kurtosis(audio, SAMPLING_RATE) * scale)


def add_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to every element
    """
    return lambda x: x + (torch.ones_like(x) * r)


def add_sparse(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to 25% of the elements
    """
    return lambda x: x + ((torch.rand_like(x) < 0.05) * r)


def add_noise(r):
    """
    Return a fn that adds Gaussian noise mutliplied by r to x
    """

    return lambda x: x + (torch.randn_like(x) * r)


def multiply_scalar(r):
    return lambda x: x * (torch.ones_like(x) * r)


def subtract_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    subtracted from every element
    """
    return lambda x: x - (torch.ones_like(x) * r)


def threshold(r):
    def thresh(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: y if abs(y) >= r else 0)
        x = x.to(device)
        return x
    return thresh


def soft_threshold(r):
    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
    def fn(x):
        x = x / x.abs() * torch.maximum(x.abs() - r, torch.zeros_like(x))
        return x
    return fn


def soft_threshold2(r):
    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
    def fn(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: 0 if abs(y) < r else y*(1-r))
        x = x.to(device)
        return x
    return fn


def inversion(r):
    def foo(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: 1./r - y)
        x = x.to(device)
        return x
    return foo


def inversion2():
    return lambda x: -1 * x


def log(r):
    return lambda x: torch.log(x)


def power(r):
    return lambda x: torch.pow(x, r)


def add_dim(r, dim, i):
    def foo(x):
        """
        Add value r to the given dim at index i
        dim = 0 means adding in "z" dimension (shape = 4)
        dim = 1 means adding to row i
        dim = 2 means adding to column i
        @return: modified x
        """
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] += r
        elif dim == 1:
            x[:, i:i + 1] += r
        elif dim == 2:
            x[:, :, i:i + 1] += r
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def add_rand_cols(r, k):
    """
    Return a fn that will add value 'r' to a fraction of the cols of a tensor
    Assume x is a 3-tensor and rows refer to the third dimension
    k should be between 0 and 1
    """
    def foo(x):
        dim = x.shape[1]
        cols = random.sample(range(dim), int(k * dim))
        for col in cols:
            x[:, :, col:col + 1] += r
        return x

    return foo


def add_rand_rows(r, k):
    """
    Return a fn that will add value 'r' to a fraction of the rows of a tensor
    Assume x is a 3-tensor and rows refer to the second dimension
    k should be between 0 and 1
    """
    def foo(x):
        dim = x.shape[2]
        rows = random.sample(range(dim), int(k * dim))
        for row in rows:
            x[:, row:row + 1] += r
        return x

    return foo


def invert_dim(r, dim, i):
    def foo(x):
        """
        Apply inversion (1/r. - x) at the given dim at index i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        invert = inversion(r)
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] = invert(x[a, i, i])
        elif dim == 1:
            x[:, i:i + 1] = invert(x[:, i:i + 1])
        elif dim == 2:
            x[:, :, i:i + 1] += invert(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def apply_to_dim(func, r, dim, i):
    def foo(x):
        """
        Apply func at the given dim at i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        fn = func(r)
        if dim == 0:
            for a in range(x.shape[0]):
                try:
                    x[a, i[0], i[1]] = fn(x[a, i[0], i[1]])
                except TypeError:
                    x[a, i, i] = fn(x[a, i, i])
        elif dim == 1:
            try:
                x[:, i[0]:i[1]] = fn(x[:, i[0]:i[1]])
            except TypeError:
                x[:, i:i + 1] = fn(x[:, i:i + 1])
        elif dim == 2:
            try:
                x[:, :, i[0]:i[1]] = fn(x[:, :, i[0]:i[1]])
            except TypeError:
                x[:, :, i:i + 1] = fn(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def apply_sparse(func, sparsity):
    """
    return a function that applies the given function a random fraction of the elements, as determined by 'sparsity'
    0 < sparsity < 1
    """
    def fn(x):
        mask = torch.rand_like(x) < sparsity
        x = (x * ~mask) + (func(x) * mask)
        return x
    return fn


def add_normal(r):
    """
    Add a 2D normal gaussian (bell curve) to the center of the tensor
    """
    def foo(x):
        # chatgpt wrote this
        # Define the size of the matrix
        size = 64

        # Generate grid coordinates centered at (0,0)
        a = np.linspace(-5, 5, size)
        b = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(a, b)

        # Standard deviation
        sigma = 1.5  # You can adjust this value as desired

        # Generate a 2D Gaussian distribution with peak at the center and specified standard deviation
        Z = np.exp(-0.5 * ((X / sigma) ** 2 + (Y / sigma) ** 2)) / \
            (2 * np.pi * sigma ** 2)
        Z *= r
        Z = torch.from_numpy(Z).to(x.get_device())

        for i in range(x.shape[0]):
            x[i] += Z
        return x
    return foo


def tensor_exp(r):
    """
    Return a fn that computes the matrix exponential of a given tensor
    """
    def foo(x):
        device = x.get_device()
        x = x.cpu().numpy()
        x = scipy.linalg.expm(x)
        x = torch.from_numpy(x).to(device)
        return x
    return foo


def rotate_z(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "z" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c, -1 * s, 0, 0],
            [s,    c,   0, 0],
            [0,    0,   1, 0],
            [0,    0,   0, 1]
        ]
        op = torch.tensor(rotation_matrix, dtype=x.dtype, device=x.device)
        x = x.squeeze(0)
        x = torch.tensordot(op, x, dims=1)
        x = x.unsqueeze(0)
        return x
    return fn


def rotate_image(degrees):
    def fn(x):
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.float()
        B = x.shape[0]
        angle_tensor = torch.full((B,), degrees, device=x.device, dtype=x.dtype)
        out = KT.rotate(x, angle=angle_tensor)
        if orig_dtype == torch.float16:
            out = out.to(orig_dtype)
        return out
    return fn

def scale_image(scale_factor):
    def fn(x):
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.float()
        scale_tensor = torch.tensor(
            [[scale_factor, scale_factor]], device=x.device, dtype=x.dtype)
        out = KT.scale(x, scale_tensor)
        if orig_dtype == torch.float16:
            out = out.to(orig_dtype)
        return out
    return fn


def rotate_x(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "x" axis
    """
    def fn(x):

        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [1, 0,   0,    0],
            [0, c, -1 * s, 0],
            [0, s,   c,    0],
            [0, 0,   0,    1]
        ]
        op = torch.tensor(rotation_matrix, dtype=x.dtype, device=x.device)

        x = x.squeeze(0)
        x = torch.tensordot(op, x, dims=1)
        x = x.unsqueeze(0)
        return x
    return fn


def rotate_y(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c,      0, s, 0],
            [0,      1, 0, 0],
            [-1 * s, 0, c, 0],
            [0,      0, 0, 1]
        ]
        op = torch.tensor(rotation_matrix, dtype=x.dtype, device=x.device)

        x = x.squeeze(0)
        x = torch.tensordot(op, x, dims=1)
        x = x.unsqueeze(0)

        return x
    return fn


def rotate_y2(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c,      0, 0, s],
            [0,      1, 0, 0],
            [0,      0, 1, 0],
            [-1 * s, 0, 0, c]
        ]
        op = torch.tensor(rotation_matrix, dtype=x.dtype, device=x.device)
        x = x.squeeze(0)
        x = torch.tensordot(op, x, dims=1)
        x = x.unsqueeze(0)

        return x
    return fn


def reflect(r):
    """
    Return a fn that reflects across the given dimension r
    r can be 0, 1, 2, or 3
    """
    def fn(x):
        op = torch.eye(4, device=x.device)  # identity matrix
        op[r, r] *= -1
        x = x.squeeze(0)
        x = torch.tensordot(op, x, dims=1)
        x = x.unsqueeze(0)
        return x
    return fn


def hadamard1():
    def fn(x):
        h = scipy.linalg.hadamard(4)
        op = torch.tensor(h, dtype=x.dtype, device=x.device)
        x = x.squeeze(0)
        x = torch.tensordot(op, x, dims=1)
        x = x.unsqueeze(0)

        return x
    return fn


def hadamard2(r):
    def fn(x):
        device = x.get_device()
        h = scipy.linalg.hadamard(64)
        op = torch.tensor(h).to(x.dtype).to(device)
        x = torch.tensordot(x, op, dims=[[1], [1]])
        return x
    return fn


def apply_both(fn1, fn2, r):
    def fn(x):
        return fn1(fn2(r))
    return fn


def normalize(func):
    """
    First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)  # first apply the network bending function
        # then normalize the result
        max = x.abs().max()
        x = x / max
        return x
    return fn


def normalize2(func):
    """
    First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)  # first apply the network bending function
        # then normalize the result
        x = torch.nn.functional.normalize(x, dim=0)
        return x
    return fn


def normalize3(func):
    """
        First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)
        x = x - x.mean()
        return x
    return fn


def normalize4(func, dim=0):
    """
        First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)
        x = x - torch.mean(x, dim=dim, keepdim=True)
        return x
    return fn


def gradient(r):
    def fn(x):
        # x = x.unsqueeze(0)
        kernel = torch.ones(r, r, dtype=x.dtype, device=x.device)
        x = morphology.gradient(x, kernel)
        x = x.squeeze(0)
        return x
    return fn


def dilation(r):
    def fn(x):
        # x = x.unsqueeze(0)
        kernel = torch.ones(r, r, dtype=x.dtype, device=x.device)
        x = morphology.dilation(x, kernel)
        x = x.squeeze(0)
        return x
    return fn


def erosion(r):
    def fn(x):
        #  x = x.unsqueeze(0)
        kernel = torch.ones(r, r, dtype=x.dtype, device=x.device)
        x = morphology.erosion(x, kernel)
        x = x.squeeze(0)
        return x
    return fn


def sobel(r=True):
    def fn(x):
        # x = x.unsqueeze(0)
        x = filters.sobel(x, normalized=r)
        x = x.squeeze(0)
        return x
    return fn


def absolute():
    """
    Return a fn that computes the absolute value of a tensor
    """
    def fn(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: abs(y))
        x = x.to(device)
        return x
    return fn


def log(r=math.e):
    """
    Return a fn that computes the log of a tensor with base r. Must first ensure that it is non-negative
    """
    def fn(x):
        device = x.get_device() if x.get_device() is not None else 'cpu'
        x = x.cpu()
        x = x.apply_(lambda y: abs(y))
        x = x.to(device)
        return torch.log(x) / math.log(r)
    return fn


def clamp(r1, r2):
    """
    Return a fn that clamps a tensor between min and max
    """
    def fn(x):
        min, max = r1, r2
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: min if y < min else max if y > max else y)
        x = x.to(device)
        return x
    return fn


def scale(r1, r2):
    """
    Return a fn that scales a tensor between min and max based on the tensor's min and max
    """
    def fn(x):
        min, max = r1, r2
        device = x.get_device()
        x = x.cpu()
        xmin = x.min()
        xmax = x.max()
        x = x.apply_(lambda y: (y - xmin) / (xmax - xmin) * (max - min) + min)
        x = x.to(device)
        return x
    return fn



def resolve_module(root: nn.Module, layer_path: str):
    """Navigate dot-separated path on an nn.Module (numeric parts index into containers)."""
    cur = root
    for part in layer_path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


# ---------------------------------------------------------------------------
# Hook engine shared by every applier (Model Bending, SD Blocks, JSON, probes, steering)
# ---------------------------------------------------------------------------

LOG_TAG = "[model-bending]"
log = logging.getLogger("model-bending")


def warn(msg, *args):
    log.warning(LOG_TAG + " " + msg, *args)


def info(msg, *args):
    log.info(LOG_TAG + " " + msg, *args)


def _is_bypassed_container(mod: nn.Module) -> bool:
    """
    True for containers whose own forward() is never called by the model, so hooks on them never fire.
    ComfyUI's UNet runs TimestepEmbedSequential blocks via forward_timestep_embed(), which iterates the
    children directly.
    """
    return any(cls.__name__ == "TimestepEmbedSequential" for cls in type(mod).__mro__)


def _close_matches(root: nn.Module, path: str, n=5):
    import difflib
    names = [name for name, _ in root.named_modules() if name]
    return difflib.get_close_matches(path, names, n=n, cutoff=0.6)


def resolve_hook_target(root: nn.Module, path: str, strict: bool = False, report: dict = None):
    """
    Resolve a layer path to a module a forward hook will actually fire on.
    - Containers bypassed by the model's forward (TimestepEmbedSequential) expand to their last child,
      whose output is the container's output.
    - nn.ModuleList cannot be called and is skipped.
    Returns (hooked_path, module); module is None when the path is skipped. strict=True raises instead.
    """
    def skip(reason):
        if strict:
            raise ValueError(f"{LOG_TAG} {reason}")
        warn(reason)
        if report is not None:
            report.setdefault("skipped", []).append({"path": path, "reason": reason})
        return path, None

    try:
        mod = resolve_module(root, path)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        hint = _close_matches(root, path)
        return skip(f"could not resolve path {path!r} ({exc})" + (f"; did you mean {' | '.join(hint)}?" if hint else ""))

    hooked = path
    while _is_bypassed_container(mod):
        children = list(mod.named_children())
        if not children:
            return skip(f"path {path!r} is an empty container")
        last_name, mod = children[-1]
        hooked = f"{hooked}.{last_name}"
    if hooked != path:
        info("expanded container %s -> %s (the container's forward is never called)", path, hooked)
        if report is not None:
            report.setdefault("expanded", []).append({"path": path, "hooked": hooked})

    if isinstance(mod, nn.ModuleList):
        n = len(mod)
        return skip(f"path {path!r} is a ModuleList, which is never called; hook one of {path}.0 ... {path}.{n - 1}")
    return hooked, mod


WILDCARD_CHARS = set("*?[")


def has_wildcard(path: str) -> bool:
    return any(ch in WILDCARD_CHARS for ch in path)


def expand_path_pattern(root: nn.Module, pattern: str):
    """
    Expand a dotted path whose segments may use shell wildcards ('output_blocks.*.1', 'input_blocks.[4-8].0',
    '*.attn2') against the module tree, in model order. Paths without wildcards are returned unchanged.
    """
    import fnmatch
    pattern = pattern.strip()
    if not has_wildcard(pattern):
        return [pattern]
    frontier = [("", root)]
    for part in pattern.split("."):
        nxt = []
        for path, mod in frontier:
            for name, child in mod.named_children():
                if fnmatch.fnmatchcase(name, part):
                    nxt.append((f"{path}.{name}" if path else name, child))
        frontier = nxt
    order = {name: i for i, (name, _) in enumerate(root.named_modules())}
    return sorted((p for p, _ in frontier), key=lambda p: order.get(p, len(order)))


def current_step(transformer_options):
    """Index of the current denoising step, derived from the sampler's sigma schedule (None if unknown)."""
    transformer_options = transformer_options or {}
    schedule = transformer_options.get("sample_sigmas")
    cur = transformer_options.get("sigmas")
    if schedule is None or cur is None:
        return None
    try:
        schedule = schedule.detach().float().cpu()
        c = float(cur.detach().float().flatten()[0].cpu())
        close = torch.isclose(schedule, torch.tensor(c), rtol=1e-4, atol=1e-6).nonzero(as_tuple=True)[0]
        if close.numel() > 0:
            return int(close[0])
        # Intermediate sigma (e.g. 2nd-order samplers): belongs to the step whose interval contains it.
        return max(0, int((schedule > c).sum()) - 1)
    except Exception:
        return None


def total_steps(transformer_options):
    schedule = (transformer_options or {}).get("sample_sigmas")
    if schedule is None:
        return None
    return max(1, len(schedule) - 1)


def model_sampling_of(apply_model):
    """
    The model_sampling object behind the apply_model passed to a model_function_wrapper: either the bound
    BaseModel.apply_model, or a replacement function that carries it as a `model_sampling` attribute
    (see with_model_sampling, used by wrappers that pass their own function down the chain).
    """
    ms = getattr(apply_model, "model_sampling", None)
    if ms is not None:
        return ms
    owner = getattr(apply_model, "__self__", None)
    return getattr(owner, "model_sampling", None)


def with_model_sampling(fn, apply_model):
    """Tag a replacement apply function so wrappers below it can still resolve diffusion time."""
    fn.model_sampling = model_sampling_of(apply_model)
    return fn


def normalized_t(model_sampling, sigma):
    """
    Diffusion time in [0, 1] (1 = pure noise) for the current sigma: model_sampling.timestep(sigma) rescaled
    by the model's own range. Discrete eps/v models give timestep/999; flow models (Flux, SD3) give sigma,
    so windows follow the noise level independently of step count, scheduler, shift or denoise.
    """
    if model_sampling is None or sigma is None:
        return None
    try:
        s = sigma.detach().flatten()[:1] if isinstance(sigma, torch.Tensor) else torch.tensor([float(sigma)])
        ts = float(model_sampling.timestep(s.float()).flatten()[0])
        t_max = float(model_sampling.timestep(torch.as_tensor(model_sampling.sigma_max).reshape(1).float()).flatten()[0])
        t_min = float(model_sampling.timestep(torch.as_tensor(model_sampling.sigma_min).reshape(1).float()).flatten()[0])
        if t_max == t_min:
            return None
        return min(1.0, max(0.0, (ts - t_min) / (t_max - t_min)))
    except Exception:
        return None


def t_window_from(t_start, t_end):
    """(hi, lo) window, or None when it covers the whole [0, 1] range."""
    hi, lo = max(t_start, t_end), min(t_start, t_end)
    if hi >= 1.0 and lo <= 0.0:
        return None
    return (hi, lo)


def in_window(step, t, steps_to_bend=None, t_window=None):
    if steps_to_bend is not None and step is not None and step not in steps_to_bend:
        return False
    if t_window is not None and t is not None:
        hi, lo = max(t_window), min(t_window)
        if t > hi + 1e-6 or t < lo - 1e-6:
            return False
    return True


GUARD_NAN_MODES = ("zero", "clamp", "none")


def blend_and_guard(x, y, weight, on_nonfinite=None, guard=None, flags=None):
    """
    x + weight * (y - x), then the guard:
    - "nan": "zero" (default) replaces NaN/Inf with 0, "clamp" maps NaN to 0 and Inf to the dtype's range,
      "none" leaves them
    - "max_std_ratio": r shrinks the result around its mean so its std is at most r times the unbent std
    - "preserve_norm": true rescales each channel to the unbent channel's L2 norm (changes direction, not energy)
    Non-finite detection: with `flags` (a list), a device-side flag tensor is appended and no sync happens; the
    caller checks the flags once per forward pass. Without it, on_nonfinite is called right away (one sync).
    """
    guard = guard or {}
    if weight != 1.0:
        y = x + weight * (y - x)
    if not y.is_floating_point():
        return y
    nan_mode = guard.get("nan", "zero")
    if nan_mode != "none":
        if flags is not None:
            flags.append(~torch.isfinite(y).all())
            replace = True  # nan_to_num is the identity on finite values, so apply it without looking
        else:
            replace = not bool(torch.isfinite(y).all())
            if replace and on_nonfinite is not None:
                on_nonfinite()
        if replace and nan_mode == "clamp":
            info = torch.finfo(y.dtype)
            y = torch.nan_to_num(y, nan=0.0, posinf=info.max, neginf=info.min)
        elif replace:
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = guard.get("max_std_ratio")
    if ratio and y.numel() > 1:
        yf = y.float()
        mean = yf.mean()
        factor = (float(ratio) * x.float().std() / yf.std().clamp_min(1e-12)).clamp(max=1.0)
        y = (mean + (yf - mean) * factor).to(y.dtype)
    if guard.get("preserve_norm") and y.shape == x.shape and y.ndim >= 2:
        cdim = y.ndim - 1 if y.ndim == 3 else 1
        other = tuple(d for d in range(y.ndim) if d != cdim)
        nx = x.float().pow(2).sum(dim=other, keepdim=True).sqrt()
        ny = y.float().pow(2).sum(dim=other, keepdim=True).sqrt()
        y = (y.float() * nx / ny.clamp_min(1e-12)).to(y.dtype)
    return y


class BendSpec:
    """One bend: a module applied at one or more paths, with optional step/t windows and blend weight."""

    def __init__(self, paths, module, steps_to_bend=None, t_window=None, blend=1.0, label=None, guard=None):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.module = module
        self.steps_to_bend = steps_to_bend
        self.t_window = t_window
        self.blend = float(blend)
        self.label = label
        self.guard = guard


def make_bends_wrapper(unet: nn.Module, specs, prev_wrapper=None, strict: bool = False, report: dict = None,
                       reverse_hooks: bool = False):
    """
    Return a model_function_wrapper that, for each forward pass:
    1. works out the current step and normalised t,
    2. registers forward hooks only for the bends whose window contains them,
    3. calls prev_wrapper (or apply_model) so several bending nodes chain,
    4. removes every hook in a finally block, so nothing persists on the shared model.
    No state is written to the bending modules, so one module can feed several appliers.
    Bends on the same module run in list order; reverse_hooks=True runs them last-to-first instead
    (the historical order of bends from JSON, which used one nested wrapper per bend).
    """
    resolved = []
    for spec in specs:
        targets = []
        for p in spec.paths:
            if not p:
                continue
            hooked, mod = resolve_hook_target(unet, p, strict=strict, report=report)
            if mod is not None:
                targets.append((hooked, mod))
        resolved.append((spec, targets))
        if report is not None:
            report.setdefault("resolved", []).extend(
                {"path": h, "op": type(_ungated(spec.module)).__name__, "label": spec.label,
                 "steps": spec.steps_to_bend, "t": list(spec.t_window) if spec.t_window else None,
                 "blend": spec.blend, **({"guard": spec.guard} if spec.guard else {})}
                for h, _ in targets
            )

    needs_t = any(s.t_window is not None or _is_gated(s.module) for s, _ in resolved)
    warned = set()
    ever_fired = set()
    # Video DiTs (WAN): (B, L, C) token outputs are laid out on their (frames, h, w) grid, so spatial ops act
    # on the picture rather than on the token-by-channel matrix. Image DiTs keep the historical behaviour.
    video_dit = is_video_dit(unet)
    pass_state = {"latent_shape": None}

    def warn_once(key, msg, *args):
        if key not in warned:
            warned.add(key)
            warn(msg, *args)

    def make_hook(path, module, weight, guard, flags):
        def bend(x, args, kwargs):
            if video_dit and x.ndim == 3:
                layout = token_layout_for(unet, x.shape[1], pass_state["latent_shape"], allow_extra=False)
                if layout is not None:
                    y = bend_on_token_grid(x, lambda v: module(v, *args, **kwargs), layout)
                    return blend_and_guard(x, y, weight, guard=guard, flags=flags)
            return blend_and_guard(x, module(x, *args, **kwargs), weight, guard=guard, flags=flags)

        def _hook(mod, args, kwargs, output):
            ever_fired.add(path)
            if isinstance(output, torch.Tensor):
                return bend(output, args, kwargs)
            if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                msg = (f"{path} returns a {type(output).__name__} of {len(output)} tensors; only the first is bent. "
                       "Use 'DiT Block Bending' for transformer blocks, or hook a Linear/Conv child.")
                if strict:
                    raise ValueError(f"{LOG_TAG} {msg}")
                warn_once(("tuple", path), msg)
                return type(output)([bend(output[0], args, kwargs), *output[1:]])
            msg = f"{path} returns {type(output).__name__}, which cannot be bent"
            if strict:
                raise ValueError(f"{LOG_TAG} {msg}")
            warn_once(("type", path), msg)
            return None
        return _hook

    def wrapper(apply_model, params):
        c = params["c"] or {}
        transformer_options = c.get("transformer_options", {})
        step = current_step(transformer_options)
        pass_state["latent_shape"] = tuple(params["input"].shape) if isinstance(params.get("input"), torch.Tensor) else None
        t = normalized_t(model_sampling_of(apply_model), params["timestep"]) if needs_t else None
        if needs_t and t is None:
            # Fail closed: a bend that must follow diffusion time is skipped rather than applied at every step.
            msg = ("cannot resolve diffusion time (no model_sampling reachable from apply_model); "
                   "time-windowed bends are skipped for this pass")
            if strict:
                raise ValueError(f"{LOG_TAG} {msg}")
            warn_once("no_t", msg)

        handles = []
        registered = []
        flags = []  # device-side non-finite flags, checked once after the forward pass
        for spec, targets in (reversed(resolved) if reverse_hooks else resolved):
            windowed = spec.t_window is not None or _is_gated(spec.module)
            if windowed and t is None:
                continue
            if not in_window(step, t, spec.steps_to_bend, spec.t_window):
                continue
            weight = spec.blend
            module = spec.module
            if _is_gated(module):
                weight *= module.weight(t)
                module = module.inner
            if weight == 0.0:
                continue
            for path, target in targets:
                handles.append(target.register_forward_hook(
                    make_hook(path, module, weight, spec.guard, flags), with_kwargs=True))
                registered.append(path)

        try:
            if prev_wrapper is not None:
                out = prev_wrapper(apply_model, params)
            else:
                out = apply_model(params["input"], params["timestep"], **c)
        finally:
            for h in handles:
                h.remove()
        # The forward pass completed: report hooks that were registered but have never fired.
        for path in registered:
            if path not in ever_fired:
                warn_once(("never", path),
                          "hook on %s never fired: the model's forward does not call this module. "
                          "Hook one of its children instead.", path)
        if flags and "nan" not in warned and bool(torch.stack(flags).any()):
            warn_once("nan", "a bend at %s produced NaN/Inf; replaced", ", ".join(sorted(set(registered))))
        return out

    return wrapper


def _is_gated(module):
    """GatedBendingModule (bending_modules.py) is detected by duck typing to avoid a circular import."""
    return getattr(module, "is_gated", False)


def _ungated(module):
    return module.inner if _is_gated(module) else module


def make_bending_unet_wrapper(unet: nn.Module, mod_paths, bending_module: nn.Module, prev_wrapper=None,
                              steps_to_bend=None, t_window=None, strict=False, report=None):
    """
    Single-module convenience around make_bends_wrapper (kept for existing callers).
    steps_to_bend defaults to the module's own steps_to_bend attribute (e.g. FourierAmplifyModule).
    """
    if steps_to_bend is None:
        steps_to_bend = getattr(bending_module, "steps_to_bend", None)
    spec = BendSpec(mod_paths, bending_module, steps_to_bend=steps_to_bend, t_window=t_window)
    return make_bends_wrapper(unet, [spec], prev_wrapper=prev_wrapper, strict=strict, report=report)


# ---------------------------------------------------------------------------
# Token <-> grid helpers for transformer (DiT) activations
# ---------------------------------------------------------------------------

def infer_token_grid(num_tokens, latent_shape):
    """
    Find the image-token grid for a (B, L, C) activation from the latent shape seen by the wrapper.
    Returns (frames, h, w) with frames*h*w <= num_tokens, or None. Tokens are assumed to be ordered
    frame-major, row-major, image first (extra tokens such as Kontext references follow the image).
    """
    if latent_shape is None or len(latent_shape) < 4:
        return None
    H, W = int(latent_shape[-2]), int(latent_shape[-1])
    T = int(latent_shape[2]) if len(latent_shape) == 5 else 1
    best = None
    # Video DiTs (WAN, HunyuanVideo) patchify 1x2x2: try that first, otherwise p=1, tp=4 gives the same token
    # count whenever T % 4 == 0 and wins with the wrong grid.
    patches = ((2, 1),) if T > 1 else ()
    patches += tuple((p, tp) for p in (1, 2, 4, 8, 16, 32, 64) for tp in ((1,) if T == 1 else (1, 2, 4)))
    for p, tp in patches:
        f, h, w = -(-T // tp), -(-H // p), -(-W // p)
        n = f * h * w
        if n == num_tokens:
            return (f, h, w)
        if n < num_tokens and (best is None or n > best[0] * best[1] * best[2]):
            best = (f, h, w)
    return best


def tokens_to_grid(x, grid):
    """(B, L, C) -> ((B*F, C, h, w), rest) where rest holds tokens beyond the grid (or None)."""
    f, h, w = grid
    n = f * h * w
    b, _, c = x.shape
    img, rest = x[:, :n], (x[:, n:] if x.shape[1] > n else None)
    img = img.reshape(b, f, h, w, c).permute(0, 1, 4, 2, 3).reshape(b * f, c, h, w)
    return img, rest


def grid_to_tokens(y, grid, batch, rest=None):
    f, h, w = grid
    c = y.shape[1]
    y = y.reshape(batch, f, c, h, w).permute(0, 1, 3, 4, 2).reshape(batch, f * h * w, c)
    return torch.cat((y, rest), dim=1) if rest is not None else y


class TokenLayout(NamedTuple):
    """Where the image tokens of a (B, L, C) DiT activation sit: `prefix` extra tokens, then the
    frame-major, row-major (frames, h, w) grid, then `suffix` extra tokens."""
    grid: tuple
    prefix: int = 0
    suffix: int = 0

    @property
    def num_image_tokens(self):
        return self.grid[0] * self.grid[1] * self.grid[2]


def dit_patch_size(dm):
    """(pt, ph, pw) patch size of a DiT, or None. WAN/HunyuanVideo store a 3-tuple, Flux/SD3 an int."""
    p = getattr(dm, "patch_size", None)
    if isinstance(p, int) and p > 0:
        return (1, p, p)
    if isinstance(p, (tuple, list)) and all(isinstance(v, int) and v > 0 for v in p):
        if len(p) == 3:
            return tuple(p)
        if len(p) == 2:
            return (1, p[0], p[1])
    return None


def is_video_dit(dm):
    p = getattr(dm, "patch_size", None)
    return isinstance(p, (tuple, list)) and len(p) == 3 and dit_patch_size(dm) is not None


def _prepends_extra_tokens(dm):
    # WAN's reference-latent tokens (ref_conv) are concatenated in front of the video tokens.
    return getattr(dm, "ref_conv", None) is not None


def token_layout_for(dm, num_tokens, latent_shape, allow_extra=True):
    """
    TokenLayout for a (B, num_tokens, C) activation of diffusion model `dm`, given the latent shape the
    model_function_wrapper saw. With a known patch size the grid is computed exactly, as the model's own
    pad_to_patch_size does (ceil); otherwise it is guessed with infer_token_grid. Extra tokens are prepended
    for models with reference tokens in front (WAN ref_conv) and appended otherwise (Kontext, S2V).
    allow_extra=False only accepts an exact match (for generic hooks, which may see text-token tensors too).
    Returns None when no grid fits.
    """
    if latent_shape is None or len(latent_shape) < 4 or num_tokens <= 0:
        return None
    patch = dit_patch_size(dm) if dm is not None else None
    if patch is not None:
        pt, ph, pw = patch
        T = int(latent_shape[2]) if len(latent_shape) == 5 else 1
        H, W = int(latent_shape[-2]), int(latent_shape[-1])
        f, h, w = -(-T // pt), -(-H // ph), -(-W // pw)
        n = f * h * w
        if n == num_tokens:
            return TokenLayout((f, h, w))
        if n < num_tokens and allow_extra:
            extra = num_tokens - n
            if _prepends_extra_tokens(dm):
                return TokenLayout((f, h, w), prefix=extra)
            return TokenLayout((f, h, w), suffix=extra)
        if n < num_tokens and _prepends_extra_tokens(dm) and (num_tokens - n) % (h * w) == 0:
            return TokenLayout((f, h, w), prefix=num_tokens - n)
        return None
    grid = infer_token_grid(num_tokens, latent_shape)
    if grid is None:
        return None
    n = grid[0] * grid[1] * grid[2]
    if n != num_tokens and not allow_extra:
        return None
    return TokenLayout(tuple(grid), suffix=num_tokens - n)


def tokens_to_video(x, layout):
    """(B, L, C) -> ((B, C, F, h, w), prefix tokens or None, suffix tokens or None)."""
    f, h, w = layout.grid
    n = f * h * w
    b, _, c = x.shape
    p = layout.prefix
    pre = x[:, :p] if p else None
    post = x[:, p + n:] if x.shape[1] > p + n else None
    v = x[:, p:p + n].reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
    return v, pre, post


def video_to_tokens(v, pre=None, post=None):
    """Inverse of tokens_to_video."""
    b, c, f, h, w = v.shape
    y = v.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)
    parts = [t for t in (pre, y, post) if t is not None]
    return torch.cat(parts, dim=1) if len(parts) > 1 else y


def bend_on_token_grid(x, fn, layout):
    """Apply fn to the image tokens of x laid out as a (B, C, F, h, w) video; extra tokens pass through."""
    v, pre, post = tokens_to_video(x, layout)
    y = fn(v)
    if not isinstance(y, torch.Tensor) or y.shape != v.shape:
        raise ValueError(f"{LOG_TAG} a bend on the token grid must keep its shape {tuple(v.shape)}, "
                         f"got {tuple(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__}")
    return video_to_tokens(y.to(x.dtype), pre, post)


def parse_step_str_to_ranges(s, max_steps=1000):
    default_value = None
    if not isinstance(s, str):
        return default_value

    s = s.strip()
    if not s or s == "*":
        return default_value

    result = set()
    s_nospace = s.replace(" ", "")
    parts = s_nospace.split(",")

    try:
        for p in parts:
            if "-" in p:
                left, right = p.split("-")

                # "-X" => 0..X
                if left == "":
                    end = int(right)
                    result.update(range(0, min(end, max_steps) + 1))

                # "X-" => X..max_steps
                elif right == "":
                    start = int(left)
                    result.update(range(max(start, 0), max_steps + 1))

                # "X-Y" => X..Y
                else:
                    start = int(left)
                    end = int(right)
                    if start <= end:
                        result.update(range(max(start, 0), min(end, max_steps) + 1))
                    else:
                        raise ValueError("Invalid range order")

            else:
                # Single number
                num = int(p)
                if 0 <= num <= max_steps:
                    result.add(num)
                else:
                    raise ValueError("Number out of bounds")

    except Exception:
        # On ANY parsing error -> treat as "*"
        return default_value

    return sorted(result) if result else default_value

# Define the functions that will be used in the latent bending operations
operations = {
    "parse_step_str_to_ranges": parse_step_str_to_ranges,
    "add_full": add_full,
    "add_sparse": add_sparse,
    "add_noise": add_noise,
    "add_normal": add_normal,
    "multiply_scalar": multiply_scalar,
    "rotate_x": rotate_x,
    "rotate_y": rotate_y,
    "rotate_z": rotate_z,
    "rotate_image": rotate_image,
    "threshold": threshold,
    "soft_threshold": soft_threshold,
    "inversion": inversion,
    "reflect": reflect,
    "absolute": absolute,
    "log": log,
    "clamp": clamp,
    "scale": scale,
    "scale_image": scale_image,
    "gradient": gradient,
    "dilation": dilation,
    "erosion": erosion,
    "sobel": sobel,
    "hadamard1": hadamard1
}
