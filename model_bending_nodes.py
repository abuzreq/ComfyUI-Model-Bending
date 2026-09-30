# Standalone model bending nodes (inspector, PCA/HSpace, latent ops, VAE/conditioning bending, etc.).
# Uses shared bendutils and bending_modules from this package.
import copy
import json
import math
import re
from typing import Optional

import torch
import torch.nn as nn

import comfy.model_management
import comfy.utils
from server import PromptServer

from .bendutils import (
    operations,
    get_model_tree,
    process_path,
    parse_step_str_to_ranges,
    resolve_hook_target,
    make_bending_unet_wrapper,
    current_step,
    normalized_t,
    model_sampling_of,
    t_window_from,
    in_window,
    blend_and_guard,
    infer_token_grid,
    tokens_to_grid,
    grid_to_tokens,
    warn,
    with_model_sampling,
    LOG_TAG,
)
from .bending_modules import (
    GatedBendingModule,
    AddNoiseModule,
    AddScalarModule,
    FourierAmplifyModule,
    MultiplyScalarModule,
    ThresholdModule,
    RotateModule,
    ScaleModule,
    ErosionModule,
    DilationModule,
    GradientModule,
    SobelModule,
    ApplyToRandomSubsetModule,
)

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from tqdm import trange
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    PCA = None
    StandardScaler = None
    trange = None


# Helpers used by model bending nodes (avoid circular import from nodes.py)
def _get_unet_from_comfy_model(m) -> Optional[nn.Module]:
    if hasattr(m, "model") and hasattr(m.model, "diffusion_model"):
        return m.model.diffusion_model
    if hasattr(m, "stream") and hasattr(m.stream, "unet"):
        return m.stream.unet
    return None


def _set_unet_on_comfy_model(m, unet: nn.Module) -> None:
    if hasattr(m, "model") and hasattr(m.model, "diffusion_model"):
        m.model.diffusion_model = unet
    elif hasattr(m, "stream") and hasattr(m.stream, "unet"):
        m.stream.unet = unet


# Noise generators for IntermediateOutputNode / PCAPrep
class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent.get("batch_index") if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


def _select_center_child(middle_block):
    children = list(middle_block.children())
    num_children = len(children)
    if num_children == 1:
        return children[0]
    elif num_children == 2:
        return children[1]
    elif num_children == 3:
        return children[1]
    else:
        raise ValueError(f"Unexpected number of children ({num_children}).")


# ---------------------------------------------------------------------------
# Node: Visualize Feature Map (IntermediateOutputNode)
# ---------------------------------------------------------------------------
class IntermediateOutputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "layer_path": ("STRING",),
                "timestep": ("FLOAT", {"default": 0.0, "tooltip": "Denoising step to show, 1-based (0 = last step)"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "latent_image": ("LATENT",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("UNet Output", "Feature Map")
    FUNCTION = "process"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)
        noise_mask = latent.get("noise_mask") if "noise_mask" in latent else None
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_image, noise_mask=noise_mask, callback=None, disable_pbar=True, seed=noise_seed
        )
        out = latent.copy()
        out["samples"] = samples
        return (out, out)

    def process(self, model, layer_path, timestep, noise_seed, latent_image, cfg, positive, negative, sampler, sigmas):
        m = model.clone()
        unet = _get_unet_from_comfy_model(m)
        if unet is None:
            warn("Visualize Feature Map: could not find the diffusion model; returning zeros.")
            batch_size = latent_image["samples"].shape[0]
            return (torch.zeros(batch_size, 1, 1, 1), torch.zeros(batch_size, 64, 64, 3))

        mod_path = "" if layer_path is None or layer_path == "" else process_path(layer_path)
        captures = {}  # step -> activation of the conditional half, captured after any upstream bends
        state = {"latent_shape": None}

        if mod_path:
            _, target_module = resolve_hook_target(unet, mod_path)
            if target_module is not None:
                prev_wrapper = m.model_options.get("model_function_wrapper")

                def capture_wrapper(apply_model, params):
                    c = params["c"] or {}
                    transformer_options = c.get("transformer_options") or {}
                    step = current_step(transformer_options)
                    cond_or_uncond = params.get("cond_or_uncond") or [0]
                    state["latent_shape"] = tuple(params["input"].shape)

                    def _hook(module, args, kwargs, output):
                        out = output[0] if isinstance(output, (tuple, list)) else output
                        if not isinstance(out, torch.Tensor) or out.shape[0] % len(cond_or_uncond) != 0:
                            return None
                        chunks = out.chunk(len(cond_or_uncond))
                        conds = [ch for ch, cu in zip(chunks, cond_or_uncond) if cu == 0]
                        if not conds:  # an uncond-only pass: nothing to show
                            return None
                        key = step if step is not None else len(captures)
                        if key not in captures:
                            act = torch.cat(conds)
                            img_slice = transformer_options.get("img_slice")  # DiT single blocks: [txt; img]
                            if act.ndim == 3 and img_slice and act.shape[1] == img_slice[1] and 0 < img_slice[0] < act.shape[1]:
                                act = act[:, img_slice[0]:]
                            captures[key] = act.detach().float().cpu()
                        return None

                    # Register innermost (after the upstream bends' hooks) so the map shows the bent activation.
                    def inner_apply(x, t, **kw):
                        handle = target_module.register_forward_hook(_hook, with_kwargs=True)
                        try:
                            return apply_model(x, t, **kw)
                        finally:
                            handle.remove()

                    with_model_sampling(inner_apply, apply_model)
                    if prev_wrapper is not None:
                        return prev_wrapper(inner_apply, params)
                    return inner_apply(params["input"], params["timestep"], **c)

                m.set_model_unet_function_wrapper(capture_wrapper)

        batch_size = latent_image["samples"].shape[0]
        with torch.no_grad():
            final_output = self.sample(m, True, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
            final_output = final_output[0]["samples"]

        combined_features = torch.zeros((batch_size, 64, 64, 3))
        if captures:
            steps = sorted(captures)
            wanted = int(timestep)
            step = steps[-1] if wanted <= 0 else steps[min(len(steps) - 1, wanted - 1)]
            combined_features = self._to_rgb_maps(captures[step], state["latent_shape"])
        return (final_output.permute(0, 2, 3, 1), combined_features)

    @staticmethod
    def _to_rgb_maps(act, latent_shape):
        """Channel-mean map per batch item, min-max normalised, as RGB images at the pixel resolution."""
        if act.ndim == 3:  # (B, L, C) transformer tokens
            grid = infer_token_grid(act.shape[1], latent_shape)
            if grid is None:
                side = int(math.isqrt(act.shape[1]))
                grid = (1, side, act.shape[1] // max(1, side))
            act, _ = tokens_to_grid(act, grid)
        elif act.ndim == 5:  # (B, C, T, H, W): show the first frame
            act = act[:, :, 0]
        if act.ndim != 4:
            return torch.zeros((1, 64, 64, 3))
        maps = act.mean(dim=1, keepdim=True)
        lo = maps.amin(dim=(2, 3), keepdim=True)
        hi = maps.amax(dim=(2, 3), keepdim=True)
        maps = (maps - lo) / (hi - lo).clamp_min(1e-8)
        if latent_shape is not None and len(latent_shape) >= 4:
            size = (int(latent_shape[-2]) * 8, int(latent_shape[-1]) * 8)
            maps = torch.nn.functional.interpolate(maps, size=size, mode="nearest")
        return maps.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).clamp(0, 1)


# ---------------------------------------------------------------------------
# Model / VAE Inspectors
# ---------------------------------------------------------------------------
class ShowModelStructure:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), "path_placeholder": ("STRING",)}}
    RETURN_TYPES = ("STRING", "MODEL")
    FUNCTION = "show"
    CATEGORY = "model_bending"

    def show(self, model, path_placeholder):
        tree = None
        if hasattr(model, "model"):
            tree = get_model_tree(model.model)
        elif hasattr(model, "stream"):
            tree = get_model_tree(model.stream.unet)
        if tree is None:
            raise ValueError("Model structure not found.")
        PromptServer.instance.send_sync("model_bending.inspect_model", {"tree": json.dumps(tree)})
        return (path_placeholder, model)


class ShowVAEModelStructure:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",), "path_placeholder": ("STRING",)}}
    RETURN_TYPES = ("STRING", "VAE")
    FUNCTION = "show"
    CATEGORY = "model_bending"

    def show(self, vae, path_placeholder):
        tree = get_model_tree(vae.patcher.model)
        if tree is None:
            raise ValueError("Model structure not found.")
        PromptServer.instance.send_sync("model_bending.inspect_model", {"tree": json.dumps(tree)})
        return (path_placeholder, vae)


# ---------------------------------------------------------------------------
# LoRA Bending
# ---------------------------------------------------------------------------
class LoRABending:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),) if folder_paths else ("lora",),
                "bending_module": ("BENDING_MODULE",),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, clip, lora_name, bending_module):
        if folder_paths is None:
            raise RuntimeError("folder_paths not available")
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            lora = self.loaded_lora[1]
        else:
            self.loaded_lora = None
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        lora = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(lora, key_map)
        weight_adapter = getattr(comfy, "weight_adapter", None)
        for k, v in list(loaded.items()):
            if weight_adapter is not None and isinstance(v, weight_adapter.WeightAdapterBase):
                # ComfyUI uses WeightAdapter (e.g. LoRAAdapter) with .weights = (up, down, alpha, mid, dora_scale, reshape)
                w = v.weights
                bent_0 = bending_module(w[0].clone().detach().to(device=w[0].device, dtype=w[0].dtype))
                new_weights = (bent_0,) + w[1:]
                loaded[k] = type(v)(v.loaded_keys, new_weights)
            elif isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[1], (tuple, list)) and len(v[1]) >= 1:
                # Legacy format: ("diff", (t,)) or ("lora", (up, down, ...))
                tag, tail = v[0], v[1]
                loaded[k] = (tag, (bending_module(tail[0].clone().detach()),) + tuple(tail[1:]))
        if model is not None:
            new_modelpatcher = model.clone()
            k = new_modelpatcher.add_patches(loaded, strength_model=1)
        else:
            k = ()
            new_modelpatcher = None
        if clip is not None:
            new_clip = clip.clone()
            k1 = new_clip.add_patches(loaded, 1)
        else:
            k1 = ()
            new_clip = None
        for x in loaded:
            if x not in set(k) and x not in set(k1):
                warn("LoRA Bending: key not loaded %s", x)
        return (new_modelpatcher, new_clip)


# ---------------------------------------------------------------------------
# LoRA Bending (list) – bend a single LoRA component (one key from loaded)
# ---------------------------------------------------------------------------
def _make_bent_lora_value(v, bending_module, weight_adapter):
    """Return bent version of one loaded value (adapter or legacy tuple)."""
    if weight_adapter is not None and isinstance(v, weight_adapter.WeightAdapterBase):
        w = v.weights
        bent_0 = bending_module(w[0].clone().detach().to(device=w[0].device, dtype=w[0].dtype))
        new_weights = (bent_0,) + w[1:]
        return type(v)(v.loaded_keys, new_weights)
    if isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[1], (tuple, list)) and len(v[1]) >= 1:
        tag, tail = v[0], v[1]
        return (tag, (bending_module(tail[0].clone().detach()),) + tuple(tail[1:]))
    return v


def _parse_comma_separated_ints(s):
    """Parse '0, 2, 5' or '0' into list of ints; skip invalid tokens."""
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _parse_comma_separated_keys(s):
    """Parse comma-separated key strings; return list of non-empty stripped keys."""
    return [k.strip() for k in (s or "").split(",") if k.strip()]


class LoRABendingList:
    """
    Load a LoRA and bend one or more of its components.
    component_index: comma-separated indices (e.g. "0, 2, 5"). component_key: comma-separated
    weight keys (optional). If component_key is non-empty, only those keys are bent; else indices
    are used. Output all_component_keys lists every key for reference.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),) if folder_paths else ("lora",),
                "component_index": ("STRING", {"default": "0", "multiline": False}),
                "bending_module": ("BENDING_MODULE",),
            },
            "optional": {
                "component_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "all_component_keys")
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, clip, lora_name, component_index, bending_module, component_key=None):
        if folder_paths is None:
            raise RuntimeError("folder_paths not available")
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            lora = self.loaded_lora[1]
        else:
            self.loaded_lora = None
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        lora = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(lora, key_map)
        sorted_keys = sorted(loaded.keys())
        weight_adapter = getattr(comfy, "weight_adapter", None)
        keys_to_bend = []
        key_candidates = _parse_comma_separated_keys(component_key or "")
        if key_candidates:
            for k in key_candidates:
                if k in loaded:
                    keys_to_bend.append(k)
                else:
                    warn("LoRA Bending (list): component_key %r not in LoRA; skipped.", k)
        else:
            for idx in _parse_comma_separated_ints(component_index or "0"):
                if 0 <= idx < len(sorted_keys):
                    keys_to_bend.append(sorted_keys[idx])
            if not keys_to_bend and sorted_keys:
                warn(
                    "LoRA Bending (list): no valid component_index in %r (LoRA has %d components); none bent.",
                    component_index,
                    len(sorted_keys),
                )
        for key_to_bend in keys_to_bend:
            loaded[key_to_bend] = _make_bent_lora_value(
                loaded[key_to_bend], bending_module, weight_adapter
            )
        if model is not None:
            new_modelpatcher = model.clone()
            k = new_modelpatcher.add_patches(loaded, strength_model=1)
        else:
            k = ()
            new_modelpatcher = None
        if clip is not None:
            new_clip = clip.clone()
            k1 = new_clip.add_patches(loaded, 1)
        else:
            k1 = ()
            new_clip = None
        for x in loaded:
            if x not in set(k) and x not in set(k1):
                warn("LoRA Bending: key not loaded %s", x)
        all_keys_str = "\n".join(f"{i}: {key}" for i, key in enumerate(sorted_keys)) if sorted_keys else ""
        return (new_modelpatcher, new_clip, all_keys_str)


# ---------------------------------------------------------------------------
# NoiseVariations
# ---------------------------------------------------------------------------
class NoiseVariations:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"input": ("LATENT",), "scale": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, input, scale):
        input["samples"] = input["samples"] + torch.randn_like(input["samples"]) * scale
        return (input,)


# ---------------------------------------------------------------------------
# PCA / HSpace (optional on sklearn/tqdm)
# ---------------------------------------------------------------------------
class PCAPrep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "n_batches": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("PCS", "Samples")
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image
        noise = Noise_EmptyNoise().generate_noise(latent) if not add_noise else Noise_RandomNoise(noise_seed).generate_noise(latent)
        noise_mask = latent.get("noise_mask") if "noise_mask" in latent else None
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_image, noise_mask=noise_mask, callback=None, disable_pbar=True, seed=noise_seed
        )
        out = latent.copy()
        out["samples"] = samples
        return (out, out)

    def extract_h_spaces(self, m, cfg, positive, negative, sampler, sigmas, latent_image, device, n_batches):
        m.model = m.model.to(device=device)
        h_space = []
        outputs = []
        center_block_in_mid = _select_center_child(m.model.diffusion_model.middle_block)

        def get_h_space(module, inp, output):
            h_space[-1].append(output.detach().cpu())

        hook = center_block_in_mid.register_forward_hook(get_h_space)
        try:
            with torch.no_grad():
                _iter = trange(n_batches, desc="Extracting h-space batches") if (trange is not None) else range(n_batches)
                for i in _iter:
                    h_space.append([])
                    o = self.sample(m, True, i, cfg, positive, negative, sampler, sigmas, latent_image)
                    outputs.append(o[0])
        finally:
            hook.remove()
        if not h_space or not any(h_space):
            raise RuntimeError("No h-space data collected. Ensure tqdm is installed: pip install tqdm")
        h_space_tensor = torch.cat([torch.stack(batch, dim=1) for batch in h_space])
        flat = h_space_tensor.view(h_space_tensor.size(0), -1)
        return flat.numpy(), h_space_tensor.shape[1:], outputs

    def patch(self, model, cfg, positive, negative, sampler, sigmas, latent_image, n_batches):
        if not _SKLEARN_AVAILABLE or StandardScaler is None or PCA is None:
            raise RuntimeError("Compute PCA requires scikit-learn (and tqdm for progress). Install: pip install scikit-learn tqdm")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        flat_data, feature_shape, outputs = self.extract_h_spaces(
            model, cfg, positive, negative, sampler, sigmas, latent_image, device, n_batches
        )
        scaler = StandardScaler()
        n_components = min(10, n_batches)
        pca = PCA(n_components=n_components)
        pca.fit(scaler.fit_transform(flat_data))
        pcs = torch.tensor(pca.components_, dtype=model.model_dtype(), device=device)
        pcs = pcs.view(n_components, *feature_shape)
        pcs_obj = {"samples": pcs}
        combined_tensor = torch.cat([obj["samples"] for obj in outputs], dim=0)
        combined_object = {"samples": combined_tensor}
        return (pcs_obj, combined_object)


class HBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pcs": ("LATENT",),
                "direction": ("INT", {"default": 0, "min": 0, "max": 9}),
                "scale": ("FLOAT", {"default": 1.0}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def patch(self, model, pcs, direction, scale):
        m = model.clone()
        unet = _get_unet_from_comfy_model(m)
        if unet is None:
            return (model,)

        # Target module on the shared underlying UNet (same object seen by all clones).
        mod = getattr(unet.middle_block, "1")

        prev_wrapper = m.model_options.get("model_function_wrapper")
        num_pc_steps = pcs["samples"].shape[1]

        def hbending_wrapper(apply_model, params):
            inp = params["input"]
            timestep = params["timestep"]
            c = params["c"]
            # Step from the sigma schedule, so reused/cached models and split cond/uncond passes stay aligned.
            step = current_step((c or {}).get("transformer_options")) or 0
            step = min(step, num_pc_steps - 1)

            def hook_project(module, hook_inp, output):
                change = pcs["samples"][direction, step, :, :, :] * scale
                return output + change.unsqueeze(0).to(output.device, output.dtype)

            handle = mod.register_forward_hook(hook_project)
            try:
                if prev_wrapper is not None:
                    return prev_wrapper(apply_model, params)
                return apply_model(inp, timestep, **c)
            finally:
                handle.remove()

        m.set_model_unet_function_wrapper(hbending_wrapper)
        return (m,)


# ---------------------------------------------------------------------------
# SD Model Bending / Custom Model Bending
# ---------------------------------------------------------------------------

class SDBlockModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE",),
                "block_type": (["input_blocks", "middle_block", "output_blocks"], {"default": "input_blocks"}),
                "block_index": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, bending_module, block_type, block_index):
        if not (hasattr(model, "clone") and callable(getattr(model, "clone", None))):
            raise RuntimeError("Model has no clone() method; cannot patch for bending.")

        m = model.clone()
        unet = _get_unet_from_comfy_model(m)
        if unet is None:
            return (model,)

        target_container = getattr(unet, block_type)
        if block_type != "middle_block":
            if block_index >= len(target_container):
                block_index = len(target_container) - 1
            path_suffix = f"{block_type}.{block_index}"
        else:
            path_suffix = "middle_block"

        full_path = process_path(f"diffusion_model.{path_suffix}")

        prev_wrapper = m.model_options.get("model_function_wrapper")
        m.set_model_unet_function_wrapper(
            make_bending_unet_wrapper(unet, [full_path], bending_module, prev_wrapper)
        )
        return (m,)
    
class SDModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE",),
                "block": (["input_blocks", "middle_block", "output_blocks"], {"default": "input_blocks"}),
                "layer_num": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def find_conv2d_modules(self, model, parent_name=""):
        conv_layers = []
        for name, module in model.named_modules():
            full_path = f"{parent_name}.{name}" if parent_name else name
            if isinstance(module, nn.Conv2d):
                conv_layers.append((full_path, module))
        return conv_layers

    def patch(self, model, bending_module, block, layer_num):
        if not (hasattr(model, "clone") and callable(getattr(model, "clone", None))):
            raise RuntimeError("Model has no clone() method; cannot patch for bending.")
        m = model.clone()
        unet = _get_unet_from_comfy_model(m)
        if unet is None:
            return (model,)
        convs = self.find_conv2d_modules(getattr(unet, block))
        PromptServer.instance.send_sync("model_bending.bend_sd_model", {"num_layers": len(convs)})
        if layer_num >= len(convs):
            layer_num = max(0, len(convs) - 1)
        path_to_module, _ = convs[layer_num]
        mod_path = "" if not path_to_module else process_path("diffusion_model." + block + "." + path_to_module)
        if not mod_path:
            return (model,)
        prev_wrapper = m.model_options.get("model_function_wrapper")
        m.set_model_unet_function_wrapper(
            make_bending_unet_wrapper(unet, [mod_path], bending_module, prev_wrapper)
        )
        return (m,)


class CustomModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE",),
                "path": ("STRING", {"default": ""}),
            },
            "optional": {
                "steps_to_bend_str": ("STRING", {"default": "*"}),
                "max_denoising_steps": ("INT", {"default": 200}),
                "t_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Bend only while diffusion time t <= t_start (1 = pure noise)"}),
                "t_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Bend only while diffusion time t >= t_end (0 = clean image)"}),
                "strict": ("BOOLEAN", {"default": False,
                                       "tooltip": "Fail on paths that cannot be bent instead of skipping them"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, bending_module, path, steps_to_bend_str="*", max_denoising_steps=200,
              t_start=1.0, t_end=0.0, strict=False):
        if not path or path == "False" or not isinstance(path, str):
            return (model,)

        m = model.clone()
        unet = _get_unet_from_comfy_model(m)
        if unet is None:
            return (model,)

        split_paths = [p.strip() for p in path.split(",") if p.strip()]
        mod_paths = [process_path(p) for p in split_paths if p]
        mod_paths = [mp for mp in mod_paths if mp]

        steps_to_bend = parse_step_str_to_ranges(steps_to_bend_str, max_steps=max_denoising_steps)

        prev_wrapper = m.model_options.get("model_function_wrapper")
        wrapper = make_bending_unet_wrapper(unet, mod_paths, bending_module, prev_wrapper,
                                            steps_to_bend=steps_to_bend, t_window=t_window_from(t_start, t_end),
                                            strict=strict)
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


# ---------------------------------------------------------------------------
# Timestep-gated bending (diffusion-time windows with optional ramps)
# ---------------------------------------------------------------------------
class TimestepGatedBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bending_module": ("BENDING_MODULE",),
                "t_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Window start in diffusion time (1 = pure noise)"}),
                "t_end": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Window end in diffusion time (0 = clean image)"}),
                "ramp": (["hard", "linear", "cosine"], {"default": "hard"}),
                "ramp_width": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Width of the fade-in/fade-out at each window edge"}),
            }
        }
    RETURN_TYPES = ("BENDING_MODULE",)
    FUNCTION = "gate"
    CATEGORY = "model_bending"
    DESCRIPTION = ("Limits a bend to a window of diffusion time t (1 = noise, 0 = image), independent of the step "
                   "count, scheduler or shift. Rough guide: 1-0.7 composition, 0.7-0.2 shapes/style, 0.2-0 detail.")

    def gate(self, bending_module, t_start, t_end, ramp, ramp_width):
        return (GatedBendingModule(bending_module, t_start=t_start, t_end=t_end, ramp=ramp, ramp_width=ramp_width),)


# ---------------------------------------------------------------------------
# DiT block bending (Flux, Chroma, SD3, WAN, Qwen-Image, LTX, HunyuanVideo, ...)
# ---------------------------------------------------------------------------
_BLOCK_LISTS = {
    "double": ("double_blocks", "joint_blocks", "transformer_blocks", "blocks", "layers"),
    "single": ("single_blocks",),
}


def _block_list(dm, kind):
    for name in _BLOCK_LISTS[kind]:
        blocks = getattr(dm, name, None)
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            return name, len(blocks)
    return None, None


def _parse_block_spec(spec, dm):
    """'double:0-6, single:25-37' -> [("double", 0), ..., ("single", 37)]"""
    found = re.findall(r"(double|single)\s*:\s*([0-9*,\-\s]*)", spec or "")
    if not found:
        raise ValueError(f"{LOG_TAG} blocks must look like 'double:0-6, single:25-37', got {spec!r}")
    out = []
    for kind, rng in found:
        _, count = _block_list(dm, kind)
        rng = rng.strip(" ,")
        max_index = (count - 1) if count else 127
        indices = parse_step_str_to_ranges(rng, max_steps=max_index)
        if indices is None:
            if rng not in ("", "*"):
                raise ValueError(f"{LOG_TAG} invalid block range {rng!r} for {kind}")
            indices = list(range(max_index + 1))
        out.extend((kind, i) for i in indices if i <= max_index)
    return out


class DiTBlockBending:
    """
    Bends the output of transformer blocks through ComfyUI's native block-replace patches
    (patches_replace["dit"][("double_block" | "single_block", i)]), which most DiT models support.
    The image and text token streams are bent separately; with spatial=True image tokens are laid out on
    their 2-D grid so spatial ops (rotate, scale, erosion, ...) act on the picture, not on the token list.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE",),
                "blocks": ("STRING", {"default": "double:0-3",
                                      "tooltip": "e.g. 'double:0-6, single:25-37', 'double:*'"}),
                "stream": (["img", "txt", "both"], {"default": "img"}),
                "spatial": ("BOOLEAN", {"default": True,
                                        "tooltip": "Lay image tokens out on their 2-D grid before bending"}),
            },
            "optional": {
                "t_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "t_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strict": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "report")
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, bending_module, blocks, stream, spatial, t_start=1.0, t_end=0.0, strict=False):
        m = model.clone()
        dm = _get_unet_from_comfy_model(m)
        if dm is None:
            raise ValueError(f"{LOG_TAG} DiT Block Bending: no diffusion model found")
        targets = _parse_block_spec(blocks, dm)
        report = {
            "model": type(dm).__name__,
            "block_lists": {k: dict(zip(("name", "count"), _block_list(dm, k))) for k in _BLOCK_LISTS},
            "blocks": [f"{k}:{i}" for k, i in targets],
            "stream": stream, "spatial": spatial,
        }
        if strict and not any(_block_list(dm, k)[0] for k in _BLOCK_LISTS):
            raise ValueError(f"{LOG_TAG} {type(dm).__name__} has no transformer block lists; is this a DiT model?")

        t_window = t_window_from(t_start, t_end)
        gated = getattr(bending_module, "is_gated", False)
        module = bending_module.inner if gated else bending_module
        state = {"t": None, "step": None, "latent_shape": None, "txt_len": None, "flags": []}
        fired, warned = set(), set()

        def warn_once(key, msg, *args):
            if key not in warned:
                warned.add(key)
                warn(msg, *args)

        def bend_tokens(x, which, weight, where):
            if spatial and which == "img" and x.ndim == 3:
                grid = infer_token_grid(x.shape[1], state["latent_shape"])
                if grid is not None:
                    g, rest = tokens_to_grid(x, grid)
                    y = grid_to_tokens(module(g), grid, x.shape[0], rest)
                    return blend_and_guard(x, y, weight, flags=state["flags"])
                warn_once(("grid", where), "could not infer the image-token grid at %s; bending tokens as a flat list", where)
            elif spatial and which == "txt":
                msg = "text tokens have no spatial layout; spatial ops act on the token list"
                if strict:
                    raise ValueError(f"{LOG_TAG} {msg}")
                warn_once(("txt", where), msg)
            y = module(x)
            return blend_and_guard(x, y, weight, flags=state["flags"])

        def make_patch(kind, index, previous):
            where = f"{kind}:{index}"

            def patch_fn(args, extra):
                out = previous(args, extra) if previous is not None else extra["original_block"](args)
                fired.add(where)
                if kind == "double" and args.get("txt") is not None:
                    state["txt_len"] = args["txt"].shape[1]
                if needs_t and state["t"] is None:  # fail closed: no diffusion time, no windowed bend
                    return out
                if not in_window(state["step"], state["t"], None, t_window):
                    return out
                weight = bending_module.weight(state["t"]) if gated else 1.0
                if weight == 0.0:
                    return out
                out = dict(out)
                if kind == "single":
                    joint = out["img"]
                    to = args.get("transformer_options") or {}
                    txt_len = (to.get("img_slice") or [state["txt_len"]])[0]
                    if txt_len is None:
                        warn_once(("split", where), "cannot split text/image tokens at %s; bending the joint stream", where)
                        out["img"] = bend_tokens(joint, "joint", weight, where)
                        return out
                    txt, img = joint[:, :txt_len], joint[:, txt_len:]
                    if stream in ("img", "both"):
                        img = bend_tokens(img, "img", weight, where)
                    if stream in ("txt", "both"):
                        txt = bend_tokens(txt, "txt", weight, where)
                    out["img"] = torch.cat((txt, img), dim=1)
                else:
                    if stream in ("img", "both") and isinstance(out.get("img"), torch.Tensor):
                        out["img"] = bend_tokens(out["img"], "img", weight, where)
                    if stream in ("txt", "both") and isinstance(out.get("txt"), torch.Tensor):
                        out["txt"] = bend_tokens(out["txt"], "txt", weight, where)
                return out
            return patch_fn

        existing = m.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for kind, index in targets:
            key = f"{kind}_block"
            m.set_model_patch_replace(make_patch(kind, index, existing.get((key, index))), "dit", key, index)

        prev_wrapper = m.model_options.get("model_function_wrapper")
        needs_t = t_window is not None or gated

        def wrapper(apply_model, params):
            c = params["c"] or {}
            state["step"] = current_step(c.get("transformer_options"))
            state["t"] = normalized_t(model_sampling_of(apply_model), params["timestep"]) if needs_t else None
            state["latent_shape"] = tuple(params["input"].shape)
            state["flags"] = []
            if needs_t and state["t"] is None:
                msg = "DiT Block Bending: cannot resolve diffusion time; time-windowed bends are skipped for this pass"
                if strict:
                    raise ValueError(f"{LOG_TAG} {msg}")
                warn_once("no_t", msg)
            if prev_wrapper is not None:
                out = prev_wrapper(apply_model, params)
            else:
                out = apply_model(params["input"], params["timestep"], **c)
            if state["flags"] and "nan" not in warned and bool(torch.stack(state["flags"]).any()):
                warn_once("nan", "DiT Block Bending: a bend produced NaN/Inf; replaced")
            if not fired:
                warn_once("none", "DiT Block Bending: no block patch was called; %s may not support block "
                          "replacement (patches_replace['dit'])", type(dm).__name__)
            return out

        m.set_model_unet_function_wrapper(wrapper)
        return (m, json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Layer catalogue: which paths can be bent, and how
# ---------------------------------------------------------------------------
class BendableLayerCatalogue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filter": ("STRING", {"default": "", "tooltip": "Only list paths containing this text"}),
                "max_depth": ("INT", {"default": 3, "min": 1, "max": 12}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("catalogue",)
    FUNCTION = "catalogue"
    CATEGORY = "model_bending"

    def catalogue(self, model, filter, max_depth):
        dm = _get_unet_from_comfy_model(model)
        if dm is None:
            raise ValueError(f"{LOG_TAG} no diffusion model found")
        block_lists = {}
        for kind in _BLOCK_LISTS:
            name, count = _block_list(dm, kind)
            if name:
                block_lists[name] = {"count": count, "dit_block_bending": f"{kind}:0-{count - 1}"}
        dit_lists = set(block_lists)
        entries = []
        for path, mod in dm.named_modules():
            if not path or path.count(".") >= max_depth or (filter and filter not in path):
                continue
            entry = {"path": path, "class": type(mod).__name__}
            parent, _, leaf = path.rpartition(".")
            if isinstance(mod, nn.ModuleList):
                entry.update(hookable=False, note=f"ModuleList is never called; use {path}.<i>")
            elif any(cls.__name__ == "TimestepEmbedSequential" for cls in type(mod).__mro__):
                hooked, _ = resolve_hook_target(dm, path)
                entry.update(hookable=True, note=f"container; bends apply to its last child {hooked}")
            elif parent in dit_lists and leaf.isdigit():
                kind = "single" if parent == "single_blocks" else "double"
                entry.update(hookable=True, note=f"transformer block; use DiT Block Bending with '{kind}:{leaf}' "
                                                 "(direct hooks bend only the first output)")
            else:
                entry["hookable"] = True
                if not any(True for _ in mod.children()):
                    entry["leaf"] = True
            entries.append(entry)
        return (json.dumps({"model": type(dm).__name__, "block_lists": block_lists, "layers": entries}, indent=1),)


# ---------------------------------------------------------------------------
# BENDING_MODULE factory nodes
# ---------------------------------------------------------------------------
class BaseModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    RETURN_TYPES = ("BENDING_MODULE",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self):
        pass


class ApplyToRandomSubsetModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bending_module": ("BENDING_MODULE",),
                "percentage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "dimension": (["batch", "channel", "spatial"], {"default": "batch"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def patch(self, bending_module, percentage, dimension, seed):
        wrapped = ApplyToRandomSubsetModule(module=bending_module, percentage=percentage, seed=seed, dim=dimension)
        return (wrapped,)


class AddNoiseModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"noise_std": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})}}

    def patch(self, noise_std, seed):
        return (AddNoiseModule(noise_std=noise_std, seed=seed),)


class AddScalarModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scalar": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0})}}

    def patch(self, scalar):
        return (AddScalarModule(scalar=scalar),)


class MultiplyScalarModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scalar": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0})}}

    def patch(self, scalar):
        return (MultiplyScalarModule(scalar=scalar),)


class ThresholdModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"threshold": ("FLOAT", {"default": 0.0})}}

    def patch(self, threshold):
        return (ThresholdModule(threshold=threshold),)


#WIP: Attempting to implmenet "Enhancing Creative Generation on Stable Diffusion-based Models"
class FourierModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cutoff_freq": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.25}), "amp_factor": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.25})}}

    def patch(self, cutoff_freq, amp_factor):
        return (FourierAmplifyModule(cutoff_freq=cutoff_freq, amp_factor=amp_factor),)


class RotateModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"angle_degrees": ("FLOAT", {"default": 0.0, "min": -360, "max": 360, "step": 0.01})}}

    def patch(self, angle_degrees):
        return (RotateModule(angle_degrees=angle_degrees),)


class ScaleModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scale_factor": ("FLOAT", {"default": 1.0, "min": -100, "max": 100})}}

    def patch(self, scale_factor):
        return (ScaleModule(scale_factor=scale_factor),)


class ErosionModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"kernel_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})}}

    def patch(self, kernel_size):
        return (ErosionModule(kernel_size=kernel_size),)


class DilationModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"kernel_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})}}

    def patch(self, kernel_size):
        return (DilationModule(kernel_size=kernel_size),)


class GradientModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"kernel_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})}}

    def patch(self, kernel_size):
        return (GradientModule(kernel_size=kernel_size),)


class SobelModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"normalized": ("BOOLEAN", {"default": True})}}

    def patch(self, normalized):
        return (SobelModule(normalized=normalized),)


# ---------------------------------------------------------------------------
# Latent Operation To Module / LatentApplyBendingOperationCFG
# ---------------------------------------------------------------------------
class LatentOperationToModule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"operation": ("LATENT_OPERATION",)}}
    RETURN_TYPES = ("BENDING_MODULE",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def patch(self, operation):
        class BendingModuleWrapper(nn.Module):
            def forward(self, image):
                return operation(image)
        return (BendingModuleWrapper(),)


class LatentApplyBendingOperationCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), "operation": ("LATENT_OPERATION",), "step": ("INT", {"default": 0, "min": 0})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, operation, step):
        @torch.no_grad()
        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            transformer_options = args.get("model_options", {}).get("transformer_options", {})
            sigmas = transformer_options.get("sample_sigmas")
            if sigmas is not None:
                sigmas = sigmas.to(device="cpu")
                step_num = (sigmas == args["sigma"].cpu()).nonzero(as_tuple=True)[0]
                if step_num.nelement() > 0 and step_num[0] == step:
                    if len(conds_out) == 2:
                        conds_out[0] = operation(latent=(conds_out[0] - conds_out[1])) + conds_out[1]
                    else:
                        conds_out[0] = operation(latent=conds_out[0])
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m,)


# ---------------------------------------------------------------------------
# Latent operations (return LATENT_OPERATION)
# ---------------------------------------------------------------------------
class BaseLatentOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"
    CATEGORY = "model_bending"

    def op(self):
        pass


class LatentOperationMultiplyScalar(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scalar": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})}}

    def op(self, scalar):
        return (lambda latent: latent * scalar,)


class LatentOperationAddScalar(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"scalar": ("FLOAT", {"default": 0.0})}}

    def op(self, scalar):
        return (lambda latent: latent + scalar,)


class LatentOperationThreshold(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"threshold": ("FLOAT", {"default": 0.0})}}

    def op(self, threshold):
        return (operations["threshold"](threshold),)


class LatentOperationAddNoise(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"std": ("FLOAT", {"default": 0.05})}}

    def op(self, std):
        return (operations["add_noise"](std),)


class LatentOperationRotate(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"axis": (["x", "y", "z"], {}), "angle": ("FLOAT", {"default": 0.0})}}

    def op(self, axis, angle):
        def rotate(latent):
            if axis == "x":
                return operations["rotate_x"](angle)(latent)
            elif axis == "y":
                return operations["rotate_y"](angle)(latent)
            else:
                return operations["rotate_z"](angle)(latent)
        return (rotate,)


class LatentOperationGeneric(BaseLatentOperation):
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"operation": (list(operations.keys()), {})},
            "optional": {
                "float_param": ("FLOAT", {"default": 0.0}),
                "float_param2": ("FLOAT", {"default": 0.0}),
                "int_param": ("INT", {"default": 5}),
                "int_param2": ("INT", {"default": 5}),
                "bool_param": ("BOOLEAN", {"default": False}),
            },
        }

    def op(self, operation, float_param=0.0, float_param2=0.0, int_param=5, int_param2=5, bool_param=False):
        int_operations = ["reflect", "dilation", "erosion"]
        none_operations = ["absolute", "log", "gradient", "hadamard1"]
        dual_float_operations = ["clamp", "scale"]
        bool_operations = ["sobel"]

        def process(latent):
            if operation in int_operations:
                return operations[operation](int_param)(latent)
            elif operation in none_operations:
                return operations[operation]()(latent)
            elif operation in bool_operations:
                return operations[operation](bool_param)(latent)
            elif operation in dual_float_operations:
                return operations[operation](float_param, float_param2)(latent)
            else:
                return operations[operation](float_param)(latent)
        return (process,)


# ---------------------------------------------------------------------------
# VAE Bending / Conditioning Apply
# ---------------------------------------------------------------------------
class CustomModuleVAEBending:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",), "path": ("STRING",), "bending_module": ("BENDING_MODULE",)}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, vae, path, bending_module):
        mod_path = "" if not path else process_path(path, ["AutoencoderKL", "TAESD"])
        if not mod_path:
            return (vae,)

        # Shallow-copy the VAE Python wrapper so the caller gets a distinct object
        # without duplicating any weight tensors.
        m = copy.copy(vae)
        # Give it a cloned patcher so object_patches are isolated to this copy.
        m.patcher = vae.patcher.clone()

        # Build a thin wrapper: run the original submodule then apply bending_module.
        original_submod = comfy.utils.get_attr(m.patcher.model, mod_path)
        bent_submod = nn.Sequential(original_submod, bending_module)
        # add_object_patch installs bent_submod during patch_model() and restores
        # the original in unpatch_model() — no weight duplication.
        m.patcher.add_object_patch(mod_path, bent_submod)
        return (m,)


class ConditioningApplyOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"cond": ("CONDITIONING",), "operation": ("LATENT_OPERATION",), "zero_out": ("BOOLEAN",)}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def patch(self, cond, operation, zero_out):
        c = []
        for t in cond:
            d = t[1].copy()
            if zero_out:
                pooled_output = d.get("pooled_output")
                if pooled_output is not None:
                    d["pooled_output"] = torch.zeros_like(pooled_output)
            n = [operation(t[0]), d]
            c.append(n)
        return (c,)


# ---------------------------------------------------------------------------
# NODE_CLASS_MAPPINGS for model_bending_nodes
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Compute PCA": PCAPrep,
    "HSpace Bending": HBending,
    "NoiseVariations": NoiseVariations,
    "Latent Operation To Module": LatentOperationToModule,
    "Model Bending": CustomModelBending,
    "Model Bending (SD Layers)": SDModelBending,
    "Model Bending (SD Blocks)": SDBlockModelBending,
    "Model VAE Bending": CustomModuleVAEBending,
    "Model Inspector": ShowModelStructure,
    "Model VAE Inspector": ShowVAEModelStructure,
    "Apply To Subset (Bending)": ApplyToRandomSubsetModelBending,
    "Add Noise Module (Bending)": AddNoiseModelBending,
    "Add Scalar Module (Bending)": AddScalarModelBending,
    "Multiply Scalar Module (Bending)": MultiplyScalarModelBending,
    "Threshold Module (Bending)": ThresholdModelBending,
    "Fourier Amplify Module (Bending)": FourierModelBending,
    "Rotate Module (Bending)": RotateModelBending,
    "Scale Module (Bending)": ScaleModelBending,
    "Erosion Module (Bending)": ErosionModelBending,
    "Gradient Module (Bending)": GradientModelBending,
    "Dilation Module (Bending)": DilationModelBending,
    "Sobel Module (Bending)": SobelModelBending,
    "LoRA Bending": LoRABending,
    "LoRA Bending (list)": LoRABendingList,
    "Visualize Feature Map": IntermediateOutputNode,
    "LatentApplyOperationCFGToStep": LatentApplyBendingOperationCFG,
    "Latent Operation (Multiply Scalar)": LatentOperationMultiplyScalar,
    "Latent Operation (Add Scalar)": LatentOperationAddScalar,
    "Latent Operation (Threshold)": LatentOperationThreshold,
    "Latent Operation (Rotate)": LatentOperationRotate,
    "Latent Operation (Add Noise)": LatentOperationAddNoise,
    "Latent Operation (Custom)": LatentOperationGeneric,
    "ConditioningApplyOperation": ConditioningApplyOperation,
    "Timestep Gated Bending": TimestepGatedBending,
    "DiT Block Bending": DiTBlockBending,
    "Bendable Layer Catalogue": BendableLayerCatalogue,
}
