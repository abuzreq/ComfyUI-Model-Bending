# Standalone model bending nodes (inspector, PCA/HSpace, latent ops, VAE/conditioning bending, etc.).
# Uses shared bendutils and bending_modules from this package.
import copy
import json
import logging
from typing import Optional

import torch
import torch.nn as nn

import comfy.model_management
import comfy.utils
from server import PromptServer

from .bendutils import (
    operations,
    inject_module,
    hook_module,
    get_model_tree,
    process_path,
    parse_step_str_to_ranges,
    ensure_clone_has_own_inner,
)
from .bending_modules import (
    AddNoiseModule,
    AddScalarModule,
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
                "timestep": ("FLOAT", {"default": 0.0}),
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
        ensure_clone_has_own_inner(model, m)
        if _get_unet_from_comfy_model(m) is None:
            m = copy.deepcopy(model)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        m.model = m.model.to(device=device)
        mod_path = "" if layer_path is None or layer_path == "" else process_path(layer_path)
        intermediate = {"output": []}
        if mod_path != "":
            def hook(module, inputs, output):
                intermediate["output"] = [output]
            parts = mod_path.split(".")
            target_module = m.model.diffusion_model
            for part in parts:
                target_module = target_module[int(part)] if part.isdigit() else getattr(target_module, part)
            handle = target_module.register_forward_hook(hook)
        batch_size = latent_image["samples"].shape[0]
        with torch.no_grad():
            final_output = self.sample(m, True, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
            final_output = final_output[0]["samples"]
        if mod_path != "":
            handle.remove()
        combined_features = torch.zeros((batch_size, 64, 64, 4), device=device)
        intermediate_outputs = intermediate.get("output")
        if intermediate_outputs is not None:
            feature_maps = []
            for one_step_output in intermediate_outputs:
                for j in range(one_step_output.shape[0]):
                    iout = one_step_output[j]
                    gray_scale = torch.sum(iout, 0) / iout.shape[0]
                    gray_scale = gray_scale.unsqueeze(0).unsqueeze(0)
                    repeated_gray_scale = gray_scale.permute(0, 2, 3, 1).repeat(1, 1, 1, 4)
                    feature_maps.append(repeated_gray_scale)
                combined_features = torch.cat(feature_maps, dim=0)
        return (final_output.permute(0, 2, 3, 1), combined_features)


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
            ensure_clone_has_own_inner(model, new_modelpatcher)
            if _get_unet_from_comfy_model(new_modelpatcher) is None:
                new_modelpatcher = copy.deepcopy(model)
            k = new_modelpatcher.add_patches(loaded, strength_model=1)
        else:
            k = ()
            new_modelpatcher = None
        if clip is not None:
            new_clip = copy.deepcopy(clip)
            k1 = new_clip.add_patches(loaded, 1)
        else:
            k1 = ()
            new_clip = None
        for x in loaded:
            if x not in set(k) and x not in set(k1):
                logging.warning("NOT LOADED %s", x)
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
                    logging.warning("LoRA Bending (list): component_key %r not in LoRA; skipped.", k)
        else:
            for idx in _parse_comma_separated_ints(component_index or "0"):
                if 0 <= idx < len(sorted_keys):
                    keys_to_bend.append(sorted_keys[idx])
            if not keys_to_bend and sorted_keys:
                logging.warning(
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
            ensure_clone_has_own_inner(model, new_modelpatcher)
            if _get_unet_from_comfy_model(new_modelpatcher) is None:
                new_modelpatcher = copy.deepcopy(model)
            k = new_modelpatcher.add_patches(loaded, strength_model=1)
        else:
            k = ()
            new_modelpatcher = None
        if clip is not None:
            new_clip = copy.deepcopy(clip)
            k1 = new_clip.add_patches(loaded, 1)
        else:
            k1 = ()
            new_clip = None
        for x in loaded:
            if x not in set(k) and x not in set(k1):
                logging.warning("NOT LOADED %s", x)
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
        ensure_clone_has_own_inner(model, m)
        if _get_unet_from_comfy_model(m) is None:
            m = copy.deepcopy(model)

        def hook_project(module, inp, output):
            change = pcs["samples"][direction, module.step, :, :, :] * scale
            module.step += 1
            return output + change.unsqueeze(0)

        mod = getattr(m.model.diffusion_model.middle_block, "1")
        mod.step = 0
        if getattr(mod, "hspace_hook", None) is not None:
            mod.hspace_hook.remove()
        setattr(mod, "hspace_hook", mod.register_forward_hook(hook_project))
        return (m,)


# ---------------------------------------------------------------------------
# SD Model Bending / Custom Model Bending
# ---------------------------------------------------------------------------
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
        ensure_clone_has_own_inner(model, m)
        unet = _get_unet_from_comfy_model(m)
        if unet is None:
            return (model,)
        convs = self.find_conv2d_modules(getattr(unet, block))
        PromptServer.instance.send_sync("model_bending.bend_sd_model", {"num_layers": len(convs)})
        if layer_num >= len(convs):
            layer_num = max(0, len(convs) - 1)
        path_to_module, _ = convs[layer_num]
        mod_path = "" if not path_to_module else process_path("diffusion_model." + block + "." + path_to_module)
        hook_module(m.model.diffusion_model, mod_path, bending_module)
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
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, bending_module, path, steps_to_bend_str="*", max_denoising_steps=200):
        m = model.clone()
        ensure_clone_has_own_inner(model, m)
        if _get_unet_from_comfy_model(m) is None:
            m = copy.deepcopy(model)
        if not path or path == "False" or not isinstance(path, str):
            return (model,)
        split_paths = [p.strip() for p in path.split(",") if p.strip()]
        module = m.model.diffusion_model if hasattr(m, "model") else (m.stream.unet if hasattr(m, "stream") else None)
        if module is None:
            return (model,)

        def my_unet_wrapper(apply_model, params):
            inp, timestep, c = params["input"], params["timestep"], params["c"]
            transformer_options = params["c"].get("transformer_options", {})
            sigmas = transformer_options.get("sample_sigmas")
            if sigmas is not None:
                sigmas = sigmas.to(device="cpu")
                all_sigmas = transformer_options.get("sigmas")
                if all_sigmas is not None:
                    current_step = (sigmas == all_sigmas.cpu()).nonzero(as_tuple=True)[0]
                    if current_step.numel() > 0:
                        bending_module.current_step = current_step[0].item()
            return apply_model(inp, timestep, **c)

        m.set_model_unet_function_wrapper(my_unet_wrapper)
        bending_module.steps_to_bend = parse_step_str_to_ranges(steps_to_bend_str, max_steps=max_denoising_steps)
        for p in split_paths:
            mod_path = "" if not p else process_path(p)
            if mod_path:
                hook_module(module, mod_path, bending_module)
        return (m,)


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
        m = copy.deepcopy(vae)
        mod_path = "" if not path else process_path(path, ["AutoencoderKL", "TAESD"])
        if mod_path:
            hook_module(m.patcher.model, mod_path, bending_module)
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
    "Model VAE Bending": CustomModuleVAEBending,
    "Model Inspector": ShowModelStructure,
    "Model VAE Inspector": ShowVAEModelStructure,
    "Apply To Subset (Bending)": ApplyToRandomSubsetModelBending,
    "Add Noise Module (Bending)": AddNoiseModelBending,
    "Add Scalar Module (Bending)": AddScalarModelBending,
    "Multiply Scalar Module (Bending)": MultiplyScalarModelBending,
    "Threshold Module (Bending)": ThresholdModelBending,
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
}
