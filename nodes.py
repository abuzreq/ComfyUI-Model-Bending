import copy
import torch
import json

import comfy.model_management
import comfy.utils
from comfy.model_base import BaseModel
from server import PromptServer
import folder_paths

import logging
from .py.custom_code_module import CodeNode
from .py.bendutils import operations, inject_module, hook_module, get_model_tree, process_path
from .py.bending_modules import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tqdm import trange

# Custom node that outputs the intermediate result.
class IntermediateOutputNode:
    """
    A custom node that:
      - Takes a MODEL (PyTorch model),
      - A LAYER_PATH (dot-separated string),
      - An INPUT tensor,
      - A timestep (float),

    It runs the model and returns the intermediate activation from the target layer as well as the final output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),        # A PyTorch model.
                "layer_path": ("STRING",),  # e.g. "features.1"
                "input": ("LATENT",),       # Input tensor for the model.
                "timestep": ("FLOAT", {"default": 0.0}),

            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("UNet Output", "Feature Map")
    FUNCTION = "process"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    DESCRIPTION = "Runs the UNet of the model and outputs the final results (i.e. denoising results given timestep) as well as the feature map (aggregated) for the chosen layer."

    def process(self, model, layer_path, input, timestep):
        m = copy.deepcopy(model)
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        m.model = m.model.to(device=device)

        mod_path = "" if layer_path == None or layer_path == "" else process_path(
            layer_path)
        intermediate = {}
        if mod_path != "":
            def hook(module, inputs, output):
                intermediate['output'] = output

            # Navigate to the target module using the layer_path.
            parts = mod_path.split('.')
            target_module = m.model.diffusion_model
            for part in parts:
                if part.isdigit():
                    target_module = target_module[int(part)]
                else:
                    target_module = getattr(target_module, part)

            # Register the hook.
            handle = target_module.register_forward_hook(hook)

        # Preparing inputs for the model.
        x = input["samples"]
        batch_size = x.shape[0]
        timesteps = torch.tensor(
            [timestep] * batch_size, device=device, dtype=m.model_dtype())
        c_crossattn = torch.zeros(
            batch_size, 77, 768, device=device, dtype=m.model_dtype())
        x = x.type(m.model_dtype()).to(device=device)

        # forward pass to get the intermediate output
        with torch.no_grad():
            final_output = m.model.apply_model(
                x, timesteps, c_crossattn=c_crossattn)  # , transformer_options=topt

            print("final_output", final_output.shape)

        # Remove the hook.
        if mod_path != "":
            handle.remove()

        # Aggregate the weights of the intermediate output and put it in the right shape so we could visualize it
        combined_features = torch.zeros((batch_size, 64, 64, 4), device=device)
        intermediate_outputs = intermediate.get('output', None)
        if intermediate_outputs is not None:
            feature_maps = []
            for i in range(0, intermediate_outputs.shape[0]):
                iout = intermediate_outputs[i]
                gray_scale = torch.sum(iout, 0)
                gray_scale = gray_scale / iout.shape[0]
                gray_scale = gray_scale.unsqueeze(0).unsqueeze(0)
                repeated_gray_scale = gray_scale.permute(
                    0, 2, 3, 1).repeat(1, 1, 1, 4)
                feature_maps.append(repeated_gray_scale)

            combined_features = torch.cat(feature_maps, dim=0)

        return (final_output.permute(0, 2, 3, 1),   combined_features,)

class ShowModelStructure:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "path_placeholder": ("STRING",)
            }
        }
    RETURN_TYPES = ("STRING", "MODEL")
    FUNCTION = "show"
    CATEGORY = "model_bending"
    DESCRIPTION = "Pick a layer by clicking on it in this inspector."

    def show(self, model, path_placeholder):

        tree = None
        if (hasattr(model, "model")):
            # If the model has a .model attribute, use it    
            tree = get_model_tree(model.model)
        elif hasattr(model, "stream"): # to accomodate stream diffusion models
            tree = get_model_tree(model.stream.unet)
        
        if tree is None:
            raise ValueError("Model structure not found.")
        else:
            PromptServer.instance.send_sync("model_bending.inspect_model", {
                                            "tree": json.dumps(tree)})

            # with open('data.json', 'w', encoding='utf-8') as f:
            #    json.dump(tree, f, ensure_ascii=False, indent=4)
            return (path_placeholder, model)

    # @classmethod
    # def IS_CHANGED(self, model, path_placeholder):
    #    return hash(str(model.model))


class LoRABending:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "bending_module": ("BENDING_MODULE", ),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    DESCRIPTION = "Bending LoRAs. Finds all the LoRA matrices that were added to the model and applies a bending_module to them."

    def patch(self, model, clip, lora_name, bending_module):
        # m = copy.deepcopy(model)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
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
            key_map = comfy.lora.model_lora_keys_clip(
                clip.cond_stage_model, key_map)

        lora = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(lora, key_map)

        for k, v in loaded.items():
            x = v[1][0]
            loaded[k] = ('lora', (bending_module(x), *v[1][1:]))

        if model is not None:
            new_modelpatcher = copy.deepcopy(model)  # model.clone()
            k = new_modelpatcher.add_patches(loaded, strength_model=1)
        else:
            k = ()
            new_modelpatcher = None

        if clip is not None:
            new_clip = copy.deepcopy(clip)  # clip.clone()
            k1 = new_clip.add_patches(loaded, 1)  # strength_clip=1
        else:
            k1 = ()
            new_clip = None
        k = set(k)
        k1 = set(k1)
        for x in loaded:
            if (x not in k) and (x not in k1):
                logging.warning("NOT LOADED {}".format(x))

        return (new_modelpatcher, new_clip)

class NoiseVariations:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("LATENT",),
                "scale": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    DESCRIPTION = ""
    def patch(self, input, scale):
        # Create a random noise tensor with the same shape as the input
        noise = torch.randn_like(input["samples"]) * scale

        # Add the noise to the input
        input["samples"] = input["samples"] + noise

        return (input,)

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
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class PCAPrep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
                    {"model": ("MODEL",),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),

                    "n_batches": ("INT", {"default": 1, "min": 1, "max": 100}),
                     }
                }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    DESCRIPTION = ""
        
    def extract_h_spaces(self, m, cfg, positive, negative, sampler, sigmas, latent_image, device, n_batches):
        m.model = m.model.to(device=device)
       

        '''Extract h-space representations for use in PCA'''
        h_space = []

        def get_h_space(module, inp, output):
            print("hook", module.__class__.__name__, output.shape)
            # Save a detached copy of output from the middle block
            h_space[-1].append(output.detach().cpu())

        # Register the forward hook on middle_block
        hook = getattr(m.model.diffusion_model.middle_block, "1").register_forward_hook(get_h_space)

        # Next: pass in a latent sample and move it along direction of user choice
        try:
            with torch.no_grad():
                for i in trange(n_batches, desc="Extracting h-space batches"):
                    h_space.append([])
                    self.sample(m, True, i, cfg, positive, negative, sampler, sigmas, latent_image, )  # Triggers the hook
        finally:
            hook.remove()

        # stack and flatten
        h_space_tensor = torch.cat([torch.stack(batch, dim=0) for batch in h_space], dim=0)
        flat = h_space_tensor.view(h_space_tensor.size(0), -1)
        return flat.numpy(), h_space_tensor.shape[1:]  # also return shape for reshaping PCs
    
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

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=None, disable_pbar=True, seed=noise_seed)
            out = latent.copy()
            out["samples"] = samples
            return (out, out)
    
    def patch(self, model, cfg, positive, negative, sampler, sigmas, latent_image, n_batches):
    
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        flat_data, feature_shape = self.extract_h_spaces(model, cfg, positive, negative, sampler, sigmas, latent_image, device, n_batches)
       
        # Apply PCA      
        scaler = StandardScaler()
        n_components = 10
        pca = PCA(n_components=n_components)
        pca.fit(scaler.fit_transform(flat_data))
        pcs = torch.tensor(pca.components_, dtype=model.model_dtype(), device=device)
        pcs = pcs.view(n_components, *feature_shape)  # dynamic shape

        modified_input = copy.deepcopy(latent_image)
        modified_input["samples"] = pcs
        return (modified_input, )

class HBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pca": ("LATENT", ),
                "direction": ("INT", {"default": 0, "min": 0, "max": 9}),
                "scale": ("FLOAT", {"default": 1.0}),
            }

        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, pca, direction, scale):
        m = copy.deepcopy(model)
       
        def hook_project(module, inp, output):
            # Save a detached copy of output from the middle block
            output = output +  pca["samples"][direction] * scale
            return output
        
        # check if module has hook first.
        mod = getattr(m.model.diffusion_model.middle_block, "1")
        if getattr(mod, "hspace_hook", None) is not None:
            mod.hspace_hook.remove()
        setattr(mod, "hspace_hook", mod.register_forward_hook(hook_project))
    
        return (m, )

class SDModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE", ),
                "block": (["input_blocks", "middle_block", "output_blocks"], {"default": "input_blocks"}),
                "layer_num": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    DESCRIPTION = "Pick a layer out of input, middle or output blocks. This assumes a specific model structure that aligns with SD models, not tested on others."

    def find_conv2d_modules(self, model: nn.Module, parent_name: str = ''):
        conv_layers = []
        for name, module in model.named_modules():
            full_path = f"{parent_name}.{name}" if parent_name else name
            if isinstance(module, nn.Conv2d):
                conv_layers.append((full_path, module))
        return conv_layers

    def patch(self, model, bending_module, block, layer_num):
        m = copy.deepcopy(model)

        convs = self.find_conv2d_modules(
            getattr(m.model.diffusion_model, block))
        PromptServer.instance.send_sync("model_bending.bend_sd_model", {
                                        "num_layers": len(convs)})

        if layer_num >= len(convs):
            layer_num = len(convs) - 1

        path_to_module, _ = convs[layer_num]
        mod_path = "" if path_to_module == None or path_to_module == "" else process_path(
            "diffusion_model."+block + "." + path_to_module)

        hook_module(m.model.diffusion_model, mod_path, bending_module)
        return (m, )


    
class CustomModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE", ),
                "path": ("STRING",),
            }

        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self, model, bending_module, path):
        m = copy.deepcopy(model)
        '''
        All ModelPatcher instances have a .model property that is an instance of BaseModel.
        All models in comfy/model_base.py extend BaseModel. These include SDXL, Flux, Hunyuan, PixArt ...
        All BaseModel's have the property .diffusion_model as well
        '''
        mod_path = "" if path == None or path == "" else process_path(path)
        
        module = None
        if (hasattr(m, "model")):
            # If the model has a .model attribute, use it    
            module = m.model.diffusion_model
        elif hasattr(m, "stream"):
            module = m.stream.unet
        
        hook_module(module, mod_path, bending_module)
        return (m, )

# ----------------------- (U-Net) Model Bending -------------------------


class BaseModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    RETURN_TYPES = ("BENDING_MODULE",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"

    def patch(self):
        pass


class AddNoiseModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_std": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
            }
        }

    def patch(self, noise_std, seed):
        return (AddNoiseModule(noise_std=noise_std, seed=seed), )


class AddScalarModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scalar": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            }
        }

    def patch(self, scalar):
        return (AddScalarModule(scalar=scalar), )


class MultiplyScalarModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scalar": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            }
        }

    def patch(self, scalar):
        return (MultiplyScalarModule(scalar=scalar), )


class ThresholdModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "threshold": ("FLOAT", {"default": 0.0, }),
            }
        }

    def patch(self, threshold):
        return (ThresholdModule(threshold=threshold), )


class RotateModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "angle_degrees": ("FLOAT", {"default": 0.0, "min": -360, "max": 360, "step": 1}),
            }
        }

    def patch(self, angle_degrees):
        return (RotateModule(angle_degrees=angle_degrees), )


class ScaleModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scale_factor": ("FLOAT", {"default": 1.0, "min": -100, "max": 100}),
            }
        }

    def patch(self, scale_factor):
        return (ScaleModule(scale_factor=scale_factor), )


class ErosionModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kernel_size": ("INT", {"default": 0, "min": 1, "max": 10, "step": 1}),
            }
        }

    def patch(self, kernel_size):
        return (ErosionModule(kernel_size=kernel_size), )


class DilationModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kernel_size": ("INT", {"default": 0, "min": 1, "max": 10, "step": 1}),
            }
        }

    def patch(self, kernel_size):
        return (DilationModule(kernel_size=kernel_size), )


class GradientModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kernel_size": ("INT", {"default": 0, "min": 1, "max": 10, "step": 1}),
            }
        }

    def patch(self, kernel_size):
        return (GradientModule(kernel_size=kernel_size), )


class SobelModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normalized": ("BOOLEAN", {"default": True}),
            }
        }

    def patch(self, normalized):
        return (SobelModule(normalized=normalized), )


class LatentOperationToModule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "operation": ("LATENT_OPERATION",),
        }}
    RETURN_TYPES = ("BENDING_MODULE",)
    FUNCTION = "patch"

    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def patch(self, operation):
        class BendingModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, image):
                return operation(image)

        return (BendingModule(), )

# ------------------- LATENT OPERATIONS -------------------------------------


class LatentApplyBendingOperationCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "operation": ("LATENT_OPERATION",),
                             "step": ("INT", {"default": 0, "min": 0}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_bending"
    DESCRIPTION = """
    Applies the provided operation to an intermediate step in the latent denoising process.
    """

    def patch(self, model, operation, step):

        @torch.no_grad()
        def pre_cfg_function(args):
            conds_out = args["conds_out"]
            '''
            Comfy does not provide information about the current step easily. So I am getting around it by comparing the 
            current value of sigma against all sigmas
            '''
            sigmas = args["model_options"].get("transformer_options").get(
                "sample_sigmas").to(device='cpu')
            step_num = (sigmas == args["sigma"].cpu()
                        ).nonzero(as_tuple=True)[0]

            if step_num.nelement() > 0:
                if step_num[0] == step:
                    if len(conds_out) == 2:
                        conds_out[0] = operation(
                            latent=(conds_out[0] - conds_out[1])) + conds_out[1]
                    else:
                        conds_out[0] = operation(latent=conds_out[0])
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m, )


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
        return {
            "required": {
                "scalar": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.5}),
            }
        }

    def op(self, scalar):
        def scale(latent):
            return latent * scalar

        return (scale,)


class LatentOperationAddScalar(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scalar": ("FLOAT", {"default": 0.0, }),
            }
        }

    def op(self, scalar):
        def add(latent):
            return latent + scalar

        return (add,)


class LatentOperationThreshold(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "threshold": ("FLOAT", {"default": 0.0, }),
            }
        }

    def op(self, r):
        return (operations['threshold'](r),)


class LatentOperationAddNoise(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "std": ("FLOAT", {"default": 0.05, }),
            }
        }

    def op(self, std):
        return (operations['add_noise'](std),)


class LatentOperationRotate(BaseLatentOperation):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "axis": (["x", "y", "z"], {}),
                "angle": ("FLOAT", {"default": 0.0, }),
            }
        }

    def op(self, axis, angle):
        def rotate(latent):
            if axis == 'x':
                return operations['rotate_x'](angle)(latent)
            elif axis == 'y':
                return operations['rotate_y'](angle)(latent)
            else:
                return operations['rotate_z'](angle)(latent)

        return (rotate,)


class LatentOperationGeneric(BaseLatentOperation):
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation": (list(operations.keys()), {}),
            },
            "optional": {
                "float_param": ("FLOAT", {"default": 0.0, }),
                "float_param2": ("FLOAT", {"default": 0.0, }),
                "int_param": ("INT", {"default": 5, }),
                "int_param2": ("INT", {"default": 5, }),
                "bool_param": ("BOOLEAN", {"default": False}),
            }
        }

    def op(self, operation, float_param, float_param2, int_param, int_param2, bool_param):
        int_operations = ["reflect", "dilation", "erosion"]
        none_operations = ["absolute", "log", "gradient", "hadamard1"]
        dual_float_operations = ["clamp", "scale"]
        bool_operations = ["sobel"]

        print("operation", operation, float_param,
              float_param2, int_param, int_param2, bool_param)

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


# ------------Additional forms of bending (VAEs) and Latent Operations to Conditionings -------------------------
class CustomModuleVAEBending:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "path": ("STRING",),
                "bending_module": ("BENDING_MODULE", ),
            }
        }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    DESCRIPTION = "Bends a VAE model by injecting a module at the specified path."

    def patch(self, vae, path, bending_module):

        m = copy.deepcopy(vae)
        hook_module(m.patcher.model, path, bending_module)
        return (m, )


class ConditioningApplyOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "operation": ("LATENT_OPERATION", ),
                "zero_out": ("BOOLEAN",)
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def patch(self, cond, operation, zero_out):
        c = []
        for t in cond:
            d = t[1].copy()
            if zero_out:
                pooled_output = d.get("pooled_output", None)
                if pooled_output is not None:
                    d["pooled_output"] = torch.zeros_like(pooled_output)
            n = [operation(t[0]), d]
            c.append(n)
        return (c, )


# Finally, let ComfyUI know about the node:
NODE_CLASS_MAPPINGS = {
    "PCAPrep": PCAPrep,
    "HSpace Bending": HBending,

    "NoiseVariations": NoiseVariations,
    "Latent Operation To Module": LatentOperationToModule,
    "Custom Code Module": CodeNode,
    "Model Bending": CustomModelBending,
    "Model Bending (SD Layers)": SDModelBending,
    "Model VAE Bending": CustomModuleVAEBending,
    "Model Inspector": ShowModelStructure,

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
