import copy
import torch
import json

import comfy.model_management
import comfy.utils
from comfy.model_base import BaseModel
from server import PromptServer

from .py.custom_code_module import CodeNode
from .py.bendutils import operations, inject_module, get_model_tree, process_path
from .py.bending_modules import *


class ShowModelStructure:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "path_placeholder": ("STRING",)
            }
        }
    RETURN_TYPES = ("STRING","MODEL")
    FUNCTION = "show"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True
    DESCRIPTION = "Pick a layer by clicking on it in this inspector."

    def show(self, model, path_placeholder):
        
        tree = get_model_tree(model.model)
       
        PromptServer.instance.send_sync("model_bending.inspect_model", {"tree": json.dumps(tree)})

        # with open('data.json', 'w', encoding='utf-8') as f:
        #    json.dump(tree, f, ensure_ascii=False, indent=4)
        return (path_placeholder, model)

    #@classmethod
    #def IS_CHANGED(self, model, path_placeholder):
    #    return hash(str(model.model))

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
    EXPERIMENTAL = True
    def find_conv2d_modules(self, model: nn.Module, parent_name: str = ''):
                conv_layers = []
                for name, module in model.named_modules():
                    full_path = f"{parent_name}.{name}" if parent_name else name
                    if isinstance(module, nn.Conv2d):
                        conv_layers.append((full_path, module))
                return conv_layers
    
    def patch(self, model, bending_module, block, layer_num):
        m = copy.deepcopy(model)

        convs = self.find_conv2d_modules(getattr(m.model.diffusion_model, block))
        PromptServer.instance.send_sync("model_bending.bend_sd_model", {"num_layers": len(convs)})

        if layer_num >= len(convs):
            layer_num = len(convs) - 1 
         
        path_to_module, _ = convs[layer_num]
        mod_path = "" if path_to_module == None or path_to_module == "" else process_path("diffusion_model."+block+ "."+ path_to_module)
        
        inject_module(m.model.diffusion_model,mod_path, bending_module)
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
    EXPERIMENTAL = True

    

   
    def patch(self, model, bending_module, path):
        m = copy.deepcopy(model)
        '''
        All ModelPatcher instances have a .model property that is an instance of BaseModel.
        All models in comfy/model_base.py extend BaseModel. These include SDXL, Flux, Hunyuan, PixArt ...
        All BaseModel's have the property .diffusion_model as well
        '''
        mod_path = "" if path == None or path == "" else process_path(
            path)

        inject_module(m.model.diffusion_model, mod_path, bending_module)
        return (m, )

# ----------------------- (U-Net) Model Bending -------------------------

class BaseModelBending:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    RETURN_TYPES = ("BENDING_MODULE",)
    FUNCTION = "patch"
    CATEGORY = "model_bending"
    EXPERIMENTAL = True

    def patch(self):
        pass


class AddNoiseModelBending(BaseModelBending):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_std": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            }
        }

    def patch(self, noise_std):
        return (AddNoiseModule(noise_std=noise_std), )


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
                "threshold": ("FLOAT", {"default": 0.0,}),
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
    EXPERIMENTAL = True
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
    EXPERIMENTAL = True

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
    EXPERIMENTAL = True

    def patch(self, vae, path, bending_module):

        m = copy.deepcopy(vae)
        inject_module(m.patcher.model, path, bending_module)
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

    "LatentApplyOperationCFGToStep": LatentApplyBendingOperationCFG,
    "Latent Operation (Multiply Scalar)": LatentOperationMultiplyScalar,
    "Latent Operation (Add Scalar)": LatentOperationAddScalar,
    "Latent Operation (Threshold)": LatentOperationThreshold,
    "Latent Operation (Rotate)": LatentOperationRotate,
    "Latent Operation (Add Noise)": LatentOperationAddNoise,
    "Latent Operation (Custom)": LatentOperationGeneric,
    "ConditioningApplyOperation": ConditioningApplyOperation,

}
