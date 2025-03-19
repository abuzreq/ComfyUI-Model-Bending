import torch
import comfy.samplers
import comfy.model_management
import latent_preview
import comfy.utils

from .bendutils import operations

# Define the hook function that will be called during each denoising step.
def custom_transform_hook(args, threshold, transform_factor):
    """
    Hook that checks the current sigma value (args["sigma"]) and, if it exceeds the threshold,
    scales the latent tensor (args["input"]) by transform_factor.
    """
    print("my_custom_transform_hook", args.get("sigma", 0), threshold, transform_factor)
    sigma = args.get("sigma", 0)

    if sigma > threshold:
        # Apply your custom transformation (e.g. scaling)
        args["input"] = args["input"] * transform_factor
    # Return the denoised output unchanged so that the pipeline can continue normally.
    return args["conds_out"]

# Helper to add the hook into the model_options.
def apply_custom_transform_hook(model_options, threshold, transform_factor):
    def hook(args):
        return custom_transform_hook(args, threshold, transform_factor)
    if "sampler_pre_cfg_function" not in model_options:
        model_options["sampler_pre_cfg_function"] = []
    model_options["sampler_pre_cfg_function"].append(hook)
    return model_options

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
        


class LatentApplyBendingOperationCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "operation": ("LATENT_OPERATION",),
                             "layer": ("INT", {"default": 0, "min": 0}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Custom"
    EXPERIMENTAL = True

    def patch(self, model, operation, layer):
        
        @torch.no_grad()
        def pre_cfg_function(args):
            conds_out = args["conds_out"]
                  
            sigmas = args["model_options"].get("transformer_options").get("sample_sigmas").to(device='cpu')        
            step_num = (sigmas == args["sigma"].cpu()).nonzero(as_tuple=True)[0]
            
            if step_num.nelement() > 0:
                if step_num[0] == layer:
                    if len(conds_out) == 2:
                        conds_out[0] = operation(latent=(conds_out[0] - conds_out[1])) + conds_out[1]
                    else:
                        conds_out[0] = operation(latent=conds_out[0])
            return conds_out
        
        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m, )
    

class LatentOperationMultiplyScalar:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": { 
                    "scalar": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.5}),
                    }
                }

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"

    CATEGORY = "Custom"

    def op(self, scalar):
        def scale(latent):
            return latent * scalar
        
        return (scale,)

class LatentOperationAddScalar:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": { 
                    "scalar": ("FLOAT", {"default": 0.0,}),
                    }
                }

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"

    CATEGORY = "Custom"

    def op(self, scalar):
        def add(latent):
            return latent + scalar
        
        return (add,)

class LatentOperationThreshold:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": { 
                    "threshold": ("FLOAT", {"default": 0.0,}),
                    }
                }

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"

    CATEGORY = "Custom"

    def op(self, r):     
        return (operations['threshold'](r),)

class LatentOperationRotate:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": { 
                    "axis": (["x", "y", "z"], {}),
                    "angle": ("FLOAT", {"default": 0.0,}),
                    }
                }

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "op"

    CATEGORY = "Custom"

    def op(self, axis, angle):
        def rotate(latent):
            if axis == 'x':
                return operations['rotate_x'](angle)(latent)
            elif axis == 'y':
                return operations['rotate_y'](angle)(latent)
            else:
                return operations['rotate_z'](angle)(latent)
        
        return (rotate,)
    

class LatentOperation:
    @classmethod
    def INPUT_TYPES(s):
        print("LATENT_OPERATION", s, dir(s))

        return {
                "required": { 
                    "operation": (list(operations.keys()), {}),
                    },
                "optional":{
                        "float_param": ("FLOAT", {"default": 0.0,}),
                        "float_param2": ("FLOAT", {"default": 0.0,}),
                        "int_param": ("INT", {"default": 5,}),
                        "int_param2": ("INT", {"default": 5,}),
                        "bool_param": ("BOOLEAN", {"default": False}),
                    }
                }

    RETURN_TYPES = ("LATENT_OPERATION",)
    FUNCTION = "pick_operation"

    CATEGORY = "Custom"

    def pick_operation(self, operation, float_param, float_param2, int_param, int_param2, bool_param):
        int_operations = ["reflect", "dilation", "erosion"]
        none_operations = ["absolute", "log", "gradient"]
        dual_float_operations = ["clamp", "scale"]
        bool_operations = ["sobel"]
        
        print("operation", operation, float_param, float_param2, int_param, int_param2, bool_param)
        
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
    
# Finally, let ComfyUI know about the node:
NODE_CLASS_MAPPINGS = {
    "LatentApplyBendingOperationCFG": LatentApplyBendingOperationCFG,
    "LatentOperationMultiplyScalar": LatentOperationMultiplyScalar,
    "LatentOperationAddScalar": LatentOperationAddScalar,
    "LatentOperationThreshold": LatentOperationThreshold,
    "LatentOperationRotate": LatentOperationRotate,
    "LatentOperation": LatentOperation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "CustomHookSamplerNode": "CustomHookSamplerNode",
    "LatentApplyBendingOperationCFG": "LatentApplyBendingOperationCFG",
    "LatentOperationMultiplyScalar": "LatentOperationMultiplyScalar",
    "LatentOperationAddScalar": "LatentOperationAddScalar",
    "LatentOperationThreshold": "LatentOperationThreshold",
    "LatentOperationRotate": "LatentOperationRotate",
    "LatentOperation": "LatentOperation",
}
