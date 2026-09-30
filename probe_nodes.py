# Experimental nodes: activation probe and steering vectors (logic lives in probe.py).
import json

import numpy as np
import torch

from .bendutils import LOG_TAG
from .probe import STAT_NAMES, attach_probe, apply_steering, build_report, compute_steering, render_heatmap

_STORE = {"stats": "stats", "stats+channel_means": "channel_means", "stats+spatial_means": "spatial_means"}


class ActivationProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sites": ("STRING", {"default": "auto",
                                     "tooltip": "Comma-separated layer paths; '*' matches every child "
                                                "(e.g. 'output_blocks.*.0'). 'auto' picks one site per block."}),
                "store": (list(_STORE), {"default": "stats+channel_means",
                                         "tooltip": "channel means are needed for steering vectors; spatial means "
                                                    "also keep the layout (more memory)"}),
            },
            "optional": {
                "strict": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("MODEL", "PROBE")
    RETURN_NAMES = ("MODEL", "probe")
    FUNCTION = "attach"
    CATEGORY = "model_bending/probe"
    EXPERIMENTAL = True
    DESCRIPTION = ("Records activation statistics at the chosen layers while a sampler runs, without changing "
                   "the output. It measures after any bends, wherever it sits in the model chain. Read the results "
                   "with 'Read Activation Probe'. Use one probe per sampler.")

    def attach(self, model, sites, store, strict=False):
        m, handle = attach_probe(model, sites, store=_STORE[store], strict=strict)
        return (m, handle)


class ReadActivationProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "probe": ("PROBE",),
                "latent": ("LATENT", {"tooltip": "Connect the output of the sampler that used the probed model, "
                                                 "so this node runs after sampling"}),
                "stat": (["std", "rms", "mean", "absmax", "dead_frac"], {"default": "std"}),
                "alert_ratio": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 1000.0, "step": 0.5,
                                          "tooltip": "Alert when std differs from the reference by this factor"}),
            },
            "optional": {
                "reference": ("ACTIVATIONS", {"tooltip": "Activations of an unbent run: shows bent / unbent ratios"}),
            },
        }
    RETURN_TYPES = ("ACTIVATIONS", "IMAGE", "STRING")
    RETURN_NAMES = ("activations", "heatmap", "report")
    FUNCTION = "read"
    CATEGORY = "model_bending/probe"
    EXPERIMENTAL = True

    def read(self, probe, latent, stat, alert_ratio, reference=None):
        if not probe.has_data():
            raise ValueError(f"{LOG_TAG} the probe recorded nothing: connect its MODEL output to a sampler and "
                             "this node's latent input to that sampler's output")
        act = probe.snapshot()
        report = build_report(act, reference=reference, stat=stat, alert_ratio=alert_ratio)
        image = render_heatmap(act, reference=reference, stat=stat)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]
        return (act, tensor, json.dumps(report, indent=1))


class SteeringVectorFromActivations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "activations_a": ("ACTIVATIONS", {"tooltip": "Run with the concept to steer towards"}),
                "activations_b": ("ACTIVATIONS", {"tooltip": "Run with the concept to steer away from"}),
                "site": ("STRING", {"default": "middle_block.1"}),
                "mode": (["channel", "spatial"], {"default": "channel",
                                                  "tooltip": "channel: per-channel means (style/concept, layout-free); "
                                                             "spatial: keeps the layout (needs spatial means)"}),
            }
        }
    RETURN_TYPES = ("STEERING_VECTOR", "STRING")
    RETURN_NAMES = ("steering_vector", "info")
    FUNCTION = "compute"
    CATEGORY = "model_bending/probe"
    EXPERIMENTAL = True
    DESCRIPTION = "Steering vector = mean activation of run A minus run B at one layer, per denoising step."

    def compute(self, activations_a, activations_b, site, mode):
        vector = compute_steering(activations_a, activations_b, site.strip(), mode)
        info = {"site": vector["site"], "mode": mode, "steps": int(vector["delta"].shape[0]),
                "shape": list(vector["delta"].shape[1:]),
                "relative_norm": round(vector["rel_norm"], 4)}
        return (vector, json.dumps(info, indent=1))


class ApplySteeringVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steering_vector": ("STEERING_VECTOR",),
                "strength": ("FLOAT", {"default": 4.0, "min": -100.0, "max": 100.0, "step": 0.1,
                                       "tooltip": "0 = off; negative steers towards run B"}),
            },
            "optional": {
                "t_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "t_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_to": (["both", "cond"], {"default": "both",
                                                "tooltip": "both CFG halves, or only the conditional one"}),
                "normalize": ("BOOLEAN", {"default": False,
                                          "tooltip": "Rescale the vector to the layer's activation size, "
                                                     "so strength 1 = one activation's worth"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_bending/probe"
    EXPERIMENTAL = True

    def apply(self, model, steering_vector, strength, t_start=1.0, t_end=0.0, apply_to="both", normalize=False):
        return (apply_steering(model, steering_vector, strength, t_start=t_start, t_end=t_end,
                               apply_to=apply_to, normalize=normalize),)


NODE_CLASS_MAPPINGS = {
    "ActivationProbe": ActivationProbe,
    "ReadActivationProbe": ReadActivationProbe,
    "SteeringVectorFromActivations": SteeringVectorFromActivations,
    "ApplySteeringVector": ApplySteeringVector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ActivationProbe": "Activation Probe",
    "ReadActivationProbe": "Read Activation Probe",
    "SteeringVectorFromActivations": "Steering Vector (from Activations)",
    "ApplySteeringVector": "Apply Steering Vector",
}
