# ComfyUI-Web-Bend-Demo/nodes.py
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ComfyUI server routing (custom routes)
from aiohttp import web
from server import PromptServer

import difflib
import fnmatch
import json
import hashlib
import math
import os
import re

from .bendutils import (
    operations, parse_step_str_to_ranges, make_bends_wrapper, BendSpec, warn, LOG_TAG, GUARD_NAN_MODES,
    expand_path_pattern, resolve_hook_target,
)
from .bending_modules import (
    BendingModule,
    AddNoiseModule,
    FourierAmplifyModule,
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

# Import rotation operations from original plugin
try:
    import kornia.geometry.transform as KT
except ImportError:
    KT = None
    print("[model-bending] kornia not available, rotation bending will not work")
# ----------------------------
# 1) In-memory per-session state
# ----------------------------

@dataclass
class BendSelection:
    """Per-path bends: list of { path, module_type, module_args }. Optional steps_min/steps_max for timestep range. selected_part = which top-level part is being bent (e.g. diffusion_model, single_blocks)."""
    bends: List[Dict[str, Any]] = field(default_factory=list)
    steps_min: Optional[int] = None
    steps_max: Optional[int] = None
    max_denoising_steps: int = 200
    selected_part: Optional[str] = None

MODEL_BY_SESSION: Dict[str, Any] = {}
SELECTION_BY_SESSION: Dict[str, BendSelection] = {}


# Bending module classes are in bending_modules.py (shared with model_bending_nodes).

# ----------------------------
# Bend ops for the JSON format
# ----------------------------
# Version 1.1 only adds optional keys to v1, so older plugin versions still apply the bends (less precisely)
# instead of skipping them, and v1 JSON keeps working unchanged.
BENDS_JSON_VERSION = 1.1
_TOP_KEYS = {"bends", "bend", "steps_min", "steps_max", "max_denoising_steps", "selected_part", "version"}
_BEND_KEYS = {"path", "module_type", "module_args", "angle", "steps", "t", "blend", "label", "guard", "inner"}

# name -> (factory, {arg: (type, default, hard limits)}). Hard limits mirror the matching module nodes'
# widget ranges and are what clamp="hard" enforces; for str args they are the allowed choices.
BEND_OPS: Dict[str, tuple] = {
    "add_scalar": (AddScalarModule, {"scalar": (float, 0.0, (-100.0, 100.0))}),
    "add_noise": (AddNoiseModule, {"noise_std": (float, 0.0, (-100.0, 100.0)), "seed": (int, 42, None)}),
    "multiply": (MultiplyScalarModule, {"scalar": (float, 1.0, (-100.0, 100.0))}),
    "rotate": (RotateModule, {"angle_degrees": (float, 0.0, (-360.0, 360.0))}),
    "threshold": (ThresholdModule, {"threshold": (float, 0.0, None)}),
    "scale": (ScaleModule, {"scale_factor": (float, 1.0, (-100.0, 100.0))}),
    "erosion": (ErosionModule, {"kernel_size": (int, 3, (1, 10))}),
    "dilation": (DilationModule, {"kernel_size": (int, 3, (1, 10))}),
    "gradient": (GradientModule, {"kernel_size": (int, 3, (1, 10))}),
    "sobel": (SobelModule, {"normalized": (bool, True, None)}),
    "fourier": (FourierAmplifyModule, {"cutoff_freq": (float, 5.0, (0.0, 10.0)),
                                       "amp_factor": (float, 2.0, (-10.0, 10.0))}),
    # Applies the "inner" op to a random subset of the batch, channels or pixels.
    "subset": (ApplyToRandomSubsetModule, {"percentage": (float, 0.5, (0.0, 1.0)),
                                           "dim": (str, "batch", ("batch", "channel", "spatial")),
                                           "seed": (int, 0, None)}),
}


def _normalize_module_args(module_args: dict) -> Dict[str, Any]:
    """Normalize args for stable JSON hashing: round floats, use int when equal."""
    out = {}
    for k, v in (module_args or {}).items():
        if isinstance(v, float):
            v = round(v, 6)
            if v == int(v):
                v = int(v)
        out[k] = v
    return out


def _coerce(typ, value):
    if typ is bool:
        if isinstance(value, str):
            if value.strip().lower() in ("true", "1", "yes"):
                return True
            if value.strip().lower() in ("false", "0", "no"):
                return False
            raise ValueError(value)
        return bool(value)
    if typ is int:
        return int(float(value))
    if typ is float:
        return float(value)
    return str(value)


def _resolve_op(node: Dict[str, Any], where: str, problems: List[str]) -> Dict[str, Any]:
    """{module_type, module_args[, inner]} with every argument present, typed and validated."""
    module_type = node.get("module_type")
    if module_type not in BEND_OPS:
        raise ValueError(f"{where}unknown module_type {module_type!r}; known: {', '.join(BEND_OPS)}")
    spec = BEND_OPS[module_type][1]
    given = node.get("module_args") or {}
    if not isinstance(given, dict):
        problems.append(f"{where}module_args must be an object; using defaults")
        given = {}
    args = {}
    for name, (typ, default, limits) in spec.items():
        value = default
        if name in given:
            try:
                value = _coerce(typ, given[name])
            except (TypeError, ValueError):
                problems.append(f"{where}{module_type}.{name}={given[name]!r} is not a {typ.__name__}; "
                                f"using {default!r}")
            if typ is str and limits and value not in limits:
                problems.append(f"{where}{module_type}.{name}={value!r} must be one of {', '.join(limits)}; "
                                f"using {default!r}")
                value = default
        args[name] = value
    for name in given:
        if name not in spec:
            hint = difflib.get_close_matches(name, list(spec), n=1)
            problems.append(f"{where}unknown argument {name!r} for {module_type} (ignored); "
                            + (f"did you mean {hint[0]!r}?" if hint else f"accepted: {', '.join(spec)}"))
    op = {"module_type": module_type, "module_args": args}
    if module_type == "subset":
        inner = node.get("inner")
        if not isinstance(inner, dict) or not inner.get("module_type"):
            raise ValueError(f"{where}subset needs an 'inner' bend with a module_type")
        op["inner"] = _resolve_op({"module_type": inner["module_type"],
                                   "module_args": _normalize_module_args(inner.get("module_args") or {}),
                                   "inner": inner.get("inner")}, where + "inner ", problems)
    elif node.get("inner") is not None:
        problems.append(f"{where}'inner' is only used by module_type 'subset' (ignored)")
    return op


def _clamp_op(op: Dict[str, Any], path: str, mode: str, safe_ranges: Optional[Dict[str, Any]],
              report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Clamp numeric arguments. "hard": each op's absolute limits. "safe": hard limits, then safe_ranges
    ({op: {arg: [lo, hi]}}, plus optional {"paths": {"<glob>": {op: {arg: [lo, hi]}}}}, most specific glob last).
    """
    if mode == "none":
        return op
    module_type = op["module_type"]
    ranges = {name: limits for name, (typ, _, limits) in BEND_OPS[module_type][1].items()
              if typ in (int, float) and limits}
    if mode == "safe" and safe_ranges:
        ranges.update(safe_ranges.get(module_type, {}))
        for pattern in sorted(safe_ranges.get("paths", {}), key=len):
            if fnmatch.fnmatchcase(path, pattern):
                ranges.update(safe_ranges["paths"][pattern].get(module_type, {}))
    args = dict(op["module_args"])
    for name, limits in ranges.items():
        value = args.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        lo, hi = float(limits[0]), float(limits[1])
        clamped = type(value)(min(max(value, lo), hi))
        if clamped != value:
            args[name] = clamped
            if report is not None:
                report.setdefault("clamped", []).append(
                    {"path": path, "op": module_type, "arg": name, "requested": value, "applied": clamped})
    out = {**op, "module_args": args}
    if "inner" in op:
        out["inner"] = _clamp_op(op["inner"], path, mode, safe_ranges, report)
    return out


def _build_op(op: Dict[str, Any]) -> nn.Module:
    factory = BEND_OPS[op["module_type"]][0]
    if op["module_type"] == "subset":
        a = op["module_args"]
        return factory(_build_op(op["inner"]), percentage=a["percentage"], seed=a["seed"], dim=a["dim"])
    return factory(**op["module_args"])


def build_bending_module(module_type: str, module_args: dict, inner: Optional[dict] = None) -> nn.Module:
    problems: List[str] = []
    module = _build_op(_resolve_op({"module_type": module_type, "module_args": module_args or {}, "inner": inner},
                                   "", problems))
    for p in problems:
        warn(p)
    return module


# ----------------------------
# 3) Model / module path helpers
# ----------------------------

def get_unet_from_comfy_model(m) -> Optional[nn.Module]:
    # matches common comfy models
    if hasattr(m, "model") and hasattr(m.model, "diffusion_model"):
        return m.model.diffusion_model
    if hasattr(m, "stream") and hasattr(m.stream, "unet"):
        return m.stream.unet
    return None


def set_unet_on_comfy_model(m, unet: nn.Module) -> None:
    """Set the UNet on a Comfy model (mirrors get_unet_from_comfy_model)."""
    if hasattr(m, "model") and hasattr(m.model, "diffusion_model"):
        m.model.diffusion_model = unet
    elif hasattr(m, "stream") and hasattr(m.stream, "unet"):
        m.stream.unet = unet


def get_root_module(model) -> Optional[nn.Module]:
    """Return the root nn.Module to inspect (e.g. model.model or model.stream). Used for listing top-level parts."""
    if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
        return model.model
    if hasattr(model, "stream") and hasattr(model.stream, "unet"):
        return model.stream
    return None


def _count_tree_leaves(node: Dict[str, Any]) -> int:
    """Count leaf nodes in a tree built by build_module_tree (node has 'children' list)."""
    children = node.get("children") or []
    if not children:
        return 1
    return sum(_count_tree_leaves(c) for c in children)


def list_top_level_parts(root_module: nn.Module, max_depth: int = 4) -> List[Dict[str, Any]]:
    """List direct children of root as top-level parts. Only includes parts that have at least one child (submodule). Returns [{ name, path, size }, ...]. path is the attribute name (e.g. diffusion_model, single_blocks). size = leaf count for default selection."""
    parts: List[Dict[str, Any]] = []
    for child_name, child in root_module.named_children():
        if not child_name:
            continue
        # Skip parts that have no children (leaf modules)
        if not any(True for _ in child.named_children()):
            continue
        try:
            sub_tree = build_module_tree(child, max_depth=max_depth)
            size = _count_tree_leaves(sub_tree)
        except Exception:
            size = 0
        parts.append({"name": child_name, "path": child_name, "size": size})
    return parts


def get_default_part(top_level_parts: List[Dict[str, Any]]) -> Optional[str]:
    """Default part: diffusion_model if present, else the part with largest size."""
    for p in top_level_parts:
        path = (p.get("path") or "").strip()
        if path == "diffusion_model" or path.endswith(".diffusion_model"):
            return path
    if not top_level_parts:
        return None
    best = max(top_level_parts, key=lambda x: x.get("size") or 0)
    return best.get("path")


def detect_model_type(root_module: nn.Module, top_level_parts: List[Dict[str, Any]]) -> str:
    """Infer model type from structure for diagram/info. Returns 'sd' | 'flux' | 'unknown'.
    Both SD (BaseModel) and Flux (Flux2) in ComfyUI often have the same root shape: diffusion_model + model_sampling.
    We distinguish by inspecting the diffusion_model module:
    - UNetModel (SD) has middle_block, input_blocks, output_blocks.
    - Flux has single_blocks, double_blocks (and no middle_block).
    Some Flux loaders expose single_blocks/double_blocks as top-level parts of root; we treat that as flux first.
    """
    paths = [str(p.get("path", "")).strip() for p in top_level_parts]
    # If single_blocks/double_blocks appear as top-level parts (some Flux loaders), treat as flux
    if any(p == "single_blocks" or p == "double_blocks" for p in paths):
        return "flux"
    # Root has diffusion_model (both SD and Flux): inspect the module under it
    if any("diffusion_model" in p or p == "diffusion_model" for p in paths):
        dm = getattr(root_module, "diffusion_model", None)
        if dm is not None:
            # SD: UNetModel has middle_block
            if hasattr(dm, "middle_block"):
                return "sd"
            # Flux: Flux module has single_blocks/double_blocks (no middle_block)
            if hasattr(dm, "single_blocks") or hasattr(dm, "double_blocks"):
                return "flux"
        return "unknown"
    return "unknown"


def resolve_module(root: nn.Module, layer_path: str) -> nn.Module:
    cur: Any = root
    for part in layer_path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def build_module_tree(root: nn.Module, max_depth: int = 10) -> Dict[str, Any]:
    def rec(mod: nn.Module, name: str, path: str, depth: int):
        # Get full module type information
        module_type = mod.__class__.__name__
        module_full_name = f"{mod.__class__.__module__}.{module_type}"
        
        node = {
            "name": name,
            "path": path,
            "type": module_type,
            "full_type": module_full_name,
            "children": []
        }
        if depth >= max_depth:
            node["truncated"] = True
            return node

        for child_name, child in mod.named_children():
            child_path = f"{path}.{child_name}" if path else child_name
            node["children"].append(rec(child, child_name, child_path, depth + 1))
        return node

    return rec(root, "root", "", 0)


# ----------------------------
# 4) Hook-based intervention for demo
#    (This is what you'll later swap for NNsight.)
# ----------------------------

def list_hooks(module: nn.Module, prefix: str = "") -> List[tuple]:
    """Recursively list all forward hooks on a module and its children. Returns [(path, n_fwd, n_pre), ...]."""
    hooks: List[tuple] = []
    fwd = getattr(module, "_forward_hooks", None)
    pre = getattr(module, "_forward_pre_hooks", None)
    n_fwd = len(fwd) if fwd else 0
    n_pre = len(pre) if pre else 0
    if n_fwd > 0 or n_pre > 0:
        path = prefix or module.__class__.__name__
        hooks.append((path, n_fwd, n_pre))
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        hooks.extend(list_hooks(child, child_prefix))
    return hooks


def clear_forward_hooks(module: nn.Module) -> None:
    """Remove all forward and pre-forward hooks from a module and its children."""
    if hasattr(module, "_forward_hooks"):
        module._forward_hooks.clear()
    if hasattr(module, "_forward_pre_hooks"):
        module._forward_pre_hooks.clear()
    for child in module.children():
        clear_forward_hooks(child)


def parse_bends_json(json_str: str, problems: Optional[List[str]] = None) -> tuple:
    """
    Parse JSON from the web UI "Copy Bends" clipboard format.
    Returns (bends, steps_min, steps_max, max_denoising_steps, selected_part).
    Supports: { "bends": [...], "steps_min": ?, "steps_max": ?, "max_denoising_steps": ?, "selected_part": ?,
                "version": ? }
    Each bend: { "path", "module_type", "module_args" } or legacy { "path", "angle" }.
    Optional per-bend fields (version 1.1): "steps" (e.g. "0-4,9" or "3-"), "t" ([hi, lo] normalised diffusion
    time, 1 = pure noise), "blend" (0..1), "label", "guard" ({"nan", "max_std_ratio", "preserve_norm"}),
    "inner" (the op wrapped by module_type "subset"). "path" may use wildcards ("output_blocks.*.1").
    Structural problems (unknown keys, bends without a path, ...) are appended to `problems`, or logged when
    no list is given. Raises ValueError on invalid JSON.
    """
    own_problems = problems is None
    problems = [] if problems is None else problems
    if not (json_str or "").strip():
        return [], None, None, 200, None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object with 'bends' array")
    for key in data:
        if key not in _TOP_KEYS:
            problems.append(f"unknown top-level key {key!r} (ignored)")
    version = data.get("version")
    if version is not None:
        try:
            if float(version) > BENDS_JSON_VERSION:
                problems.append(f"version {version} is newer than this plugin understands ({BENDS_JSON_VERSION}); "
                                "fields it does not know are ignored")
        except (TypeError, ValueError):
            problems.append(f"version {version!r} is not a number")
    raw = data.get("bends", data.get("bend", []))
    if isinstance(raw, dict):
        raw = list(raw.values()) if raw else []
    if not isinstance(raw, list):
        problems.append("'bends' must be a list; no bends applied")
        raw = []
    bends: List[Dict[str, Any]] = []
    for i, b in enumerate(raw):
        if not isinstance(b, dict):
            problems.append(f"bend #{i} is not an object; skipped")
            continue
        path = b.get("path")
        if not path or not isinstance(path, str):
            problems.append(f"bend #{i} has no 'path'; skipped")
            continue
        where = f"bend #{i} ({path}): "
        for key in b:
            if key not in _BEND_KEYS:
                problems.append(f"{where}unknown key {key!r} (ignored)")
        if "angle" in b and isinstance(b.get("angle"), (int, float)):
            module_type = "rotate"
            module_args = _normalize_module_args({"angle_degrees": float(b["angle"])})
        else:
            module_type = (b.get("module_type") or "").strip()
            if not module_type:
                problems.append(f"{where}no module_type; defaulting to 'rotate'")
                module_type = "rotate"
            module_args = _normalize_module_args(b.get("module_args") or {})
        bend = {"path": path, "module_type": module_type, "module_args": module_args}
        for key in ("steps", "t", "blend", "label", "guard", "inner"):
            if b.get(key) is not None:
                bend[key] = b[key]
        bends.append(bend)

    steps_min = data.get("steps_min")
    steps_max = data.get("steps_max")
    max_denoising_steps = data.get("max_denoising_steps", 200)
    for name, val in (("steps_min", steps_min), ("steps_max", steps_max)):
        if val is not None and not isinstance(val, int):
            try:
                if name == "steps_min":
                    steps_min = int(val)
                else:
                    steps_max = int(val)
            except (TypeError, ValueError):
                problems.append(f"{name}={val!r} is not an integer (ignored)")
                if name == "steps_min":
                    steps_min = None
                else:
                    steps_max = None
    if not isinstance(max_denoising_steps, int):
        try:
            max_denoising_steps = int(max_denoising_steps)
        except (TypeError, ValueError):
            max_denoising_steps = 200
    max_denoising_steps = max(1, min(1000, max_denoising_steps))
    selected_part = data.get("selected_part")
    if selected_part is not None and not isinstance(selected_part, str):
        selected_part = str(selected_part).strip() or None
    elif selected_part is not None:
        selected_part = (selected_part or "").strip() or None
    if own_problems:
        for p in problems:
            warn(p)
    return bends, steps_min, steps_max, max_denoising_steps, selected_part


def _steps_str(steps_min, steps_max):
    if steps_min is None and steps_max is None:
        return "*"
    return f"{'' if steps_min is None else steps_min}-{'' if steps_max is None else steps_max}"


def _parse_t_window(value):
    if value is None:
        return None
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"'t' must be [hi, lo], got {value!r}")
    hi, lo = float(max(value)), float(min(value))
    return None if (hi >= 1.0 and lo <= 0.0) else (hi, lo)


def _check_guard(guard, where: str, problems: List[str]) -> Optional[Dict[str, Any]]:
    if guard is None:
        return None
    if not isinstance(guard, dict):
        problems.append(f"{where}guard must be an object (ignored)")
        return None
    out: Dict[str, Any] = {}
    for key, value in guard.items():
        if key == "nan" and value in GUARD_NAN_MODES:
            out[key] = value
        elif key == "max_std_ratio" and isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            out[key] = float(value)
        elif key == "preserve_norm" and isinstance(value, bool):
            out[key] = value
        elif key in ("nan", "max_std_ratio", "preserve_norm"):
            problems.append(f"{where}invalid guard.{key}={value!r} (ignored); nan: {'|'.join(GUARD_NAN_MODES)}, "
                            "max_std_ratio: number > 0, preserve_norm: true|false")
        else:
            problems.append(f"{where}unknown guard key {key!r} (ignored)")
    return out or None


def apply_bends_to_model(model, bends: List[Dict[str, Any]], steps_min: Optional[int] = None,
                         steps_max: Optional[int] = None, max_denoising_steps: int = 200,
                         selected_part: Optional[str] = None, strict: bool = False,
                         report: Optional[Dict[str, Any]] = None, clamp: str = "none",
                         safe_ranges: Optional[Dict[str, Any]] = None, problems: Optional[List[str]] = None,
                         resolved: Optional[List[Dict[str, Any]]] = None):
    """
    Apply a list of bends to a model and return the patched model.
    bends: list of { "path", "module_type", "module_args", optional "steps", "t", "blend", "label", "guard",
    "inner" } (see parse_bends_json). Paths are relative to selected_part (default: diffusion_model), as shown
    by the web UI, and may contain wildcards.
    steps_min/steps_max give the default step range; either end may be omitted.
    clamp: "none" | "hard" | "safe" (see _clamp_op). Problems found are logged and listed in report["warnings"];
    with strict they raise. `resolved` receives one fully explicit bend per hooked layer.
    All bends share one per-forward wrapper; hooks are removed in a finally block, so no hooks persist
    on the shared underlying module between steps or across parallel ComfyUI branches.
    Bends on the same layer apply last-to-first, as they always have for this format.
    """
    problems = [] if problems is None else problems

    def finish():
        """Log problems, list them in the report and enforce strict. Runs on every exit path."""
        for p in problems:
            warn(p)
        if report is not None and problems:
            report["warnings"] = list(problems)
        if strict and problems:
            raise ValueError(f"{LOG_TAG} bends JSON problems (strict):\n- " + "\n- ".join(problems))

    if not bends:
        if not problems:
            problems.append("no bends to apply")
        finish()
        return (model,)
    if not (hasattr(model, "clone") and callable(getattr(model, "clone", None))):
        raise RuntimeError("Model has no clone() method; cannot patch for bending.")
    m = model.clone()
    target_module = get_unet_from_comfy_model(m)
    if target_module is None:
        problems.append("no diffusion model found on this MODEL; nothing bent")
        finish()
        return (model,)
    if selected_part and selected_part != "diffusion_model":
        try:
            target_module = resolve_module(get_root_module(m), selected_part)
        except (AttributeError, KeyError, IndexError, TypeError):
            problems.append(f"selected_part {selected_part!r} not found on this model; "
                            "resolving paths from diffusion_model")
    if clamp == "safe" and not safe_ranges:
        problems.append("clamp='safe' without safe_ranges; using each op's hard limits")
    default_steps = parse_step_str_to_ranges(_steps_str(steps_min, steps_max), max_steps=max_denoising_steps)
    specs = []
    for i, b in enumerate(bends):
        where = f"bend #{i} ({b['path']}): "
        op = _resolve_op(b, where, problems)
        steps = default_steps
        if b.get("steps") is not None:
            steps = parse_step_str_to_ranges(str(b["steps"]), max_steps=max_denoising_steps)
            if steps is None and str(b["steps"]).strip() not in ("", "*"):
                problems.append(f"{where}steps {b['steps']!r} is not a valid step list; bending all steps")
        blend = b.get("blend", 1.0)
        try:
            blend = float(blend)
        except (TypeError, ValueError):
            problems.append(f"{where}blend {blend!r} is not a number; using 1")
            blend = 1.0
        if not 0.0 <= blend <= 1.0:
            problems.append(f"{where}blend {blend} is outside 0..1; clamped")
            blend = max(0.0, min(1.0, blend))
        guard = _check_guard(b.get("guard"), where, problems)
        t_window = _parse_t_window(b.get("t"))

        paths = expand_path_pattern(target_module, b["path"])
        if not paths:
            problems.append(f"{where}the pattern matched no layers")
            continue
        if paths != [b["path"]] and report is not None:
            report.setdefault("patterns", []).append({"path": b["path"], "matches": paths})
        for path in paths:
            hooked, mod = resolve_hook_target(target_module, path, strict=strict, report=report)
            if mod is None:
                continue
            final = _clamp_op(op, hooked, clamp, safe_ranges, report)
            specs.append(BendSpec(hooked, _build_op(final), steps_to_bend=steps, t_window=t_window,
                                  blend=blend, label=b.get("label"), guard=guard))
            if resolved is not None:
                entry = {"path": hooked, **final}
                for key in ("steps", "t", "label"):
                    if b.get(key) is not None:
                        entry[key] = b[key]
                if blend != 1.0:
                    entry["blend"] = blend
                if guard:
                    entry["guard"] = guard
                resolved.append(entry)

    finish()
    prev_wrapper = m.model_options.get("model_function_wrapper")
    m.set_model_unet_function_wrapper(
        make_bends_wrapper(target_module, specs, prev_wrapper=prev_wrapper, strict=strict, report=report,
                           reverse_hooks=True)
    )
    return (m,)


def hook_module(root: nn.Module, layer_path: str, bending_module: nn.Module):
    """
    Minimal: register a forward hook that replaces output with bending_module.bend(output)
    """
    target = resolve_module(root, layer_path)

    def hook_fn(module, args, kwargs, output):
        return bending_module(output, *args, **kwargs)

    # with_kwargs=True gives (module, args, kwargs, output) signature
    return target.register_forward_hook(hook_fn, with_kwargs=True)


# ----------------------------
# 5) Custom HTTP API routes
# ----------------------------

@PromptServer.instance.routes.get("/web_bend_demo/layers")
async def api_layers(request):
    session_id = request.rel_url.query.get("session_id", "")
    part = request.rel_url.query.get("part", "").strip()
    model = MODEL_BY_SESSION.get(session_id)
    if model is None:
        return web.json_response(
            {"ok": False, "error": "Unknown session_id. Run the node once (queue a prompt) so it registers the model."},
            status=404
        )
    root = get_root_module(model)
    if root is None:
        return web.json_response({"ok": False, "error": "Could not get model root (no model.model or model.stream)."}, status=400)
    top_level_parts = list_top_level_parts(root, max_depth=4)
    if not top_level_parts:
        return web.json_response({"ok": False, "error": "Model has no top-level parts."}, status=400)
    default_part = get_default_part(top_level_parts)
    model_type = detect_model_type(root, top_level_parts)
    selected_part = part or default_part
    sel = SELECTION_BY_SESSION.get(session_id, BendSelection())
    if not selected_part and getattr(sel, "selected_part", None):
        selected_part = sel.selected_part
    if not selected_part:
        selected_part = default_part
    if selected_part:
        try:
            part_module = resolve_module(root, selected_part)
        except (AttributeError, KeyError, IndexError):
            selected_part = default_part
            part_module = resolve_module(root, selected_part) if default_part else None
    else:
        part_module = None
    if part_module is None:
        return web.json_response({"ok": False, "error": "Could not resolve selected part."}, status=400)
    tree = build_module_tree(part_module, max_depth=12)

    # Inferred model name from root class (e.g. Flux2, BaseModel) for UI display
    inferred_model_name = getattr(type(root), "__name__", "Model")

    return web.json_response({
        "ok": True,
        "tree": tree,
        "session_id": session_id,
        "top_level_parts": top_level_parts,
        "default_part": default_part,
        "selected_part": selected_part,
        "model_type": model_type,
        "inferred_model_name": inferred_model_name,
    })


@PromptServer.instance.routes.get("/web_bend_demo/poll_image")
async def api_poll_image(request):
    """Poll for the newest generated image from ComfyUI history"""
    try:
        # Use ComfyUI's built-in history endpoint by proxying the request
        # This is more reliable than accessing internal structures
        from aiohttp import ClientSession
        
        # Get the base URL from the request
        base_url = f"{request.scheme}://{request.host}"
        history_url = f"{base_url}/history"
        
        async with ClientSession() as session:
            async with session.get(history_url) as resp:
                if resp.status != 200:
                    return web.json_response({"ok": False, "error": "Failed to fetch history"}, status=resp.status)
                
                history = await resp.json()
        
        if not history:
            return web.json_response({"ok": False, "error": "No history available"}, status=404)
        
        # Find the most recent image
        newest_image = None
        newest_timestamp = 0
        
        for prompt_id, entry in history.items():
            if not entry or not isinstance(entry, dict):
                continue
                
            # Try different timestamp field names
            timestamp = entry.get("timestamp") or entry.get("time") or entry.get("created_at") or 0
            outputs = entry.get("outputs", {})
            
            if not outputs:
                continue
            
            for node_id, node_output in outputs.items():
                images = node_output.get("images", [])
                if images:
                    for img in images:
                        if isinstance(img, dict):
                            img_timestamp = timestamp
                            if img_timestamp > newest_timestamp:
                                newest_timestamp = img_timestamp
                                newest_image = {
                                    "filename": img.get("filename", ""),
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")
                                }
        
        if newest_image and newest_image["filename"]:
            return web.json_response({"ok": True, "image": newest_image})
        else:
            return web.json_response({"ok": False, "error": "No images found in history"})
            
    except Exception as e:
        import traceback
        return web.json_response({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, status=500)


def hash_list_of_dicts(data: List[Dict[str, Any]]) -> str:
    """Stable hash: convert to JSON, sort list items for order-independence."""
    sorted_data = sorted(data, key=lambda x: json.dumps(x, sort_keys=True))
    data_string = json.dumps(sorted_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data_string).hexdigest()


def _bends_hash(bends: List[Dict[str, Any]]) -> str:
    """Stable hash for bends list (order-independent)."""
    return hash_list_of_dicts(bends)


@PromptServer.instance.routes.get("/web_bend_demo/selection")
async def api_get_selection(request):
    session_id = request.rel_url.query.get("session_id", "")
    sel = SELECTION_BY_SESSION.get(session_id, BendSelection())
    return web.json_response({
        "ok": True,
        "bends": sel.bends,
        "steps_min": getattr(sel, "steps_min", None),
        "steps_max": getattr(sel, "steps_max", None),
        "max_denoising_steps": getattr(sel, "max_denoising_steps", 200),
        "selected_part": getattr(sel, "selected_part", None),
    })


@PromptServer.instance.routes.post("/web_bend_demo/selection")
async def api_set_selection(request):
    data = await request.json()
    session_id = data.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": False, "error": "missing session_id"}, status=400)

    raw = data.get("bends", []) or []
    bends: List[Dict[str, Any]] = []
    for b in raw:
        path = b.get("path")
        module_type = (b.get("module_type") or "").strip() or "rotate"
        module_args = _normalize_module_args(b.get("module_args") or {})
        if path and isinstance(path, str):
            bends.append({"path": path, "module_type": module_type, "module_args": module_args})

    selected_part = data.get("selected_part")
    if selected_part is not None and not isinstance(selected_part, str):
        selected_part = str(selected_part).strip() or None
    elif selected_part is not None:
        selected_part = (selected_part or "").strip() or None

    steps_min = data.get("steps_min")
    steps_max = data.get("steps_max")
    max_denoising_steps = data.get("max_denoising_steps", 200)
    if steps_min is not None and not isinstance(steps_min, int):
        try:
            steps_min = int(steps_min)
        except (TypeError, ValueError):
            steps_min = None
    if steps_max is not None and not isinstance(steps_max, int):
        try:
            steps_max = int(steps_max)
        except (TypeError, ValueError):
            steps_max = None
    if not isinstance(max_denoising_steps, int):
        try:
            max_denoising_steps = int(max_denoising_steps)
        except (TypeError, ValueError):
            max_denoising_steps = 200
    max_denoising_steps = max(1, min(1000, max_denoising_steps))

    SELECTION_BY_SESSION[session_id] = BendSelection(
        bends=bends,
        steps_min=steps_min,
        steps_max=steps_max,
        max_denoising_steps=max_denoising_steps,
        selected_part=selected_part,
    )
    sel = SELECTION_BY_SESSION[session_id]
    steps_part = (getattr(sel, "steps_min", None), getattr(sel, "steps_max", None), getattr(sel, "max_denoising_steps", 200))
    change_hash = _bends_hash(sel.bends) + "|" + str(steps_part) + "|" + str(getattr(sel, "selected_part", None) or "")
    return web.json_response({"ok": True, "change_hash": change_hash})


@PromptServer.instance.routes.post("/web_bend_demo/clear")
async def api_clear_selection(request):
    data = await request.json()
    session_id = data.get("session_id", "")
    SELECTION_BY_SESSION.pop(session_id, None)
    return web.json_response({"ok": True})


# ----------------------------
# History persistence (web_bend_demo_history under ComfyUI output, keyed by session_id)
# ----------------------------

def _base_session_id(session_id: str) -> str:
    """Strip _hash_ / _ts_ suffixes; match node's session keying."""
    if not session_id:
        return ""
    base = session_id
    if "_hash_" in base:
        base = base.split("_hash_", 1)[0]
    if "_ts_" in base:
        base = base.split("_ts_", 1)[0]
    return base


def _safe_session_dir(session_id: str) -> str:
    """Filesystem-safe folder name from session_id."""
    s = _base_session_id(session_id)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in s)
    return safe or "default"


def _history_base_dir() -> str:
    """ComfyUI output dir + web_bend_demo_history subfolder."""
    if not folder_paths:
        raise RuntimeError("folder_paths not available")
    out = folder_paths.get_output_directory()
    base = os.path.join(out, "web_bend_demo_history")
    os.makedirs(base, exist_ok=True)
    return base


@PromptServer.instance.routes.get("/web_bend_demo/history/load")
async def api_history_load(request):
    """Load persisted history for session_id. Returns { ok, history }."""
    session_id = request.rel_url.query.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": True, "history": []})
    try:
        base = _history_base_dir()
        sub = _safe_session_dir(session_id)
        path = os.path.join(base, sub, "history.json")
        if not os.path.isfile(path):
            return web.json_response({"ok": True, "history": []})
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = data if isinstance(data, list) else (data.get("history") if isinstance(data, dict) else [])
        if not isinstance(history, list):
            history = []
        return web.json_response({"ok": True, "history": history})
    except Exception as e:
        import traceback
        return web.json_response(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status=500,
        )


@PromptServer.instance.routes.post("/web_bend_demo/history/save")
async def api_history_save(request):
    """Persist history for session_id. Body: { session_id, history }."""
    try:
        data = await request.json()
        session_id = data.get("session_id", "")
        history = data.get("history")
        if not session_id:
            return web.json_response({"ok": False, "error": "missing session_id"}, status=400)
        if history is None:
            history = []
        if not isinstance(history, list):
            return web.json_response({"ok": False, "error": "history must be an array"}, status=400)
        base = _history_base_dir()
        sub = _safe_session_dir(session_id)
        dir_path = os.path.join(base, sub)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return web.json_response({"ok": True})
    except Exception as e:
        import traceback
        return web.json_response(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status=500,
        )


# ----------------------------
# Experiments cache (batch pre-runs for interactive UI lookup)
# ----------------------------

def _experiments_base_dir() -> str:
    """ComfyUI output dir + web_bend_demo_experiments subfolder."""
    if not folder_paths:
        raise RuntimeError("folder_paths not available")
    out = folder_paths.get_output_directory()
    base = os.path.join(out, "web_bend_demo_experiments")
    os.makedirs(base, exist_ok=True)
    return base


def _experiment_lookup_key(bends: List[Dict[str, Any]], steps_min: Optional[int], steps_max: Optional[int], prompt_hash: str = "") -> str:
    """Canonical key for cache lookup. Must match experiment script."""
    steps_part = (steps_min, steps_max, 200)
    bends_str = json.dumps(bends, sort_keys=True)
    payload = bends_str + "|" + str(steps_part)
    if prompt_hash:
        payload += "|" + prompt_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_experiment_index(batch_id: str) -> Dict[str, Dict[str, Any]]:
    """Load results.jsonl and build lookup_key -> record index."""
    base = _experiments_base_dir()
    safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in (batch_id or "default"))
    path = os.path.join(base, safe_batch, "results.jsonl")
    if not os.path.isfile(path):
        return {}
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("lookup_key")
                if key:
                    index[key] = rec
            except json.JSONDecodeError:
                continue
    return index


@PromptServer.instance.routes.get("/web_bend_demo/experiments/lookup")
async def api_experiments_lookup(request):
    """Look up a cached experiment result. Returns image URL if hit."""
    session_id = request.rel_url.query.get("session_id", "")
    batch_id = request.rel_url.query.get("batch_id", "") or session_id
    prompt_hash = request.rel_url.query.get("prompt_hash", "")
    try:
        bends_raw = request.rel_url.query.get("bends", "[]")
        bends = json.loads(bends_raw) if bends_raw else []
    except json.JSONDecodeError:
        return web.json_response({"ok": False, "error": "invalid bends JSON"}, status=400)
    steps_min = request.rel_url.query.get("steps_min", "")
    steps_max = request.rel_url.query.get("steps_max", "")
    sm = int(steps_min) if steps_min and steps_min.isdigit() else None
    sx = int(steps_max) if steps_max and steps_max.isdigit() else None

    if not batch_id:
        return web.json_response({"ok": True, "hit": False})

    try:
        bends_norm = []
        for b in (bends or []):
            path = b.get("path")
            mt = (b.get("module_type") or "rotate").strip()
            ma = _normalize_module_args(b.get("module_args") or {})
            if path and isinstance(path, str):
                bends_norm.append({"path": path, "module_type": mt, "module_args": ma})
        lookup_key = _experiment_lookup_key(bends_norm, sm, sx, prompt_hash)
        index = _load_experiment_index(batch_id)
        rec = index.get(lookup_key)
        if not rec:
            return web.json_response({"ok": True, "hit": False})
        fn = rec.get("image_filename", "")
        subfolder = rec.get("image_subfolder", "")
        img_type = rec.get("image_type", "output")
        if not fn:
            return web.json_response({"ok": True, "hit": False})
        if subfolder == "images" and img_type == "experiment":
            image_url = f"/web_bend_demo/experiments/image?batch_id={batch_id}&filename={fn}"
        else:
            image_url = f"/view?filename={fn}&subfolder={subfolder}&type={img_type}"
        return web.json_response({
            "ok": True,
            "hit": True,
            "image_url": image_url,
            "experiment_id": rec.get("id", lookup_key),
        })
    except Exception as e:
        import traceback
        return web.json_response(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status=500,
        )


@PromptServer.instance.routes.get("/web_bend_demo/api/list")
async def api_explorer_list(request):
    """API-compatible list batches (for explorer.html)."""
    return await api_experiments_list(request)


@PromptServer.instance.routes.get("/web_bend_demo/api/data/{batch_id}")
async def api_explorer_data(request):
    """API-compatible batch data (for explorer.html). batch_id from path."""
    batch_id = request.match_info.get("batch_id", "")
    if not batch_id:
        return web.json_response({"ok": False, "error": "missing batch_id"}, status=400)
    try:
        base = _experiments_base_dir()
        safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        sub = os.path.join(base, safe_batch)
        if not os.path.isdir(sub):
            return web.json_response({"ok": False, "error": "batch not found"}, status=404)
        results_path = os.path.join(sub, "results.jsonl")
        results = []
        if os.path.isfile(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        manifest = {}
        manifest_path = os.path.join(sub, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                pass
        for r in results:
            fn = r.get("image_filename", "")
            subfolder = r.get("image_subfolder", "")
            img_type = r.get("image_type", "output")
            if fn:
                if subfolder == "images" and img_type == "experiment":
                    r["image_url"] = f"/web_bend_demo/experiments/image?batch_id={batch_id}&filename={fn}"
                else:
                    r["image_url"] = f"/view?filename={fn}&subfolder={subfolder}&type={img_type}"
            lat_fn = r.get("latent_filename", "")
            lat_subfolder = r.get("latent_subfolder", "")
            if lat_fn and lat_subfolder == "latents":
                r["latent_url"] = f"/web_bend_demo/experiments/latent?batch_id={batch_id}&filename={lat_fn}"
        return web.json_response({"ok": True, "manifest": manifest, "results": results})
    except Exception as e:
        import traceback
        return web.json_response(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status=500,
        )


@PromptServer.instance.routes.get("/web_bend_demo/experiments/list")
async def api_experiments_list(request):
    """List available experiment batches."""
    try:
        base = _experiments_base_dir()
        batches = []
        for name in os.listdir(base):
            sub = os.path.join(base, name)
            if not os.path.isdir(sub):
                continue
            manifest_path = os.path.join(sub, "manifest.json")
            info = {"batch_id": name}
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        info.update(json.load(f))
                except json.JSONDecodeError:
                    pass
            results_path = os.path.join(sub, "results.jsonl")
            count = 0
            if os.path.isfile(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
            info["result_count"] = count
            batches.append(info)
        return web.json_response({"ok": True, "batches": batches})
    except Exception as e:
        import traceback
        return web.json_response(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status=500,
        )


@PromptServer.instance.routes.get("/web_bend_demo/experiments/data")
async def api_experiments_data(request):
    """Return full experiment data for a batch (for exploration tool)."""
    batch_id = request.rel_url.query.get("batch_id", "")
    if not batch_id:
        return web.json_response({"ok": False, "error": "missing batch_id"}, status=400)
    try:
        base = _experiments_base_dir()
        safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        sub = os.path.join(base, safe_batch)
        if not os.path.isdir(sub):
            return web.json_response({"ok": False, "error": "batch not found"}, status=404)
        results_path = os.path.join(sub, "results.jsonl")
        results = []
        if os.path.isfile(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        manifest = {}
        manifest_path = os.path.join(sub, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                pass
        for r in results:
            fn = r.get("image_filename", "")
            subfolder = r.get("image_subfolder", "")
            img_type = r.get("image_type", "output")
            if fn:
                if subfolder == "images" and img_type == "experiment":
                    r["image_url"] = f"/web_bend_demo/experiments/image?batch_id={batch_id}&filename={fn}"
                else:
                    r["image_url"] = f"/view?filename={fn}&subfolder={subfolder}&type={img_type}"
            # Add latent URL if available
            lat_fn = r.get("latent_filename", "")
            lat_subfolder = r.get("latent_subfolder", "")
            if lat_fn and lat_subfolder == "latents":
                r["latent_url"] = f"/web_bend_demo/experiments/latent?batch_id={batch_id}&filename={lat_fn}"
        return web.json_response({"ok": True, "manifest": manifest, "results": results})
    except Exception as e:
        import traceback
        return web.json_response(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc()},
            status=500,
        )


@PromptServer.instance.routes.get("/web_bend_demo/experiments/image")
async def api_experiments_image(request):
    """Serve image from batch images/ folder (persistent, works without ComfyUI output)."""
    batch_id = request.rel_url.query.get("batch_id", "")
    filename = request.rel_url.query.get("filename", "")
    if not batch_id or not filename:
        return web.Response(status=400)
    if ".." in filename or "/" in filename or "\\" in filename:
        return web.Response(status=400)
    try:
        base = _experiments_base_dir()
        safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        path = os.path.join(base, safe_batch, "images", filename)
        if not os.path.isfile(path):
            return web.Response(status=404)
        with open(path, "rb") as f:
            body = f.read()
        return web.Response(body=body, content_type="image/png")
    except Exception:
        return web.Response(status=500)


@PromptServer.instance.routes.get("/web_bend_demo/experiments/latent")
async def api_experiments_latent(request):
    """Serve latent from batch latents/ folder (persistent, works without ComfyUI output)."""
    batch_id = request.rel_url.query.get("batch_id", "")
    filename = request.rel_url.query.get("filename", "")
    if not batch_id or not filename:
        return web.Response(status=400)
    if ".." in filename or "/" in filename or "\\" in filename:
        return web.Response(status=400)
    try:
        base = _experiments_base_dir()
        safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        path = os.path.join(base, safe_batch, "latents", filename)
        if not os.path.isfile(path):
            return web.Response(status=404)
        with open(path, "rb") as f:
            body = f.read()
        return web.Response(body=body, content_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception:
        return web.Response(status=500)


@PromptServer.instance.routes.get("/web_bend_demo/{path:.*}")
async def serve_static_files(request):
    """Serve static files from the web directory"""
    from pathlib import Path
    import mimetypes
    import os
    
    try:
        # Get the directory where this file is located
        # Handle both string and Path objects for __file__
        file_path_obj = __file__
        if isinstance(file_path_obj, str):
            current_dir = Path(file_path_obj).parent.resolve()
        else:
            current_dir = file_path_obj.parent.resolve()
        
        path = request.match_info.get("path", "")

        # Skip API routes - they should be handled by other routes
        # These should never reach here if routes are ordered correctly, but just in case
        api_routes = ["layers", "selection", "clear", "poll_image", "history", "experiments", "api"]
        if path in api_routes or any(path.startswith(api + "/") or path.startswith(api + "?") for api in api_routes):
            return web.Response(text="Not found", status=404)
        
        # Security: prevent directory traversal
        if ".." in path:
            return web.Response(text="Invalid path", status=400)
        
        # Normalize path separators and remove leading slashes
        path = path.replace("\\", "/").lstrip("/")
        
        # Default to index.html if no path
        if not path or path == "":
            path = "index.html"
        
        # Build file path
        file_path = (current_dir / "web" / path).resolve()
        
        # Security: ensure the file is within the web directory
        web_dir = (current_dir / "web").resolve()
        try:
            file_path.relative_to(web_dir)
        except ValueError:
            return web.Response(text="Invalid path: outside web directory", status=403)
        
        if not file_path.exists():
            return web.Response(text=f"File not found: {path} (resolved to: {file_path})", status=404)
        
        if not file_path.is_file():
            return web.Response(text=f"Path is not a file: {path}", status=404)
        
        # Determine content type
        guessed_type, encoding = mimetypes.guess_type(str(file_path))
        if not guessed_type:
            # Default content types for common file extensions
            ext = file_path.suffix.lower()
            content_types = {
                ".js": "application/javascript",
                ".css": "text/css",
                ".html": "text/html",
                ".json": "application/json",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
            }
            content_type = content_types.get(ext, "application/octet-stream")
        else:
            # Remove charset from content type if present (aiohttp doesn't want it in content_type)
            content_type = guessed_type.split(";")[0].strip()
        
        # Read file content as binary
        with open(file_path, "rb") as f:
            content = f.read()
        
        return web.Response(body=content, content_type=content_type)
        
    except Exception as e:
        import traceback
        error_msg = f"Error serving file: {str(e)}\n{traceback.format_exc()}"
        print(f"[web_bend_demo] {error_msg}")  # Log to console for debugging
        # Return a simpler error message to avoid exposing internal details
        return web.Response(text=f"Internal server error: {str(e)}", status=500)


# ----------------------------
# 6) The node: MODEL -> MODEL
# ----------------------------

class InteractiveBendingWebUI:
    """
    Minimal node: takes only MODEL as required input.
    'session_id' exists as an optional widget; frontend JS auto-fills and hides it.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }, "optional": {
                 "session_id": ("STRING", {"default": ""}),
                "selection_hash": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_bending_demo"

    @staticmethod
    def _base_session_id(session_id: str) -> str:
        if not session_id:
            return ""
        base = session_id
        if "_hash_" in base:
            base = base.split("_hash_", 1)[0]
        if "_ts_" in base:
            base = base.split("_ts_", 1)[0]
        return base


    @classmethod
    def IS_CHANGED(cls, model, session_id="", selection_hash=""):
        
        base = cls._base_session_id(session_id)
        if not base:
            return "no-session"
        sel = SELECTION_BY_SESSION.get(base, BendSelection())
        steps_part = (getattr(sel, "steps_min", None), getattr(sel, "steps_max", None), getattr(sel, "max_denoising_steps", 200))
        return _bends_hash(sel.bends) + "|" + str(steps_part) + "|" + str(getattr(sel, "selected_part", None) or "")

    def patch(self, model, session_id="", selection_hash=""):
        base = self._base_session_id(session_id)
        if base:
            MODEL_BY_SESSION[base] = model
        if not base:
            return (model,)
        sel = SELECTION_BY_SESSION.get(base)
        if sel is None or not sel.bends:
            return (model,)
        return apply_bends_to_model(
            model,
            sel.bends,
            steps_min=getattr(sel, "steps_min", None),
            steps_max=getattr(sel, "steps_max", None),
            max_denoising_steps=getattr(sel, "max_denoising_steps", 200),
            selected_part=getattr(sel, "selected_part", None),
        )


# ----------------------------
# Apply Bends from JSON (paste from web UI clipboard)
# ----------------------------

class ApplyBendsFromJSON:
    """
    Apply bends by pasting the JSON copied from the web interface (Copy Bends button).
    Same format as the clipboard: { "bends": [...], "steps_min": ?, "steps_max": ?, "max_denoising_steps": ? },
    plus the optional version 1.1 fields documented in parse_bends_json.
    {{a}} ... {{d}} placeholders are replaced by the optional a..d inputs, so sliders can drive the JSON.
    Outputs a report and resolved_json: the same bends with every layer, argument and default made explicit.
    """

    PLACEHOLDER = re.compile(r"\{\{\s*([abcd])\s*\}\}")

    @classmethod
    def INPUT_TYPES(cls):
        value = {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01,
                 "tooltip": "Replaces {{name}} in bends_json"}
        return {
            "required": {
                "model": ("MODEL",),
                "bends_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Paste JSON from web UI (Copy Bends)...",
                }),
            },
            "optional": {
                "strict": ("BOOLEAN", {"default": False,
                                       "tooltip": "Fail on any problem (unknown keys or arguments, paths that "
                                                  "cannot be bent, ...) instead of warning"}),
                "a": ("FLOAT", value),
                "b": ("FLOAT", value),
                "c": ("FLOAT", value),
                "d": ("FLOAT", value),
                "clamp": (["none", "hard", "safe"], {
                    "default": "none",
                    "tooltip": "hard: each op's absolute limits; safe: hard limits narrowed by safe_ranges"}),
                "safe_ranges": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": '{"multiply": {"scalar": [0, 3]}, "paths": {"output_blocks.*": '
                                   '{"multiply": {"scalar": [0.5, 1.5]}}}}',
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "report", "resolved_json")
    FUNCTION = "patch"
    CATEGORY = "model_bending_demo"

    def patch(self, model, bends_json, strict=False, a=0.0, b=0.0, c=0.0, d=0.0, clamp="none", safe_ranges=""):
        bends_json = (bends_json or "").strip()
        if not bends_json:
            return (model, json.dumps({"resolved": [], "note": "empty bends_json"}), json.dumps({"bends": []}))
        values = {"a": a, "b": b, "c": c, "d": d}
        used = sorted(set(self.PLACEHOLDER.findall(bends_json)))
        bends_json = self.PLACEHOLDER.sub(lambda mt: repr(float(values[mt.group(1)])), bends_json)
        ranges = None
        if (safe_ranges or "").strip():
            try:
                ranges = json.loads(safe_ranges)
            except json.JSONDecodeError as e:
                raise ValueError(f"safe_ranges is not valid JSON: {e}") from e
            if not isinstance(ranges, dict):
                raise ValueError("safe_ranges must be a JSON object")
        problems: List[str] = []
        try:
            bends, steps_min, steps_max, max_denoising_steps, selected_part = parse_bends_json(bends_json, problems)
        except ValueError as e:
            raise ValueError(f"Bends JSON error: {e}") from e
        report: Dict[str, Any] = {"placeholders": {k: values[k] for k in used}} if used else {}
        if clamp != "none":
            report["clamp"] = clamp
        resolved: List[Dict[str, Any]] = []
        (patched,) = apply_bends_to_model(
            model, bends,
            steps_min=steps_min,
            steps_max=steps_max,
            max_denoising_steps=max_denoising_steps,
            selected_part=selected_part,
            strict=strict,
            report=report,
            clamp=clamp,
            safe_ranges=ranges,
            problems=problems,
            resolved=resolved,
        )
        canonical = {"version": BENDS_JSON_VERSION, "bends": resolved, "steps_min": steps_min, "steps_max": steps_max,
                     "max_denoising_steps": max_denoising_steps, "selected_part": selected_part}
        canonical = {k: v for k, v in canonical.items() if v is not None}
        return (patched, json.dumps(report, indent=2), json.dumps(canonical, indent=2))


# Import standalone model bending nodes and merge their mappings
from . import model_bending_nodes
from . import probe_nodes

NODE_CLASS_MAPPINGS = {
    "InteractiveBendingWebUI": InteractiveBendingWebUI,
    "ApplyBendsFromJSON": ApplyBendsFromJSON,
}
NODE_CLASS_MAPPINGS.update(model_bending_nodes.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(probe_nodes.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractiveBendingWebUI": "Interactive Bending WebUI",
    "ApplyBendsFromJSON": "Apply Bends from JSON",
}
NODE_DISPLAY_NAME_MAPPINGS.update(probe_nodes.NODE_DISPLAY_NAME_MAPPINGS)
