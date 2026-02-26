#!/usr/bin/env python3
"""
Run systemic bending experiments: iterate over layers × bend types × levels × timestep ranges.
Writes results to web_bend_demo_experiments/{batch_id}/ for interactive UI cache lookup
and data exploration.

Usage:
    python run_bending_experiments.py --workflow workflow_api.json --batch-id my_batch [options]

Requirements:
    - ComfyUI running with a workflow that has InteractiveBendingWebUI
    - Workflow exported in API format (Enable Dev Mode → API (Dev) in ComfyUI)
    - Run the workflow once manually to register the model, or use --queue-once
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

# Bending config (must match config.js)
BENDING_TYPES2 = {
    "add_noise": {"levels": [0, 1, 2], "module_args_key": "noise_std", "defaultValue": 0},
    "multiply": {"levels": [0, 0.5, 1, 1.5, 2], "module_args_key": "scalar", "defaultValue": 1},
    "rotate": {"levels": [0, 90, 180, 270], "module_args_key": "angle_degrees", "defaultValue": 0},
}

BENDING_TYPES = {
    "add_noise": {"levels": [0, 1, 2], "module_args_key": "noise_std", "defaultValue": 0},
    "multiply": {"levels": [0, 0.5, 1, 1.5, 2], "module_args_key": "scalar", "defaultValue": 1},
    "rotate": {"levels": [0, 90, 180, 270], "module_args_key": "angle_degrees", "defaultValue": 0},
}

KNOWN_CONTAINERS = [
    "SpatialTransformer", "Upsample", "Downsample", "ResBlock",
    "TimestepEmbedSequential", "CrossAttention",
]


def container_from_type_path(type_path: str) -> str:
    if not type_path:
        return "Other"
    parts = type_path.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in KNOWN_CONTAINERS:
            return parts[i]
    return parts[-2] if len(parts) >= 2 else (parts[0] or "Other")


def block_region_from_path(path: str) -> str:
    parts = path.split(".")
    if not parts:
        return "other"
    first = parts[0].lower()
    if "input" in first or "down" in first:
        return "input"
    if "output" in first or "up" in first:
        return "output"
    if "middle" in first or "mid" in first:
        return "middle"
    return "other"


def block_name_from_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) < 2:
        return path
    return ".".join(parts[:2])


def flatten_tree(node: dict, type_prefix: str = "") -> list:
    """Flatten module tree to list of {path, type, full_type, type_path}."""
    out = []
    name = node.get("name", "")
    path = node.get("path", "")
    p_type = node.get("type", "")
    full_type = node.get("full_type", p_type)
    type_path = f"{type_prefix}.{p_type}" if type_prefix else p_type
    if node.get("children"):
        for child in node["children"]:
            out.extend(flatten_tree(child, type_path))
    else:
        if path and path != "root" and not path.startswith("root."):
            out.append({
                "path": path.replace("root.", "") if path.startswith("root.") else path,
                "type": p_type,
                "full_type": full_type,
                "type_path": type_path,
            })
    return out


def experiment_lookup_key(bends: list, steps_min: Optional[int], steps_max: Optional[int]) -> str:
    """Must match backend _experiment_lookup_key (no prompt_hash for simpler cache match)."""
    steps_part = (steps_min, steps_max, 200)
    bends_str = json.dumps(bends, sort_keys=True)
    payload = bends_str + "|" + str(steps_part)
    return hashlib.sha256(payload.encode()).hexdigest()


def find_bending_node(workflow: dict) -> tuple:
    """Find InteractiveBendingWebUI node. Returns (node_id, session_id)."""
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "InteractiveBendingWebUI":
            inputs = node.get("inputs", {})
            sid = inputs.get("session_id", "")
            if not sid:
                sid = str(hash(str(nid) + str(time.time())))[:8]
            return str(nid), sid
    return None, None


def get_experiments_base_dir() -> Path:
    """Resolve experiments dir (ComfyUI output/web_bend_demo_experiments)."""
    out = os.environ.get("COMFYUI_OUTPUT", "")
    if not out:
        # Assume script is in ComfyUI/custom_nodes/ComfyUI-Web-Bend-Demo/scripts/
        base = Path(__file__).resolve().parent.parent.parent.parent
        out = base / "output"
    return Path(out) / "web_bend_demo_experiments"


def main():
    ap = argparse.ArgumentParser(description="Run bending experiments for ComfyUI-Web-Bend-Demo")
    ap.add_argument("--workflow", "-w", required=True, help="Workflow API JSON path")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    ap.add_argument("--batch-id", "-b", default="default", help="Experiment batch ID (use session_id from workflow node for interactive UI cache lookup)")
    ap.add_argument("--prompt", "-p", default="A floating orb", help="Prompt for generation")
    ap.add_argument("--layers-limit", type=int, default=0, help="Max layers to test (0=all)")
    ap.add_argument("--bend-types", nargs="+", default=["multiply"], help="Bend types: rotate add_noise multiply")
    ap.add_argument("--steps-ranges", nargs="+", default=["*"], help="e.g. '*' or '0-20' '20-50'")
    ap.add_argument("--skip-defaults", action="store_true", help="Skip levels that equal default (no bend)")
    ap.add_argument("--include-default", action="store_true", default=True, help="Run unbent (default) generation first for metrics comparison")
    ap.add_argument("--no-include-default", action="store_false", dest="include_default", help="Do not run unbent generation")
    ap.add_argument("--queue-once", action="store_true", help="Queue empty run first to register model")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only, no execution")
    ap.add_argument("--export-to", metavar="DIR", default="", help="After run, export batch to this directory (same as export_experiments -o)")
    ap.add_argument("--export-no-explorer", action="store_true", help="With --export-to: do not copy explorer HTML")
    args = ap.parse_args()

    base_url = args.comfy_url.rstrip("/")
    api = f"{base_url}/web_bend_demo"

    with open(args.workflow, "r", encoding="utf-8") as f:
        workflow_data = json.load(f)
    workflow = workflow_data.get("prompt", workflow_data) if isinstance(workflow_data, dict) else workflow_data
    if not isinstance(workflow, dict):
        print("Invalid workflow: expected dict of nodes")
        sys.exit(1)

    node_id, session_id = find_bending_node(workflow)
    if not node_id:
        print("No InteractiveBendingWebUI node found in workflow")
        sys.exit(1)
    if args.batch_id == "default" and session_id:
        args.batch_id = session_id
        print(f"Using session_id as batch_id for cache lookup: {args.batch_id}")
    print(f"Found bending node {node_id}, session_id={session_id}")
    
    # Extract prompt text from CLIPTextEncode nodes (positive prompt)
    prompt_text = ""
    for nid, n in workflow.items():
        if n.get("class_type") == "CLIPTextEncode":
            text = n.get("inputs", {}).get("text", "")
            if text:
                prompt_text = text
                break  # Use first non-empty prompt

    # Extract sampler parameters from KSampler node
    sampler_params = {}
    for nid, n in workflow.items():
        if n.get("class_type") in ("KSampler", "KSamplerAdvanced"):
            inp = n.get("inputs", {})
            sampler_params = {
                "seed": inp.get("seed"),
                "steps": inp.get("steps"),
                "cfg": inp.get("cfg"),
                "sampler_name": inp.get("sampler_name"),
                "scheduler": inp.get("scheduler"),
                "denoise": inp.get("denoise"),
            }
            break

    # Extract checkpoint name from CheckpointLoaderSimple
    ckpt_name = ""
    for nid, n in workflow.items():
        if n.get("class_type") in ("CheckpointLoaderSimple", "CheckpointLoader"):
            ckpt_name = n.get("inputs", {}).get("ckpt_name", "")
            if ckpt_name:
                break

    # Extract image dimensions from EmptyLatentImage
    image_size = {}
    for nid, n in workflow.items():
        if n.get("class_type") == "EmptyLatentImage":
            inp = n.get("inputs", {})
            image_size = {"width": inp.get("width"), "height": inp.get("height")}
            break

    # Ensure node has session_id input
    if "inputs" not in workflow[node_id]:
        workflow[node_id]["inputs"] = {}
    workflow[node_id]["inputs"]["session_id"] = session_id

    # Fetch layers (model must be registered)
    r = requests.get(f"{api}/layers", params={"session_id": session_id}, timeout=30)
    if r.status_code != 200:
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        err = j.get("error", r.text)
        print(f"Failed to fetch layers: {err}")
        if "Unknown session_id" in str(err) and not args.dry_run:
            print("Tip: Run the workflow once in ComfyUI to register the model, or use --queue-once")
        sys.exit(1)
    data = r.json()
    tree = data.get("tree", {})
    print("tree", tree)
    flat = flatten_tree(tree)
    layers = [x for x in flat if x.get("path")]
    print("-------------layers", layers)
    if args.layers_limit:
        layers = layers[: args.layers_limit]
    print(f"Found {len(layers)} bendable layers")

    # Build metadata for each layer
    for layer in layers:
        path = layer["path"]
        layer["block_region"] = block_region_from_path(path)
        layer["block_name"] = block_name_from_path(path)
        layer["container_type"] = container_from_type_path(layer.get("type_path", ""))

    effective_prompt = prompt_text if prompt_text else args.prompt
    prompt_hash = hashlib.sha256(effective_prompt.encode()).hexdigest()[:16]

    # Build experiment grid
    experiments = []
    for layer in layers:
        path = layer["path"]
        for bend_type in args.bend_types:
            if bend_type not in BENDING_TYPES:
                continue
            cfg = BENDING_TYPES[bend_type]
            key = cfg["module_args_key"]
            default = cfg["defaultValue"]
            for level in cfg["levels"]:
                if args.skip_defaults and level == default:
                    continue
                module_args = {key: level}
                for sr in args.steps_ranges:
                    if sr == "*":
                        steps_min, steps_max = None, None
                    else:
                        try:
                            a, b = sr.split("-")
                            steps_min, steps_max = int(a.strip()), int(b.strip())
                        except ValueError:
                            steps_min, steps_max = None, None
                    bends = [{"path": path, "module_type": bend_type, "module_args": module_args}]
                    lookup_key = experiment_lookup_key(bends, steps_min, steps_max)
                    experiments.append({
                        "layer_path": path,
                        "block_region": layer["block_region"],
                        "block_name": layer["block_name"],
                        "container_type": layer["container_type"],
                        "layer_type": layer.get("type", ""),
                        "type_path": layer.get("type_path", ""),
                        "bend_module_type": bend_type,
                        "module_args": module_args,
                        "steps_min": steps_min,
                        "steps_max": steps_max,
                        "prompt": effective_prompt,
                        "prompt_hash": prompt_hash,
                        "lookup_key": lookup_key,
                    })
    print(f"Planned {len(experiments)} experiments")
    if args.dry_run:
        print(f"Batch ID (for real run): {args.batch_id}")
        for i, ex in enumerate(experiments[:5]):
            print(f"  {i+1}. {ex}")
        if len(experiments) > 5:
            print(f"  ... and {len(experiments)-5} more")
        return

    # Queue once to register model if requested
    if args.queue_once:
        workflow[node_id]["inputs"]["selection_hash"] = "init"
        r = requests.post(f"{base_url}/prompt", json={"prompt": workflow}, timeout=60)
        if r.status_code != 200:
            print(f"Queue init failed: {r.text}")
        else:
            pid = r.json().get("prompt_id", "")
            print(f"Queued init, prompt_id={pid}")
            for _ in range(60):
                time.sleep(2)
                rh = requests.get(f"{base_url}/history/{pid}", timeout=10)
                if rh.status_code == 200:
                    break
            time.sleep(1)

    # Output dir
    exp_base = get_experiments_base_dir()
    safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in args.batch_id)
    batch_dir = exp_base / safe_batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    results_path = batch_dir / "results.jsonl"

    completed = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed.add(rec.get("lookup_key", ""))
                    except json.JSONDecodeError:
                        pass
    print(f"Resuming: {len(completed)} already completed")

    results_file = open(results_path, "a", encoding="utf-8")

    # Run default (unbent) generation first for metrics comparison
    DEFAULT_LOOKUP_KEY = experiment_lookup_key([], None, None)
    if args.include_default and DEFAULT_LOOKUP_KEY not in completed:
        print("Running default (unbent) generation...")
        r = requests.post(
            f"{api}/selection",
            json={"session_id": session_id, "bends": [], "steps_min": None, "steps_max": None},
            timeout=10,
        )
        if r.status_code == 200:
            change_hash = r.json().get("change_hash", DEFAULT_LOOKUP_KEY)
            workflow[node_id]["inputs"]["selection_hash"] = change_hash
            rq = requests.post(f"{base_url}/prompt", json={"prompt": workflow}, timeout=60)
            if rq.status_code == 200:
                pid = rq.json().get("prompt_id", "")
                img_info = None
                latent_info = None
                for _ in range(120):
                    time.sleep(2)
                    rh = requests.get(f"{base_url}/history/{pid}", timeout=10)
                    if rh.status_code == 200:
                        hist = rh.json()
                        if pid in hist:
                            for nid, out in hist[pid].get("outputs", {}).items():
                                if out.get("latents") and not latent_info:
                                    latent_info = out["latents"][0]
                                if out.get("images") and not img_info:
                                    img_info = out["images"][0]
                    if img_info:
                        break
                if img_info:
                    images_dir = batch_dir / "images"
                    images_dir.mkdir(exist_ok=True)
                    local_filename = f"{DEFAULT_LOOKUP_KEY}.png"
                    local_path = images_dir / local_filename
                    rec = {
                        "is_default": True,
                        "lookup_key": DEFAULT_LOOKUP_KEY,
                        "layer_path": "",
                        "block_region": "", "block_name": "", "container_type": "", "layer_type": "",
                        "type_path": "", "bend_module_type": "", "module_args": {},
                        "steps_min": None, "steps_max": None, "prompt": effective_prompt,
                        "prompt_hash": prompt_hash, "latent_filename": "", "latent_subfolder": "",
                    }
                    try:
                        view_url = f"{base_url}/view?filename={requests.utils.quote(img_info.get('filename', ''))}&subfolder={requests.utils.quote(img_info.get('subfolder', ''))}&type={img_info.get('type', 'output')}"
                        img_r = requests.get(view_url, timeout=30)
                        if img_r.status_code == 200:
                            with open(local_path, "wb") as f:
                                f.write(img_r.content)
                            rec.update({"image_filename": local_filename, "image_subfolder": "images", "image_type": "experiment"})
                        else:
                            rec.update({"image_filename": img_info.get("filename", ""), "image_subfolder": img_info.get("subfolder", ""), "image_type": img_info.get("type", "output")})
                    except Exception as e:
                        print(f"  Warning: could not copy default image: {e}")
                        rec.update({"image_filename": img_info.get("filename", ""), "image_subfolder": img_info.get("subfolder", ""), "image_type": img_info.get("type", "output")})
                    if latent_info:
                        latents_dir = batch_dir / "latents"
                        latents_dir.mkdir(exist_ok=True)
                        local_latent_filename = f"{DEFAULT_LOOKUP_KEY}.latent"
                        latent_src = get_experiments_base_dir().parent / latent_info.get("subfolder", "latents") / latent_info.get("filename", "")
                        if latent_src.is_file():
                            shutil.copy2(latent_src, latents_dir / local_latent_filename)
                            rec["latent_filename"] = local_latent_filename
                            rec["latent_subfolder"] = "latents"
                    results_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    results_file.flush()
                    completed.add(DEFAULT_LOOKUP_KEY)
                    print("  Default generation saved.")
                else:
                    print("  Default generation timeout.")
            else:
                print("  Default queue failed:", rq.text)
        else:
            print("  Default selection failed:", r.text)

    try:
        for i, ex in enumerate(experiments):
            if ex["lookup_key"] in completed:
                continue
            bends = [
                {"path": ex["layer_path"], "module_type": ex["bend_module_type"], "module_args": ex["module_args"]}
            ]
            # POST selection
            r = requests.post(
                f"{api}/selection",
                json={
                    "session_id": session_id,
                    "bends": bends,
                    "steps_min": ex["steps_min"],
                    "steps_max": ex["steps_max"],
                },
                timeout=10,
            )
            if r.status_code != 200:
                print(f"[{i+1}/{len(experiments)}] Selection failed: {r.text}")
                continue
            change_hash = r.json().get("change_hash", ex["lookup_key"])
            workflow[node_id]["inputs"]["selection_hash"] = change_hash
            # Queue
            r = requests.post(f"{base_url}/prompt", json={"prompt": workflow}, timeout=60)
            if r.status_code != 200:
                print(f"[{i+1}/{len(experiments)}] Queue failed: {r.text}")
                continue
            pid = r.json().get("prompt_id", "")
            # Poll
            img_info = None
            latent_info = None
            for _ in range(120):
                time.sleep(2)
                rh = requests.get(f"{base_url}/history/{pid}", timeout=10)
                if rh.status_code != 200:
                    continue
                hist = rh.json()
                if pid not in hist:
                    continue
                entry = hist[pid]
                outputs = entry.get("outputs", {})
                for nid, out in outputs.items():
                    lats = out.get("latents", [])
                    imgs = out.get("images", [])
                    if lats and not latent_info:
                        latent_info = lats[0]
                    if imgs and not img_info:
                        img_info = imgs[0]
                if img_info:
                    break
            if not img_info:
                print(f"[{i+1}/{len(experiments)}] Timeout: {ex['layer_path']} {ex['bend_module_type']} {ex['module_args']}")
                continue
            # Copy image into batch folder for persistence (works without ComfyUI)
            images_dir = batch_dir / "images"
            images_dir.mkdir(exist_ok=True)
            local_filename = f"{ex['lookup_key']}.png"
            local_path = images_dir / local_filename
            rec = {**ex, "latent_filename": "", "latent_subfolder": ""}
            try:
                view_url = f"{base_url}/view?filename={requests.utils.quote(img_info.get('filename', ''))}&subfolder={requests.utils.quote(img_info.get('subfolder', ''))}&type={img_info.get('type', 'output')}"
                img_r = requests.get(view_url, timeout=30)
                if img_r.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(img_r.content)
                    rec.update({"image_filename": local_filename, "image_subfolder": "images", "image_type": "experiment"})
                else:
                    rec.update({"image_filename": img_info.get("filename", ""), "image_subfolder": img_info.get("subfolder", ""), "image_type": img_info.get("type", "output")})
            except Exception as e:
                print(f"  Warning: could not copy image: {e}")
                rec.update({"image_filename": img_info.get("filename", ""), "image_subfolder": img_info.get("subfolder", ""), "image_type": img_info.get("type", "output")})
            
            # Copy latent into batch folder for persistence
            if latent_info:
                latents_dir = batch_dir / "latents"
                latents_dir.mkdir(exist_ok=True)
                local_latent_filename = f"{ex['lookup_key']}.latent"
                local_latent_path = latents_dir / local_latent_filename
                try:
                    # SaveLatent files are in ComfyUI output folder
                    out_base = get_experiments_base_dir().parent
                    latent_subfolder = latent_info.get("subfolder", "latents")
                    latent_fn = latent_info.get("filename", "")
                    latent_src = out_base / latent_subfolder / latent_fn
                    if latent_src.is_file():
                        shutil.copy2(latent_src, local_latent_path)
                        rec["latent_filename"] = local_latent_filename
                        rec["latent_subfolder"] = "latents"
                    else:
                        print(f"  Warning: latent file not found: {latent_src}")
                except Exception as e:
                    print(f"  Warning: could not copy latent: {e}")
            
            results_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results_file.flush()
            completed.add(ex["lookup_key"])
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{len(experiments)}")
    finally:
        results_file.close()

    manifest = {
        "batch_id": args.batch_id,
        "session_id": session_id,
        "prompt": prompt_text if prompt_text else args.prompt,
        "ckpt_name": ckpt_name,
        **sampler_params,
        **image_size,
        "experiment_count": len(experiments),
        "completed_count": len(completed),
        "default_lookup_key": DEFAULT_LOOKUP_KEY,
    }
    with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Results: {results_path}")

    if args.export_to:
        try:
            from export_experiments import export_batch
            print(f"Exporting to {args.export_to}...")
            export_batch(
                args.batch_id,
                args.export_to,
                layers_limit=args.layers_limit or 0,
                no_explorer=args.export_no_explorer,
            )
        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
