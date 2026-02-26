#!/usr/bin/env python3
"""
Compute cosine distance (latent), LPIPS (image), DINOv2-base and CLIP (image) feature distances
between each bent result and the default unbent. Writes cosine_distance, lpips_distance,
dinov2_distance, and clip_distance into each result. Run as part of export or standalone.

Usage:
  python compute_metrics.py --dir /path/to/export_folder
  python compute_metrics.py --batch-dir /path/to/web_bend_demo_experiments/batch_id

Requires: pip install lpips
DINOv2-base: torch.hub (facebookresearch/dinov2, dinov2_vitb14).
CLIP: pip install transformers (openai/clip-vit-base-patch32).
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

try:
    import lpips
except ImportError:
    print("LPIPS required: pip install lpips")
    sys.exit(1)

try:
    import safetensors.torch
except ImportError:
    print("safetensors required: pip install safetensors")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"PIL/numpy required: pip install Pillow numpy ({e})")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_latent(path: Path, device: torch.device = None) -> torch.Tensor:
    """Load .latent file (safetensors) and return latent_tensor on device."""
    device = device or DEVICE
    data = safetensors.torch.load_file(str(path), device="cpu")
    t = data["latent_tensor"].float()
    if "latent_format_version_0" not in data:
        t = t * (1.0 / 0.18215)
    return t.to(device)



def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine_similarity. Returns scalar in [0, 2]."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    if a_flat.numel() != b_flat.numel():
        return float("nan")
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1).item()
    return float(1.0 - cos_sim)


def load_image_as_tensor(path: Path, device: torch.device = None) -> torch.Tensor:
    """Load PNG, convert to RGB, normalize to [-1, 1], shape [1, 3, H, W]."""
    device = device or DEVICE
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    arr = (arr * 2.0 - 1.0).to(device)
    return arr


def compute_lpips(loss_fn, img_a: torch.Tensor, img_b: torch.Tensor, target_size: int = 256) -> float:
    """LPIPS between two images. Resize to target_size if needed."""
    if img_a.shape != img_b.shape:
        min_h = min(img_a.shape[2], img_b.shape[2])
        min_w = min(img_a.shape[3], img_b.shape[3])
        img_a = F.interpolate(img_a, size=(min_h, min_w), mode="bilinear", align_corners=False)
        img_b = F.interpolate(img_b, size=(min_h, min_w), mode="bilinear", align_corners=False)
    h, w = img_a.shape[2], img_a.shape[3]
    if h != target_size or w != target_size:
        img_a = F.interpolate(img_a, size=(target_size, target_size), mode="bilinear", align_corners=False)
        img_b = F.interpolate(img_b, size=(target_size, target_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        d = loss_fn(img_a, img_b).item()
    return float(d)


# DINOv2-base: ImageNet normalization and input size
DINOV2_SIZE = 224
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def load_image_for_dinov2(path: Path, device: torch.device = None) -> torch.Tensor:
    """Load image for DINOv2: RGB [0,1], resize to 224, ImageNet normalize, [1, 3, H, W]."""
    device = device or DEVICE
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    arr = arr.to(device)
    arr = F.interpolate(arr, size=(DINOV2_SIZE, DINOV2_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(DINOV2_MEAN, dtype=arr.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINOV2_STD, dtype=arr.dtype, device=device).view(1, 3, 1, 1)
    arr = (arr - mean) / std
    return arr


def get_dinov2_model(device: torch.device = None):
    """Load DINOv2-base (ViT-B/14) from torch.hub. Cached after first run."""
    device = device or DEVICE
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False)
    model = model.to(device).eval()
    return model


def compute_dinov2_distance(model, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """Cosine distance (1 - cos_sim) between DINOv2 feature vectors of two images.
    img_a, img_b: [1, 3, 224, 224] in ImageNet-normalized form."""
    with torch.no_grad():
        feat_a = model(img_a)
        feat_b = model(img_b)
    # model returns [1, 768] CLS token embedding
    return cosine_distance(feat_a, feat_b)


# CLIP (transformers): ViT-B/32, OpenAI
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def get_clip_model(device: torch.device = None):
    """Load CLIP ViT-B/32 (OpenAI) via transformers. Requires: pip install transformers."""
    device = device or DEVICE
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def load_image_for_clip(path: Path, processor, device: torch.device = None) -> torch.Tensor:
    """Load image and preprocess for CLIP. Returns [1, C, H, W] pixel_values on device."""
    device = device or DEVICE
    img = Image.open(path).convert("RGB")
    out = processor(images=img, return_tensors="pt")
    return out["pixel_values"].to(device)


def compute_clip_distance2(model, pixel_a: torch.Tensor, pixel_b: torch.Tensor) -> float:
    """Cosine distance (1 - cos_sim) between CLIP image feature vectors."""
    with torch.no_grad():
        feat_a = model.get_image_features(pixel_a)
        feat_b = model.get_image_features(pixel_b)
    return cosine_distance(feat_a, feat_b)

def compute_clip_distance(model, pixel_a, pixel_b) -> float:
    with torch.no_grad():
        feat_a = F.normalize(model.get_image_features(pixel_a).pooler_output, dim=-1)
        feat_b = F.normalize(model.get_image_features(pixel_b).pooler_output, dim=-1)
        cos_sim = (feat_a * feat_b).sum(dim=-1).item()
    return float(1.0 - cos_sim)


def find_default_entry(results: list, default_lookup_key: str = None) -> dict:
    """Find default (unbent) entry in results."""
    for r in results:
        if r.get("is_default"):
            return r
        if default_lookup_key and r.get("lookup_key") == default_lookup_key:
            return r
    # Fallback: empty bends
    for r in results:
        bends = r.get("bends", [])
        if bends == [] or (not r.get("bend_module_type") and not r.get("layer_path")):
            return r
    return None


def run_on_export_dir(export_dir: Path) -> bool:
    """Compute metrics for export folder (data.json, images/, latents/)."""
    device = DEVICE
    print(f"Using device: {device}")

    data_path = export_dir / "data.json"
    if not data_path.is_file():
        print(f"No data.json in {export_dir}")
        return False
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    manifest = data.get("manifest", {})
    default_lookup_key = manifest.get("default_lookup_key")
    default_entry = find_default_entry(results, default_lookup_key)
    if not default_entry:
        print("No default (unbent) entry found. Run experiments with --include-default.")
        return False

    images_dir = export_dir / "images"
    latents_dir = export_dir / "latents"
    def_img_fn = default_entry.get("image_filename") or (default_entry.get("lookup_key", "") + ".png")
    default_img_path = images_dir / def_img_fn
    def_lat_fn = default_entry.get("latent_filename") or (default_entry.get("lookup_key", "") + ".latent")
    default_latent_path = latents_dir / def_lat_fn

    has_latent = default_latent_path.is_file()
    has_image = default_img_path.is_file()
    if not has_image:
        print(f"Default image not found: {default_img_path}")
        return False

    default_img = load_image_as_tensor(default_img_path, device) if has_image else None
    default_latent = load_latent(default_latent_path, device) if has_latent else None
    loss_fn = lpips.LPIPS(net="alex").eval().to(device)

    dinov2_model = None
    default_dinov2_img = None
    if has_image and default_img_path.is_file():
        try:
            dinov2_model = get_dinov2_model(device)
            default_dinov2_img = load_image_for_dinov2(default_img_path, device)
        except Exception as e:
            print(f"  Warning: DINOv2 model/default load failed: {e}")

    clip_model = None
    clip_preprocess = None
    default_clip_img = None
    if has_image and default_img_path.is_file():
        try:
            clip_model, clip_preprocess = get_clip_model(device)
            default_clip_img = load_image_for_clip(default_img_path, clip_preprocess, device)
        except Exception as e:
            print(f"  Warning: CLIP model/default load failed: {e}")

    bent_results = [r for r in results if not r.get("is_default")]
    updated = 0
    for r in bent_results:
        img_fn = r.get("image_filename") or (r.get("lookup_key", "") + ".png")
        img_path = images_dir / img_fn
        lat_fn = r.get("latent_filename") or (r.get("lookup_key", "") + ".latent")
        lat_path = latents_dir / lat_fn

        cos_dist = None
        if has_latent and default_latent is not None and lat_path.is_file():
            try:
                lat = load_latent(lat_path, device)
                cos_dist = cosine_distance(default_latent, lat)
            except Exception as e:
                print(f"  Warning: cosine for {r.get('lookup_key')}: {e}")
        r["cosine_distance"] = round(cos_dist, 6) if cos_dist is not None else None

        lpips_dist = None
        if has_image and img_path.is_file():
            try:
                img = load_image_as_tensor(img_path, device)
                lpips_dist = compute_lpips(loss_fn, default_img, img)
            except Exception as e:
                print(f"  Warning: LPIPS for {r.get('lookup_key')}: {e}")
        r["lpips_distance"] = round(lpips_dist, 6) if lpips_dist is not None else None

        dinov2_dist = None
        if dinov2_model is not None and default_dinov2_img is not None and img_path.is_file():
            try:
                bent_dinov2_img = load_image_for_dinov2(img_path, device)
                dinov2_dist = compute_dinov2_distance(dinov2_model, default_dinov2_img, bent_dinov2_img)
            except Exception as e:
                print(f"  Warning: DINOv2 for {r.get('lookup_key')}: {e}")
        r["dinov2_distance"] = round(dinov2_dist, 6) if dinov2_dist is not None else None

        clip_dist = None
        if clip_model is not None and clip_preprocess is not None and default_clip_img is not None and img_path.is_file():
            try:
                bent_clip_img = load_image_for_clip(img_path, clip_preprocess, device)
                clip_dist = compute_clip_distance(clip_model, default_clip_img, bent_clip_img)
            except Exception as e:
                print(f"  Warning: CLIP for {r.get('lookup_key')}: {e}")
        r["clip_distance"] = round(clip_dist, 6) if clip_dist is not None else None

        if cos_dist is not None or lpips_dist is not None or dinov2_dist is not None or clip_dist is not None:
            updated += 1

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Computed metrics for {updated}/{len(bent_results)} bent results.")
    return True


def run_on_batch_dir(batch_dir: Path) -> bool:
    """Compute metrics for batch folder (results.jsonl, manifest.json, images/, latents/)."""
    device = DEVICE
    print(f"Using device: {device}")

    results_path = batch_dir / "results.jsonl"
    manifest_path = batch_dir / "manifest.json"
    if not results_path.is_file():
        print(f"No results.jsonl in {batch_dir}")
        return False
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    default_lookup_key = None
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                default_lookup_key = json.load(f).get("default_lookup_key")
        except json.JSONDecodeError:
            pass
    default_entry = find_default_entry(results, default_lookup_key)
    if not default_entry:
        print("No default (unbent) entry found. Run experiments with --include-default.")
        return False

    images_dir = batch_dir / "images"
    latents_dir = batch_dir / "latents"
    def_img_fn = default_entry.get("image_filename") or default_entry.get("lookup_key", "") + ".png"
    default_img_path = images_dir / def_img_fn
    def_lat_fn = default_entry.get("latent_filename") or default_entry.get("lookup_key", "") + ".latent"
    default_latent_path = latents_dir / def_lat_fn

    has_latent = default_latent_path.is_file()
    has_image = default_img_path.is_file()
    if not has_image:
        print(f"Default image not found: {default_img_path}")
        return False

    default_img = load_image_as_tensor(default_img_path, device)
    default_latent = load_latent(default_latent_path, device) if has_latent else None
    loss_fn = lpips.LPIPS(net="alex").eval().to(device)

    dinov2_model = None
    default_dinov2_img = None
    if default_img_path.is_file():
        try:
            dinov2_model = get_dinov2_model(device)
            default_dinov2_img = load_image_for_dinov2(default_img_path, device)
        except Exception as e:
            print(f"  Warning: DINOv2 model/default load failed: {e}")

    clip_model = None
    clip_preprocess = None
    default_clip_img = None
    if default_img_path.is_file():
        try:
            clip_model, clip_preprocess = get_clip_model(device)
            default_clip_img = load_image_for_clip(default_img_path, clip_preprocess, device)
        except Exception as e:
            print(f"  Warning: CLIP model/default load failed: {e}")

    bent_results = [r for r in results if not r.get("is_default")]
    updated = 0
    for r in bent_results:
        img_fn = r.get("image_filename") or (r.get("lookup_key", "") + ".png")
        img_path = images_dir / img_fn
        lat_fn = r.get("latent_filename") or (r.get("lookup_key", "") + ".latent")
        lat_path = latents_dir / lat_fn

        cos_dist = None
        if has_latent and lat_path.is_file() and default_latent is not None:
            try:
                lat = load_latent(lat_path, device)
                cos_dist = cosine_distance(default_latent, lat)
            except Exception as e:
                print(f"  Warning: cosine for {r.get('lookup_key')}: {e}")
        r["cosine_distance"] = round(cos_dist, 6) if cos_dist is not None else None

        lpips_dist = None
        if img_path.is_file():
            try:
                img = load_image_as_tensor(img_path, device)
                lpips_dist = compute_lpips(loss_fn, default_img, img)
            except Exception as e:
                print(f"  Warning: LPIPS for {r.get('lookup_key')}: {e}")
        r["lpips_distance"] = round(lpips_dist, 6) if lpips_dist is not None else None

        dinov2_dist = None
        if dinov2_model is not None and default_dinov2_img is not None and img_path.is_file():
            try:
                bent_dinov2_img = load_image_for_dinov2(img_path, device)
                dinov2_dist = compute_dinov2_distance(dinov2_model, default_dinov2_img, bent_dinov2_img)
            except Exception as e:
                print(f"  Warning: DINOv2 for {r.get('lookup_key')}: {e}")
        r["dinov2_distance"] = round(dinov2_dist, 6) if dinov2_dist is not None else None

        clip_dist = None
        if clip_model is not None and clip_preprocess is not None and default_clip_img is not None and img_path.is_file():
            try:
                bent_clip_img = load_image_for_clip(img_path, clip_preprocess, device)
                clip_dist = compute_clip_distance(clip_model, default_clip_img, bent_clip_img)
            except Exception as e:
                print(f"  Warning: CLIP for {r.get('lookup_key')}: {e}")
        r["clip_distance"] = round(clip_dist, 6) if clip_dist is not None else None

        if cos_dist is not None or lpips_dist is not None or dinov2_dist is not None or clip_dist is not None:
            updated += 1

    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Computed metrics for {updated}/{len(bent_results)} bent results.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Compute cosine distance and LPIPS vs default unbent")
    ap.add_argument("--dir", help="Export folder (data.json + images/ + latents/)")
    ap.add_argument("--batch-dir", help="Batch folder (results.jsonl + manifest + images/ + latents/)")
    args = ap.parse_args()
    if args.dir:
        ok = run_on_export_dir(Path(args.dir).resolve())
    elif args.batch_dir:
        ok = run_on_batch_dir(Path(args.batch_dir).resolve())
    else:
        print("Specify --dir (export folder) or --batch-dir (batch folder)")
        sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
