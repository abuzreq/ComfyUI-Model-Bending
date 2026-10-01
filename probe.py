# Activation probe and steering-vector core.
# Kept free of ComfyUI node classes and server imports so both the nodes (probe_nodes.py) and,
# later, the interactive web UI can attach probes on demand.
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bendutils import (
    LOG_TAG,
    current_step,
    expand_path_pattern,
    in_window,
    infer_token_grid,
    model_sampling_of,
    normalized_t,
    resolve_hook_target,
    t_window_from,
    total_steps,
    warn,
    with_model_sampling,
)

STAT_NAMES = ("mean", "std", "rms", "absmax", "nonfinite", "dead_frac")
DEFAULT_UNET_SITES = "input_blocks.*.0, middle_block.1, output_blocks.*.0"
SPATIAL_MAX_ELEMENTS = 1 << 23  # per sample, per site and step (16 MB in fp16)
_DIT_BLOCK_LISTS = ("double_blocks", "joint_blocks", "transformer_blocks", "blocks", "layers", "single_blocks")


def diffusion_model_of(model_patcher) -> Optional[nn.Module]:
    if hasattr(model_patcher, "model") and hasattr(model_patcher.model, "diffusion_model"):
        return model_patcher.model.diffusion_model
    if hasattr(model_patcher, "stream") and hasattr(model_patcher.stream, "unet"):
        return model_patcher.stream.unet
    return None


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

def default_sites(root: nn.Module) -> str:
    if hasattr(root, "input_blocks") and hasattr(root, "output_blocks"):
        return DEFAULT_UNET_SITES
    lists = [n for n in _DIT_BLOCK_LISTS if isinstance(getattr(root, n, None), nn.ModuleList)]
    if lists:
        return ", ".join(f"{n}.*" for n in lists)
    return "*"


def expand_sites(root: nn.Module, spec: str, strict: bool = False) -> List[str]:
    """
    'input_blocks.*.0, middle_block.1' -> concrete hookable paths. Segments take shell wildcards
    ('*', '?', '[4-8]');
    'auto' (or empty) picks a default set for the architecture. Containers expand to their last child.
    """
    spec = (spec or "").strip()
    if spec in ("", "auto"):
        spec = default_sites(root)
    out: List[str] = []
    for pattern in (p.strip() for p in spec.split(",")):
        if not pattern:
            continue
        paths = expand_path_pattern(root, pattern)
        if not paths:
            msg = f"probe site {pattern!r} matched nothing"
            if strict:
                raise ValueError(f"{LOG_TAG} {msg}")
            warn(msg)
        for path in paths:
            hooked, mod = resolve_hook_target(root, path, strict=strict)
            if mod is not None and hooked not in out:
                out.append(hooked)
    # Model order (registration order roughly follows the forward pass), so heat-map rows read top to bottom.
    order = {name: i for i, (name, _) in enumerate(root.named_modules())}
    return sorted(out, key=lambda p: order.get(p, len(order)))


def _cond_part(x: torch.Tensor, cond_or_uncond):
    """The conditional samples of a CFG batch (None when the batch holds only unconditional samples)."""
    n = len(cond_or_uncond or [])
    if n <= 1 or x.shape[0] % n != 0:
        return x if (not cond_or_uncond or cond_or_uncond[0] == 0) else None
    chunks = x.chunk(n)
    conds = [ch for ch, cu in zip(chunks, cond_or_uncond) if cu == 0]
    return torch.cat(conds) if conds else None


def _channel_dim(x: torch.Tensor) -> int:
    return x.ndim - 1 if x.ndim == 3 else 1  # tokens are (B, L, C); images/videos are (B, C, ...)


def _image_tokens(x: torch.Tensor, img_slice) -> torch.Tensor:
    """For a joint [txt; img] token tensor (Flux/Chroma/Hunyuan single blocks), drop the text tokens."""
    if x.ndim == 3 and img_slice and len(img_slice) == 2 and x.shape[1] == img_slice[1] and 0 < img_slice[0] < x.shape[1]:
        return x[:, img_slice[0]:]
    return x


def _num(value):
    """JSON-safe number: None for NaN/Inf (json.dumps would otherwise emit NaN, which strict parsers reject)."""
    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class Activations:
    """Frozen CPU snapshot of a probed run: per-site tensors indexed [step, ...]."""

    def __init__(self, sites, steps, stats, channel_means, spatial_means, sigmas, latent_shape):
        self.sites: List[str] = sites
        self.steps: List[int] = steps
        self.stats: Dict[str, torch.Tensor] = stats                  # site -> [steps, len(STAT_NAMES)]
        self.channel_means: Dict[str, torch.Tensor] = channel_means  # site -> [steps, C]
        self.spatial_means: Dict[str, torch.Tensor] = spatial_means  # site -> [steps, C, h, w] or [steps, L, C]
        self.sigmas = sigmas
        self.latent_shape = latent_shape

    def stat(self, site: str, name: str) -> torch.Tensor:
        return self.stats[site][:, STAT_NAMES.index(name)]


class ProbeHandle:
    """Filled by the probe hooks while a sampler runs; resets itself when a new sampling run starts."""

    def __init__(self, sites: List[str], store: str = "channel_means"):
        self.sites = sites
        self.keep_channel = store in ("channel_means", "spatial_means")
        self.keep_spatial = store == "spatial_means"
        self._reset(None)

    def _reset(self, signature):
        self.signature = signature
        self.spatial_skipped = set()
        self.records: Dict[str, Dict[int, dict]] = {}
        self.last_step = -1
        self.sigmas = None
        self.latent_shape = None

    def begin_forward(self, transformer_options, latent_shape):
        transformer_options = transformer_options or {}
        step = current_step(transformer_options)
        schedule = transformer_options.get("sample_sigmas")
        signature = None
        if schedule is not None:
            signature = (len(schedule), round(float(schedule[0]), 5), round(float(schedule[-1]), 5))
        if signature != self.signature or (step is not None and step < self.last_step):
            self._reset(signature)
            if schedule is not None:
                self.sigmas = [float(s) for s in schedule]
        if step is not None:
            self.last_step = max(self.last_step, step)
        self.latent_shape = latent_shape
        return step

    @torch.no_grad()
    def record(self, site: str, step: Optional[int], x: torch.Tensor):
        per = self.records.setdefault(site, {})
        if step is None:
            step = len(per)
        if step in per:  # keep the first evaluation of a step (2nd-order samplers call the model twice)
            return
        xf = x.detach().float()
        finite = torch.isfinite(xf)
        xs = torch.where(finite, xf, torch.zeros_like(xf))
        cdim = _channel_dim(xs)
        other = tuple(d for d in range(xs.ndim) if d != cdim)
        if xs.numel() // xs.shape[cdim] > 1:
            dead_frac = (xs.std(dim=other) < 1e-6).float().mean()
        else:  # one value per channel (e.g. a 2-D (B, C) output): "dead" is undefined
            dead_frac = torch.tensor(float("nan"), device=xs.device)
        rec = {"stats": torch.stack([
            xs.mean(), xs.std(), xs.pow(2).mean().sqrt(), xs.abs().max(),
            (~finite).sum().float(), dead_frac,
        ])}
        if self.keep_channel:
            rec["chan"] = xs.mean(dim=other)
        if self.keep_spatial:
            per_sample = xs[0].numel()
            if per_sample <= SPATIAL_MAX_ELEMENTS:
                rec["spatial"] = xs.mean(dim=0).to("cpu", torch.float16)
            elif site not in self.spatial_skipped:
                # A WAN block output is ~33k tokens x 1.5-5k channels per step: keeping it for every site and step
                # would fill system RAM. Stats and channel means are still recorded.
                self.spatial_skipped.add(site)
                warn("probe: %s has %d values per sample (cap %d); spatial means are not kept for it. "
                     "Probe fewer sites, or use store='channel_means' for video models.",
                     site, per_sample, SPATIAL_MAX_ELEMENTS)
        per[step] = rec

    def has_data(self) -> bool:
        return any(self.records.values())

    def snapshot(self) -> Activations:
        sites = [s for s in self.sites if self.records.get(s)]
        steps = sorted({step for s in sites for step in self.records[s]})
        stats, chans, spatial = {}, {}, {}
        for site in sites:
            per = self.records[site]
            nan_row = torch.full((len(STAT_NAMES),), float("nan"))
            stats[site] = torch.stack([per[s]["stats"].cpu() if s in per else nan_row for s in steps])
            if self.keep_channel and all(s in per for s in steps):
                chans[site] = torch.stack([per[s]["chan"].cpu() for s in steps])
            if self.keep_spatial and all(s in per and "spatial" in per[s] for s in steps):
                spatial[site] = torch.stack([per[s]["spatial"].float() for s in steps])
        return Activations(sites, steps, stats, chans, spatial, self.sigmas, self.latent_shape)


def attach_probe(model_patcher, sites_spec: str = "auto", store: str = "channel_means", strict: bool = False):
    """
    Clone the model and add observe-only hooks at the given sites. Returns (model, ProbeHandle).
    Hooks are registered innermost (after any bending hooks), so they record the bent activations
    wherever the probe sits in the model chain. Outputs are never modified.
    """
    m = model_patcher.clone()
    dm = diffusion_model_of(m)
    if dm is None:
        raise ValueError(f"{LOG_TAG} probe: no diffusion model found")
    sites = expand_sites(dm, sites_spec, strict=strict)
    if not sites:
        raise ValueError(f"{LOG_TAG} probe: no hookable sites in {sites_spec!r}")
    targets = [(p, resolve_hook_target(dm, p)[1]) for p in sites]
    handle = ProbeHandle(sites, store)
    prev_wrapper = m.model_options.get("model_function_wrapper")

    def wrapper(apply_model, params):
        c = params["c"] or {}
        transformer_options = c.get("transformer_options") or {}
        step = handle.begin_forward(transformer_options, tuple(params["input"].shape))
        cond_or_uncond = params.get("cond_or_uncond") or [0]

        def inner_apply(x, t, **kw):
            hooks = []
            for path, mod in targets:
                def _hook(module, args, output, _path=path):
                    out = output[0] if isinstance(output, (tuple, list)) and output else output
                    if isinstance(out, torch.Tensor) and out.ndim >= 2:
                        part = _cond_part(out, cond_or_uncond)
                        if part is not None:
                            # DiT single blocks carry [txt; img] tokens; keep the image tokens only.
                            handle.record(_path, step, _image_tokens(part, transformer_options.get("img_slice")))
                    return None
                hooks.append(mod.register_forward_hook(_hook))
            try:
                return apply_model(x, t, **kw)
            finally:
                for h in hooks:
                    h.remove()

        with_model_sampling(inner_apply, apply_model)
        if prev_wrapper is not None:
            return prev_wrapper(inner_apply, params)
        return inner_apply(params["input"], params["timestep"], **c)

    m.set_model_unet_function_wrapper(wrapper)
    return m, handle


# ---------------------------------------------------------------------------
# Report + heat map
# ---------------------------------------------------------------------------

def _aligned(ref: Activations, site: str, name: str, n: int) -> Optional[torch.Tensor]:
    """Reference stat for `site` resampled to n steps by position (None if the site is missing)."""
    if ref is None or site not in ref.stats:
        return None
    v = ref.stat(site, name)
    if len(v) == n:
        return v
    idx = [round(i * (len(v) - 1) / max(1, n - 1)) for i in range(n)]
    return v[idx]


def build_report(act: Activations, reference: Optional[Activations] = None, stat: str = "std",
                 alert_ratio: float = 10.0) -> dict:
    alerts, per_site = [], []
    for site in act.sites:
        values = act.stat(site, stat)
        std = act.stat(site, "std")
        finite = values[torch.isfinite(values)]
        entry = {"site": site, f"{stat}_first": _num(values[0]), f"{stat}_last": _num(values[-1]),
                 f"{stat}_max": _num(finite.max()) if finite.numel() else None}
        for i, step in enumerate(act.steps):
            nonfinite, dead = act.stat(site, "nonfinite")[i], act.stat(site, "dead_frac")[i]
            if nonfinite > 0:
                alerts.append({"kind": "nonfinite", "site": site, "step": step, "count": int(nonfinite)})
            if math.isfinite(float(dead)) and dead > 0.5:
                alerts.append({"kind": "dead_channels", "site": site, "step": step, "fraction": round(float(dead), 3)})
        ref_std = _aligned(reference, site, "std", len(act.steps))
        if ref_std is not None:
            ratio = std / ref_std.clamp_min(1e-12)
            finite_ratio = ratio[torch.isfinite(ratio)]
            entry["std_ratio_max"] = round(float(finite_ratio.max()), 3) if finite_ratio.numel() else None
            entry["std_ratio_min"] = round(float(finite_ratio.min()), 3) if finite_ratio.numel() else None
            for i, r in enumerate(ratio.tolist()):
                if math.isfinite(r) and (r >= alert_ratio or r <= 1.0 / alert_ratio):
                    alerts.append({"kind": "blowup" if r >= 1 else "collapse", "site": site,
                                   "step": act.steps[i], "std_ratio": round(r, 3)})
        per_site.append(entry)
    return {
        "stat": stat, "sites": len(act.sites), "steps": act.steps, "reference": reference is not None,
        "latent_shape": list(act.latent_shape) if act.latent_shape else None,
        "alerts": alerts, "per_site": per_site,
    }


_SEQ = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]  # viridis anchors
_DIV = [(49, 99, 170), (247, 247, 247), (192, 57, 43)]


def _lerp_colors(anchors, x):
    x = min(1.0, max(0.0, x)) * (len(anchors) - 1)
    i = min(int(x), len(anchors) - 2)
    f = x - i
    a, b = anchors[i], anchors[i + 1]
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


def _font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_heatmap(act: Activations, reference: Optional[Activations] = None, stat: str = "std"):
    """Site x step heat map. Absolute values on a log scale, or log2(value / reference) when a reference is given."""
    from PIL import Image, ImageDraw

    n_sites, n_steps = len(act.sites), len(act.steps)
    cell_w = max(28, min(90, 640 // max(1, n_steps)))
    cell_h = 20
    font, small = _font(13), _font(11)
    label_w = min(320, 16 + 7 * max((len(s) for s in act.sites), default=8))
    top, bottom = 48, 46
    width = label_w + cell_w * n_steps + 16
    height = top + cell_h * n_sites + bottom
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    grid = torch.stack([act.stat(s, stat) for s in act.sites]) if n_sites else torch.zeros(0, n_steps)
    if reference is not None:
        ref = torch.stack([
            _aligned(reference, s, stat, n_steps) if s in reference.stats else torch.full((n_steps,), float("nan"))
            for s in act.sites
        ])
        vals = torch.log2(grid.abs().clamp_min(1e-12) / ref.abs().clamp_min(1e-12))
        lo, hi = -3.0, 3.0
        title = f"{stat}: bent / reference (log2, red = larger)"
    else:
        vals = torch.log10(grid.abs().clamp_min(1e-8))
        finite = vals[torch.isfinite(vals)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.numel() else (0.0, 1.0)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        title = f"{stat} per site and step (log scale)"
    draw.text((8, 6), title, fill=(20, 20, 20), font=font)

    for j, step in enumerate(act.steps):
        draw.text((label_w + j * cell_w + 4, top - 18), str(step), fill=(90, 90, 90), font=small)
    for i, site in enumerate(act.sites):
        y = top + i * cell_h
        draw.text((8, y + 3), site, fill=(30, 30, 30), font=small)
        nonfinite = act.stat(site, "nonfinite")
        for j in range(n_steps):
            x = label_w + j * cell_w
            v = float(vals[i, j])
            if nonfinite[j] > 0:
                color, text = (214, 0, 214), "NaN"
            elif not math.isfinite(v):
                color, text = (220, 220, 220), ""
            elif reference is not None:
                color = _lerp_colors(_DIV, (v - lo) / (hi - lo))
                text = f"×{2 ** v:.2g}"
            else:
                color = _lerp_colors(_SEQ, (v - lo) / (hi - lo))
                text = f"{float(grid[i, j]):.2g}"
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color)
            if cell_w >= 40 and text:
                lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                draw.text((x + 3, y + 3), text, fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=small)

    # legend
    ly = top + n_sites * cell_h + 14
    lw = min(300, width - label_w - 16)
    for k in range(lw):
        c = _lerp_colors(_DIV if reference is not None else _SEQ, k / max(1, lw - 1))
        draw.line([(label_w + k, ly), (label_w + k, ly + 10)], fill=c)
    if reference is not None:
        left, right = "÷8", "×8"
    else:
        left, right = f"{10 ** lo:.2g}", f"{10 ** hi:.2g}"
    draw.text((label_w - 8 - 7 * len(left), ly - 2), left, fill=(60, 60, 60), font=small)
    draw.text((label_w + lw + 6, ly - 2), right, fill=(60, 60, 60), font=small)
    draw.text((8, ly - 2), "step →", fill=(60, 60, 60), font=small)
    return img


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------

def compute_steering(a: Activations, b: Activations, site: str, mode: str = "channel") -> dict:
    """Per-step difference of mean activations at `site`: run A minus run B."""
    source_a = a.channel_means if mode == "channel" else a.spatial_means
    source_b = b.channel_means if mode == "channel" else b.spatial_means
    if site not in source_a or site not in source_b:
        store = "stats+channel_means" if mode == "channel" else "stats+spatial_means"
        available = sorted(set(source_a) & set(source_b))
        raise ValueError(
            f"{LOG_TAG} site {site!r} has no {mode} means in both probes. Probe with store='{store}'. "
            f"Available: {', '.join(available) or 'none'}"
        )
    da, db = source_a[site].float(), source_b[site].float()
    if da.shape[1:] != db.shape[1:]:
        raise ValueError(f"{LOG_TAG} activation shapes differ at {site}: {tuple(da.shape[1:])} vs {tuple(db.shape[1:])}")
    n = min(len(da), len(db))
    pick = lambda v: v if len(v) == n else v[[round(i * (len(v) - 1) / max(1, n - 1)) for i in range(n)]]
    da, db = pick(da), pick(db)
    delta = da - db
    rel = (delta.flatten(1).norm(dim=1) / db.flatten(1).norm(dim=1).clamp_min(1e-12)).mean()
    rms = a.stat(site, "rms")
    return {
        "site": site, "mode": mode, "delta": delta,
        "site_rms": pick(rms).tolist(),
        "rel_norm": float(rel),
    }


def _broadcast_delta(d: torch.Tensor, out: torch.Tensor, mode: str, latent_shape):
    """Shape the per-step delta to add onto `out` (None if it cannot be matched)."""
    cdim = _channel_dim(out)
    if mode == "channel" or d.ndim == 1:
        if d.ndim != 1 or d.shape[0] != out.shape[cdim]:
            return None
        shape = [1] * out.ndim
        shape[cdim] = d.shape[0]
        return d.view(shape)
    if out.ndim == 4 and d.ndim == 3:
        if d.shape[0] != out.shape[1]:
            return None
        if d.shape[-2:] != out.shape[-2:]:
            d = F.interpolate(d[None], size=out.shape[-2:], mode="bilinear", align_corners=False)[0]
        return d[None]
    if out.ndim == 3 and d.ndim == 2 and d.shape == out.shape[1:]:
        return d[None]
    return _broadcast_delta(d.mean(dim=tuple(range(d.ndim - 1)) if out.ndim == 3 else tuple(range(1, d.ndim))),
                            out, "channel", latent_shape)


def apply_steering(model_patcher, vector: dict, strength: float, t_start: float = 1.0, t_end: float = 0.0,
                   apply_to: str = "both", normalize: bool = False):
    """Clone the model and add strength * delta[step] to the output of the steering site."""
    m = model_patcher.clone()
    if strength == 0.0:
        return m
    dm = diffusion_model_of(m)
    _, target = resolve_hook_target(dm, vector["site"], strict=True)
    delta = vector["delta"].float()
    if normalize:
        # Rescale each step so the push has the RMS of the activation itself: strength 1 = one activation's worth.
        rms_site = torch.tensor(vector["site_rms"]).view(-1, *[1] * (delta.ndim - 1))
        rms_delta = delta.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-12).view_as(rms_site)
        delta = delta * rms_site / rms_delta
    mode = vector["mode"]
    t_window = t_window_from(t_start, t_end)
    prev_wrapper = m.model_options.get("model_function_wrapper")
    cache = {}
    warned = set()

    def wrapper(apply_model, params):
        c = params["c"] or {}
        to = c.get("transformer_options")
        step = current_step(to)
        t = normalized_t(model_sampling_of(apply_model), params["timestep"]) if t_window else None

        def call(fn=apply_model):
            if fn is not apply_model:
                with_model_sampling(fn, apply_model)
            if prev_wrapper is not None:
                return prev_wrapper(fn, params)
            return fn(params["input"], params["timestep"], **c)

        if t_window and t is None:  # fail closed rather than steer at every step
            if "no_t" not in warned:
                warned.add("no_t")
                warn("steering: cannot resolve diffusion time; the t window is skipped for this pass")
            return call()
        if not in_window(step, t, None, t_window):
            return call()
        n_run, n_vec = total_steps(to), delta.shape[0]
        if step is None:
            idx = 0
        elif n_run is None or n_run == n_vec:
            idx = min(step, n_vec - 1)
        else:
            idx = min(n_vec - 1, round(step * (n_vec - 1) / max(1, n_run - 1)))
        cond_or_uncond = params.get("cond_or_uncond") or [0]
        latent_shape = tuple(params["input"].shape)

        def _hook(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(out, torch.Tensor):
                return None
            key = (idx, out.device, out.dtype, tuple(out.shape[1:]))
            add = cache.get(key)
            if add is None:
                add = _broadcast_delta(delta[idx], out, mode, latent_shape)
                if add is None:
                    if "shape" not in warned:
                        warned.add("shape")
                        warn("steering vector for %s does not match the activation shape %s; skipped",
                             vector["site"], tuple(out.shape))
                    return None
                add = add.to(out.device, out.dtype)
                cache[key] = add
            add = strength * add
            if apply_to == "cond" and out.shape[0] % len(cond_or_uncond) == 0:
                # Weight by the batch's contents: an uncond-only pass (cond_or_uncond == [1]) gets nothing.
                per = out.shape[0] // len(cond_or_uncond)
                w = torch.tensor([1.0 if cu == 0 else 0.0 for cu in cond_or_uncond for _ in range(per)],
                                 device=out.device, dtype=out.dtype)
                add = add * w.view(-1, *[1] * (out.ndim - 1))
            new = out + add
            if isinstance(output, (tuple, list)):
                return type(output)([new, *output[1:]])
            return new

        def inner_apply(x, ts, **kw):
            handle = target.register_forward_hook(_hook)
            try:
                return apply_model(x, ts, **kw)
            finally:
                handle.remove()

        return call(inner_apply)

    m.set_model_unet_function_wrapper(wrapper)
    return m
