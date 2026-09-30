"""
CPU tests with tiny randomly initialised models built from ComfyUI's own classes (no weights needed).

Run with ComfyUI's Python from anywhere:
    python_embeded/python.exe ComfyUI/custom_nodes/ComfyUI-Model-Bending/tests/test_hooks.py
"""
import copy
import importlib.util
import math
import os
import sys
import types

import torch
import torch.nn as nn

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(PKG_DIR))
sys.path.insert(0, COMFY_DIR)

# Run ComfyUI on the CPU with plain PyTorch attention (must happen before comfy.model_management is imported).
from comfy.cli_args import args as _comfy_args  # noqa: E402
_comfy_args.cpu = True
_comfy_args.disable_xformers = True
_comfy_args.use_pytorch_cross_attention = True


def _stub_server():
    """nodes.py registers HTTP routes at import time; give it a PromptServer that accepts them."""
    class _Routes:
        def get(self, *_a, **_k):
            return lambda fn: fn
        post = get

    class _Instance:
        routes = _Routes()

        def send_sync(self, *_a, **_k):
            pass

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=_Instance())
    sys.modules.setdefault("server", server)


def _load_package():
    _stub_server()
    spec = importlib.util.spec_from_file_location(
        "model_bending", os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["model_bending"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


_load_package()
import comfy.ops  # noqa: E402
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel  # noqa: E402
from comfy.model_sampling import ModelSamplingDiscrete  # noqa: E402
from model_bending import bendutils, nodes, probe, model_bending_nodes as mbn  # noqa: E402
from model_bending.bending_modules import (  # noqa: E402
    GatedBendingModule, MultiplyScalarModule, RotateModule, AddScalarModule)


# ---------------------------------------------------------------------------
# Tiny models and a minimal stand-in for ComfyUI's ModelPatcher + sampler loop
# ---------------------------------------------------------------------------

def _init_random(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return module.eval()


def tiny_unet():
    unet = UNetModel(
        image_size=None, in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1,
        channel_mult=(1, 2), num_head_channels=8, use_spatial_transformer=True,
        transformer_depth=[1, 1], transformer_depth_output=[1, 1, 1, 1], transformer_depth_middle=1,
        context_dim=16, use_linear_in_transformer=True, operations=comfy.ops.disable_weight_init,
    )
    return _init_random(unet)


def tiny_flux():
    from comfy.ldm.flux.model import Flux
    flux = Flux(operations=comfy.ops.disable_weight_init, in_channels=4, out_channels=4, vec_in_dim=8,
                context_in_dim=16, hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=2, depth_single_blocks=2,
                axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=False,
                txt_ids_dims=[])
    return _init_random(flux, seed=1)


class FakeBase(nn.Module):
    def __init__(self, dm, kind="unet"):
        super().__init__()
        self.diffusion_model = dm
        self.model_sampling = ModelSamplingDiscrete(model_config=None)
        self.kind = kind

    def apply_model(self, x, sigma, **c):
        t = self.model_sampling.timestep(sigma).float()
        to = c.get("transformer_options", {})
        if self.kind == "flux":
            b = x.shape[0]
            return self.diffusion_model(x, t / 1000.0, c["c_crossattn"], y=torch.zeros(b, 8), transformer_options=to)
        return self.diffusion_model(x, timesteps=t, context=c["c_crossattn"], transformer_options=to)


class FakePatcher:
    def __init__(self, base, model_options=None):
        self.model = base
        self.model_options = model_options or {"transformer_options": {}}

    def clone(self):
        return FakePatcher(self.model, copy.deepcopy({k: v for k, v in self.model_options.items()
                                                      if k != "model_function_wrapper"}) |
                           ({"model_function_wrapper": self.model_options["model_function_wrapper"]}
                            if "model_function_wrapper" in self.model_options else {}))

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        to = self.model_options.setdefault("transformer_options", {})
        to.setdefault("patches_replace", {}).setdefault(name, {})[(block_name, number)] = patch

    def get_model_object(self, name):
        return getattr(self.model, name)


SIGMAS = torch.tensor([14.6, 6.0, 2.5, 0.9, 0.3, 0.0])


def run(patcher, x, ctx, cfg_batch=True, sigmas=SIGMAS):
    """Call the model once per step the way ComfyUI's sampler does (cond + uncond batched)."""
    base = patcher.model
    wrapper = patcher.model_options.get("model_function_wrapper")
    cond_or_uncond = [1, 0] if cfg_batch else [0]
    xb = x.repeat(len(cond_or_uncond), *[1] * (x.ndim - 1))
    cb = ctx.repeat(len(cond_or_uncond), 1, 1)
    outs = []
    with torch.no_grad():
        for sigma in sigmas[:-1]:
            to = copy.copy(patcher.model_options.get("transformer_options", {}))
            sb = sigma.repeat(xb.shape[0])
            to.update({"sample_sigmas": sigmas, "sigmas": sb, "cond_or_uncond": cond_or_uncond})
            c = {"c_crossattn": cb, "transformer_options": to}
            if wrapper is not None:
                out = wrapper(base.apply_model, {"input": xb, "timestep": sb, "c": c, "cond_or_uncond": cond_or_uncond})
            else:
                out = base.apply_model(xb, sb, **c)
            outs.append(out)
    return outs


def inputs(seed=0, shape=(1, 4, 16, 16)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g), torch.randn(1, 5, 16, generator=g)


def same(a, b):
    return torch.equal(a, b)


def patched(base_patcher, wrapper):
    m = base_patcher.clone()
    m.set_model_unet_function_wrapper(wrapper)
    return m


UNET = FakePatcher(FakeBase(tiny_unet()))
X, CTX = inputs()
BASELINE = run(UNET, X, CTX)


# ---------------------------------------------------------------------------
# Hook engine
# ---------------------------------------------------------------------------

def test_hook_on_container_never_fires_without_expansion():
    """Documents the A1/A2 bug: TimestepEmbedSequential.forward is bypassed by the UNet."""
    fired = []
    h = UNET.model.diffusion_model.middle_block.register_forward_hook(lambda *a: fired.append(1))
    try:
        run(UNET, X, CTX)
    finally:
        h.remove()
    assert not fired


def test_container_paths_expand_and_bend():
    dm = UNET.model.diffusion_model
    assert bendutils.resolve_hook_target(dm, "middle_block")[0] == "middle_block.2"
    assert bendutils.resolve_hook_target(dm, "output_blocks.1")[0] == f"output_blocks.1.{len(dm.output_blocks[1]) - 1}"
    wrapper = bendutils.make_bending_unet_wrapper(dm, ["middle_block"], MultiplyScalarModule(0.0))
    outs = run(patched(UNET, wrapper), X, CTX)
    assert not same(outs[0], BASELINE[0])


def test_sd_block_node_bends():
    node = mbn.SDBlockModelBending()
    (m,) = node.patch(UNET, MultiplyScalarModule(0.0), "middle_block", 0)
    assert not same(run(m, X, CTX)[0], BASELINE[0])


def test_modulelist_skipped_and_strict_raises():
    dm = UNET.model.diffusion_model
    assert bendutils.resolve_hook_target(dm, "input_blocks")[1] is None
    try:
        bendutils.resolve_hook_target(dm, "input_blocks", strict=True)
    except ValueError as e:
        assert "ModuleList" in str(e)
    else:
        raise AssertionError("strict should raise")
    try:
        bendutils.resolve_hook_target(dm, "midle_block.1", strict=True)
    except ValueError as e:
        assert "middle_block" in str(e)  # close-match suggestion
    else:
        raise AssertionError("strict should raise")


def test_step_gating_with_cfg_batch():
    """Old step tracking compared a batch of sigmas against the schedule and failed silently with CFG."""
    dm = UNET.model.diffusion_model
    wrapper = bendutils.make_bending_unet_wrapper(dm, ["middle_block.1"], MultiplyScalarModule(0.0), steps_to_bend=[1])
    outs = run(patched(UNET, wrapper), X, CTX)
    assert same(outs[0], BASELINE[0]) and not same(outs[1], BASELINE[1]) and same(outs[2], BASELINE[2])


def test_shared_module_two_appliers_do_not_interfere():
    shared = MultiplyScalarModule(0.0)
    node = mbn.CustomModelBending()
    (a,) = node.patch(UNET, shared, "middle_block.1", steps_to_bend_str="0")
    (b,) = node.patch(UNET, shared, "middle_block.1", steps_to_bend_str="2")
    outs_a = run(a, X, CTX)
    assert not same(outs_a[0], BASELINE[0]) and same(outs_a[2], BASELINE[2])
    assert not hasattr(shared, "steps_to_bend") or shared.steps_to_bend is None


def test_t_window_and_gated_module_match():
    dm = UNET.model.diffusion_model
    t_of = [bendutils.normalized_t(UNET.model.model_sampling, s) for s in SIGMAS[:-1]]
    assert all(t_of[i] > t_of[i + 1] for i in range(len(t_of) - 1))
    w1 = bendutils.make_bending_unet_wrapper(dm, ["middle_block.1"], MultiplyScalarModule(0.0), t_window=(1.0, 0.5))
    w2 = bendutils.make_bending_unet_wrapper(dm, ["middle_block.1"],
                                             GatedBendingModule(MultiplyScalarModule(0.0), 1.0, 0.5))
    o1, o2 = run(patched(UNET, w1), X, CTX), run(patched(UNET, w2), X, CTX)
    for i, t in enumerate(t_of):
        assert same(o1[i], o2[i])
        assert same(o1[i], BASELINE[i]) == (t < 0.5)


def test_ramped_gate_weights():
    # edges at 1.0 / 0.0 do not fade: there is nothing before pure noise or after the clean image
    g = GatedBendingModule(MultiplyScalarModule(0.0), 1.0, 0.0, ramp="linear", ramp_width=0.5)
    assert g.weight(1.0) == 1.0 and g.weight(0.0) == 1.0 and g.weight(0.5) == 1.0
    g = GatedBendingModule(MultiplyScalarModule(0.0), 1.0, 0.4, ramp="linear", ramp_width=0.2)
    assert g.weight(1.0) == 1.0 and abs(g.weight(0.5) - 0.5) < 1e-6 and g.weight(0.4) == 0.0
    assert GatedBendingModule(None, 0.8, 0.3).weight(0.9) == 0.0


def test_window_survives_a_probe_in_either_order():
    """Wrappers that pass their own function down the chain must not hide model_sampling from windowed bends."""
    node = mbn.CustomModelBending()
    (windowed,) = node.patch(UNET, MultiplyScalarModule(0.0), "middle_block.1", t_start=1.0, t_end=0.5)
    alone = run(windowed, X, CTX)
    assert not same(alone[0], BASELINE[0]) and same(alone[-1], BASELINE[-1])  # the window really gates
    probe_after, _ = probe.attach_probe(windowed, "middle_block.1")
    probe_first, _ = probe.attach_probe(UNET, "middle_block.1")
    (bend_after_probe,) = node.patch(probe_first, MultiplyScalarModule(0.0), "middle_block.1", t_start=1.0, t_end=0.5)
    for m in (probe_after, bend_after_probe):
        assert all(same(a, b) for a, b in zip(run(m, X, CTX), alone))
    # steering and the feature-map capture pass functions too
    ctx_b = inputs(seed=5)[1]
    ma, ha = probe.attach_probe(UNET, "middle_block.1")
    mb, hb = probe.attach_probe(UNET, "middle_block.1")
    run(ma, X, CTX)
    run(mb, X, ctx_b)
    vec = probe.compute_steering(ha.snapshot(), hb.snapshot(), "middle_block.1")
    steered_windowed = probe.apply_steering(windowed, vec, 0.0)  # strength 0: only the wrapper chain changes
    assert all(same(a, b) for a, b in zip(run(steered_windowed, X, CTX), alone))


def test_windowed_bend_fails_closed_without_model_sampling():
    dm = UNET.model.diffusion_model
    base = UNET.model
    plain = lambda x, t, **c: base.apply_model(x, t, **c)  # noqa: E731  (no __self__, no model_sampling)
    wrapper = bendutils.make_bending_unet_wrapper(dm, ["middle_block.1"], MultiplyScalarModule(0.0), t_window=(1.0, 0.5))
    patcher = patched(UNET, wrapper)
    outs = []
    with torch.no_grad():
        for sigma in SIGMAS[:-1]:
            sb = sigma.repeat(2)
            to = {"sample_sigmas": SIGMAS, "sigmas": sb, "cond_or_uncond": [1, 0]}
            c = {"c_crossattn": CTX.repeat(2, 1, 1), "transformer_options": to}
            outs.append(patcher.model_options["model_function_wrapper"](plain, {"input": X.repeat(2, 1, 1, 1), "timestep": sb, "c": c, "cond_or_uncond": [1, 0]}))
    assert all(same(o, b) for o, b in zip(outs, BASELINE))  # skipped, not applied everywhere
    strict_wrapper = bendutils.make_bending_unet_wrapper(dm, ["middle_block.1"], MultiplyScalarModule(0.0), t_window=(1.0, 0.5), strict=True)
    try:
        strict_wrapper(plain, {"input": X, "timestep": SIGMAS[:1], "c": {"c_crossattn": CTX, "transformer_options": {}}, "cond_or_uncond": [0]})
    except ValueError as e:
        assert "diffusion time" in str(e)
    else:
        raise AssertionError("strict should raise")


def test_all_bends_invalid_still_reports():
    import json
    js = json.dumps({"bends": [{"module_type": "multiply"}]})
    m, report, _ = nodes.ApplyBendsFromJSON().patch(UNET, js)
    assert "bend #0 has no 'path'" in "\n".join(json.loads(report)["warnings"])
    try:
        nodes.ApplyBendsFromJSON().patch(UNET, js, strict=True)
    except ValueError as e:
        assert "no 'path'" in str(e)
    else:
        raise AssertionError("strict should raise")


def test_fourier_is_per_sample_and_channel():
    from model_bending.bending_modules import FourierAmplifyModule
    mod = FourierAmplifyModule(cutoff_freq=2, amp_factor=3.0)
    x = torch.randn(2, 3, 8, 8)
    batched = mod(x)
    for b in range(2):
        for ch in range(3):
            assert torch.allclose(batched[b, ch], mod(x[b:b + 1, ch:ch + 1])[0, 0], atol=1e-5)


def test_probe_2d_outputs_and_json_safe_report():
    import json
    handle = probe.ProbeHandle(["time_embed.2"], "stats")
    handle.record("time_embed.2", 0, torch.randn(1, 16))  # one value per channel: dead_frac is undefined
    act = handle.snapshot()
    assert math.isnan(float(act.stat("time_embed.2", "dead_frac")[0]))
    report = probe.build_report(act)
    assert report["alerts"] == []
    json.dumps(report, allow_nan=False)  # would raise on NaN
    # a site missing at some steps yields NaN rows: still JSON-safe
    h2 = probe.ProbeHandle(["a", "b"], "stats")
    h2.record("a", 0, torch.randn(1, 4, 4, 4))
    h2.record("a", 1, torch.randn(1, 4, 4, 4))
    h2.record("b", 1, torch.randn(1, 4, 4, 4))
    json.dumps(probe.build_report(h2.snapshot(), reference=h2.snapshot()), allow_nan=False)


def test_steering_cond_only_skips_uncond_only_pass():
    ctx_b = inputs(seed=5)[1]
    ma, ha = probe.attach_probe(UNET, "middle_block.1")
    mb, hb = probe.attach_probe(UNET, "middle_block.1")
    run(ma, X, CTX)
    run(mb, X, ctx_b)
    vec = probe.compute_steering(ha.snapshot(), hb.snapshot(), "middle_block.1")
    m = probe.apply_steering(UNET, vec, 3.0, apply_to="cond")
    base = UNET.model
    to = {"sample_sigmas": SIGMAS, "sigmas": SIGMAS[:1], "cond_or_uncond": [1]}
    c = {"c_crossattn": CTX, "transformer_options": to}
    with torch.no_grad():
        uncond_only = m.model_options["model_function_wrapper"](base.apply_model, {"input": X, "timestep": SIGMAS[:1], "c": c, "cond_or_uncond": [1]})
        plain = base.apply_model(X, SIGMAS[:1], **c)
    assert same(uncond_only, plain)


def test_nan_guard_and_blend():
    dm = UNET.model.diffusion_model
    spec = bendutils.BendSpec("middle_block.1", MultiplyScalarModule(float("nan")))
    outs = run(patched(UNET, bendutils.make_bends_wrapper(dm, [spec])), X, CTX)
    assert torch.isfinite(outs[0]).all()
    half = bendutils.BendSpec("middle_block.1", MultiplyScalarModule(3.0), blend=0.5)
    full = bendutils.BendSpec("middle_block.1", MultiplyScalarModule(2.0))
    oh = run(patched(UNET, bendutils.make_bends_wrapper(dm, [half])), X, CTX)
    of = run(patched(UNET, bendutils.make_bends_wrapper(dm, [full])), X, CTX)
    assert torch.allclose(oh[0], of[0], atol=1e-5)  # x + 0.5 * (3x - x) == 2x


def test_5d_input_folds_time():
    x = torch.randn(2, 3, 4, 8, 8)
    y = RotateModule(90)(x)
    frame = RotateModule(90)(x[:, :, 1])
    assert y.shape == x.shape and torch.allclose(y[:, :, 1], frame, atol=1e-5)


# ---------------------------------------------------------------------------
# JSON path
# ---------------------------------------------------------------------------

def test_json_per_bend_steps_fourier_and_open_range():
    bends = [
        {"path": "middle_block.1", "module_type": "multiply", "module_args": {"scalar": 0}, "steps": "1"},
        {"path": "output_blocks.1", "module_type": "fourier", "module_args": {"cutoff_freq": 2, "amp_factor": 3}},
    ]
    report = {}
    (m,) = nodes.apply_bends_to_model(UNET, bends[:1], report=report)
    outs = run(m, X, CTX)
    assert same(outs[0], BASELINE[0]) and not same(outs[1], BASELINE[1])
    (m,) = nodes.apply_bends_to_model(UNET, bends[1:], steps_min=3, report=report)  # open-ended "3-"
    outs = run(m, X, CTX)
    assert same(outs[2], BASELINE[2]) and not same(outs[3], BASELINE[3])
    assert any(e["path"] == "output_blocks.1" for e in report["expanded"])


def _json(bends, **top):
    import json
    return json.dumps({"bends": bends, **top})


def test_json_same_layer_bends_keep_legacy_order():
    """Two bends on one layer apply last-to-first, as the old one-wrapper-per-bend chain did."""
    dm = UNET.model.diffusion_model
    mul = {"path": "middle_block.1", "module_type": "multiply", "module_args": {"scalar": 2}}
    add = {"path": "middle_block.1", "module_type": "add_scalar", "module_args": {"scalar": 1}}
    m, _, _ = nodes.ApplyBendsFromJSON().patch(UNET, _json([mul, add]))
    add_then_mul = bendutils.make_bends_wrapper(dm, [bendutils.BendSpec("middle_block.1", AddScalarModule(1)),
                                                     bendutils.BendSpec("middle_block.1", MultiplyScalarModule(2))])
    assert same(run(m, X, CTX)[0], run(patched(UNET, add_then_mul), X, CTX)[0])
    m2, _, _ = nodes.ApplyBendsFromJSON().patch(UNET, _json([add, mul]))
    assert not same(run(m, X, CTX)[0], run(m2, X, CTX)[0])


def test_json_wildcards_and_resolved_json_round_trip():
    import json
    bends = [{"path": "output_blocks.*.1", "module_type": "multiply", "module_args": {"scalar": 1.5}, "steps": "0-2"},
             {"path": "middle_block", "module_type": "fourier", "t": [1.0, 0.4]}]
    m, report, resolved = nodes.ApplyBendsFromJSON().patch(UNET, _json(bends))
    report, resolved = json.loads(report), json.loads(resolved)
    dm = UNET.model.diffusion_model
    expected = [p for p in bendutils.expand_path_pattern(dm, "output_blocks.*.1")]
    assert expected and report["patterns"][0]["matches"] == expected
    assert [b["path"] for b in resolved["bends"]] == expected + ["middle_block.2"]
    fourier = resolved["bends"][-1]
    assert fourier["module_args"] == {"cutoff_freq": 5.0, "amp_factor": 2.0} and fourier["t"] == [1.0, 0.4]
    assert resolved["version"] == nodes.BENDS_JSON_VERSION and "warnings" not in report
    m2, _, again = nodes.ApplyBendsFromJSON().patch(UNET, json.dumps(resolved))
    assert json.loads(again) == resolved
    assert all(same(a, b) for a, b in zip(run(m, X, CTX), run(m2, X, CTX)))


def test_json_validation_warns_and_strict_raises():
    import json
    bad = {"bends": [{"path": "middle_block.1", "module_type": "multiply", "module_args": {"scaler": 0}},
                     {"module_type": "multiply"},
                     {"path": "middle_block.1", "module_type": "multiply", "colour": "red"},
                     {"path": "nothing_here.*", "module_type": "multiply"}],
           "version": 2, "extra": 1}
    m, report, _ = nodes.ApplyBendsFromJSON().patch(UNET, json.dumps(bad))
    warnings = "\n".join(json.loads(report)["warnings"])
    for needle in ("did you mean 'scalar'", "bend #1 has no 'path'", "unknown key 'colour'",
                   "matched no layers", "newer than this plugin", "unknown top-level key 'extra'"):
        assert needle in warnings, needle
    # the typo falls back to the default (x1), so nothing is bent: that is exactly what the warning is for
    assert same(run(m, X, CTX)[0], BASELINE[0])
    try:
        nodes.ApplyBendsFromJSON().patch(UNET, json.dumps(bad), strict=True)
    except ValueError as e:
        assert "did you mean 'scalar'" in str(e)
    else:
        raise AssertionError("strict should raise")
    # v1 JSON from the web UI produces no warnings
    web = {"bends": [{"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90}}],
           "steps_min": None, "steps_max": None, "max_denoising_steps": 200, "selected_part": "diffusion_model"}
    _, report, _ = nodes.ApplyBendsFromJSON().patch(UNET, json.dumps(web))
    assert "warnings" not in json.loads(report)


def test_json_clamp_hard_and_safe_ranges():
    import json
    bends = [{"path": "middle_block.1", "module_type": "multiply", "module_args": {"scalar": 500}},
             {"path": "output_blocks.3.1", "module_type": "multiply", "module_args": {"scalar": 5}}]
    _, report, resolved = nodes.ApplyBendsFromJSON().patch(UNET, _json(bends), clamp="hard")
    clamped = json.loads(report)["clamped"]
    assert clamped == [{"path": "middle_block.1", "op": "multiply", "arg": "scalar", "requested": 500.0,
                        "applied": 100.0}]
    safe = json.dumps({"multiply": {"scalar": [0, 3]}, "paths": {"output_blocks.*": {"multiply": {"scalar": [0.5, 1.5]}}}})
    _, report, resolved = nodes.ApplyBendsFromJSON().patch(UNET, _json(bends), clamp="safe", safe_ranges=safe)
    applied = {c["path"]: c["applied"] for c in json.loads(report)["clamped"]}
    assert applied == {"middle_block.1": 3.0, "output_blocks.3.1": 1.5}
    assert [b["module_args"]["scalar"] for b in json.loads(resolved)["bends"]] == [3.0, 1.5]


def test_json_subset_and_guard():
    import json
    bends = [{"path": "middle_block.1", "module_type": "subset", "module_args": {"percentage": 0.5, "dim": "channel"},
              "inner": {"module_type": "multiply", "module_args": {"scalar": 0}},
              "guard": {"max_std_ratio": 1.0, "nan": "clamp"}}]
    m, report, resolved = nodes.ApplyBendsFromJSON().patch(UNET, _json(bends))
    assert not same(run(m, X, CTX)[0], BASELINE[0])
    entry = json.loads(resolved)["bends"][0]
    assert entry["inner"]["module_args"] == {"scalar": 0.0} and entry["guard"]["max_std_ratio"] == 1.0
    try:
        nodes.ApplyBendsFromJSON().patch(UNET, _json([{"path": "middle_block.1", "module_type": "subset"}]))
    except ValueError as e:
        assert "inner" in str(e)
    else:
        raise AssertionError("subset without inner should raise")


def test_guards():
    x = torch.randn(2, 8, 4, 4)
    y = x * 10
    capped = bendutils.blend_and_guard(x, y, 1.0, guard={"max_std_ratio": 2.0})
    assert abs(float(capped.std() / x.std()) - 2.0) < 1e-3
    kept = bendutils.blend_and_guard(x, y.flip(-1), 1.0, guard={"preserve_norm": True})
    assert torch.allclose(kept.pow(2).sum((0, 2, 3)), x.pow(2).sum((0, 2, 3)), rtol=1e-4)
    inf = bendutils.blend_and_guard(x, torch.full_like(x, float("inf")), 1.0, guard={"nan": "clamp"})
    assert torch.isfinite(inf).all() and float(inf.max()) == torch.finfo(x.dtype).max
    assert torch.isinf(bendutils.blend_and_guard(x, torch.full_like(x, float("inf")), 1.0, guard={"nan": "none"})).all()


def test_json_node_placeholders_and_report():
    js = '{"bends": [{"path": "middle_block.1", "module_type": "multiply", "module_args": {"scalar": {{a}}}}]}'
    m, report, _ = nodes.ApplyBendsFromJSON().patch(UNET, js, a=0.0)
    assert '"a": 0.0' in report and '"path": "middle_block.1"' in report
    assert not same(run(m, X, CTX)[0], BASELINE[0])
    m1, _, _ = nodes.ApplyBendsFromJSON().patch(UNET, js, a=1.0)
    assert same(run(m1, X, CTX)[0], BASELINE[0])


# ---------------------------------------------------------------------------
# Probe + steering
# ---------------------------------------------------------------------------

def test_probe_is_observe_only_and_records_every_site_and_step():
    m, handle = probe.attach_probe(UNET, "auto")
    outs = run(m, X, CTX)
    assert all(same(a, b) for a, b in zip(outs, BASELINE))
    act = handle.snapshot()
    dm = UNET.model.diffusion_model
    assert len(act.sites) == len(dm.input_blocks) + 1 + len(dm.output_blocks)
    assert act.steps == list(range(len(SIGMAS) - 1))
    assert act.channel_means["middle_block.1"].shape == (len(act.steps), 64)
    img = probe.render_heatmap(act)
    assert img.size[0] > 100 and img.size[1] > 100
    # a second run resets the handle instead of mixing runs
    run(m, X, CTX)
    assert handle.snapshot().steps == act.steps


def test_probe_measures_after_bends_in_either_order():
    node = mbn.CustomModelBending()
    ref_model, ref_handle = probe.attach_probe(UNET, "middle_block.1, output_blocks.3.0")
    run(ref_model, X, CTX)
    ref = ref_handle.snapshot()
    (bent,) = node.patch(UNET, MultiplyScalarModule(3.0), "middle_block.1")
    probed_after, h1 = probe.attach_probe(bent, "middle_block.1, output_blocks.3.0")
    run(probed_after, X, CTX)
    probed_first, h2 = probe.attach_probe(UNET, "middle_block.1, output_blocks.3.0")
    (bent_after_probe,) = node.patch(probed_first, MultiplyScalarModule(3.0), "middle_block.1")
    run(bent_after_probe, X, CTX)
    for h in (h1, h2):
        ratio = h.snapshot().stat("middle_block.1", "std") / ref.stat("middle_block.1", "std")
        assert torch.allclose(ratio, torch.full_like(ratio, 3.0), rtol=1e-3)
    report = probe.build_report(h1.snapshot(), reference=ref, alert_ratio=2.0)
    assert any(a["kind"] == "blowup" and a["site"] == "middle_block.1" for a in report["alerts"])
    probe.render_heatmap(h1.snapshot(), reference=ref)


def test_steering_vector_zero_is_identity_and_nonzero_steers():
    ctx_b = inputs(seed=5)[1]
    ma, ha = probe.attach_probe(UNET, "middle_block.1", store="spatial_means")
    mb, hb = probe.attach_probe(UNET, "middle_block.1", store="spatial_means")
    run(ma, X, CTX)
    run(mb, X, ctx_b)
    a, b = ha.snapshot(), hb.snapshot()
    vec = probe.compute_steering(a, b, "middle_block.1", "channel")
    assert vec["delta"].shape == (len(SIGMAS) - 1, 64)
    zero = probe.apply_steering(UNET, vec, 0.0)
    assert all(same(o, r) for o, r in zip(run(zero, X, ctx_b), BASELINE_B()))
    # steering B's prompt by the full A - B difference at every step moves B's activations onto A's
    full = probe.apply_steering(UNET, vec, 1.0)
    check, hc = probe.attach_probe(full, "middle_block.1")
    run(check, X, ctx_b)
    got = hc.snapshot().channel_means["middle_block.1"]
    assert torch.allclose(got, a.channel_means["middle_block.1"], atol=1e-4)
    spatial = probe.compute_steering(a, b, "middle_block.1", "spatial")
    assert spatial["delta"].shape[1:] == (64, 8, 8)
    run(probe.apply_steering(UNET, spatial, 2.0, apply_to="cond", normalize=True), X, ctx_b)


_BASELINE_B = []


def BASELINE_B():
    if not _BASELINE_B:
        _BASELINE_B.extend(run(UNET, X, inputs(seed=5)[1]))
    return _BASELINE_B


# ---------------------------------------------------------------------------
# DiT tokens
# ---------------------------------------------------------------------------

def test_token_grid_roundtrip_keeps_reference_tokens():
    assert bendutils.infer_token_grid(4096, (1, 16, 128, 128)) == (1, 64, 64)
    assert bendutils.infer_token_grid(21 * 30 * 52, (1, 16, 21, 60, 104)) == (21, 30, 52)
    x = torch.randn(2, 64 + 10, 8)  # 8x8 image tokens + 10 reference tokens
    grid = bendutils.infer_token_grid(x.shape[1], (2, 4, 16, 16))
    g, rest = bendutils.tokens_to_grid(x, grid)
    assert g.shape == (2, 8, 8, 8) and rest.shape == (2, 10, 8)
    assert torch.equal(bendutils.grid_to_tokens(g, grid, 2, rest), x)
    # position (row 1, col 2) of the grid is token 1*8+2
    assert torch.equal(g[0, :, 1, 2], x[0, 10])


def test_dit_block_bending_on_tiny_flux():
    flux = FakePatcher(FakeBase(tiny_flux(), kind="flux"))
    x, ctx = inputs(shape=(1, 4, 8, 8))
    base = run(flux, x, ctx)
    node = mbn.DiTBlockBending()
    m, report = node.patch(flux, RotateModule(90), "double:0-1, single:0", "img", True)
    assert '"double:1"' in report and '"single:0"' in report
    bent = run(m, x, ctx)
    assert all(torch.isfinite(o).all() for o in bent) and not same(bent[0], base[0])

    # stream isolation: bending only txt at the last single block leaves the image output untouched
    m_txt, _ = node.patch(flux, MultiplyScalarModule(0.0), "single:1", "txt", False)
    assert all(same(a, b) for a, b in zip(run(m_txt, x, ctx), base))

    # spatial rotate of the image stream equals rotating the 4x4 token grid
    captured = {}
    m_rot, _ = node.patch(flux, RotateModule(90), "single:1", "img", True)
    last = m_rot.model_options["transformer_options"]["patches_replace"]["dit"][("single_block", 1)]

    def spy(args, extra):
        # Flux blocks modify their input in place, so capture the one real call instead of calling twice.
        def original_block(a):
            out = extra["original_block"](a)
            captured["orig"] = out["img"].clone()
            return out
        out = last(args, {**extra, "original_block": original_block})
        captured["bent"] = out["img"]
        return out
    m_rot.model_options["transformer_options"]["patches_replace"]["dit"][("single_block", 1)] = spy
    run(m_rot, x, ctx, sigmas=SIGMAS[:2])
    txt_len = captured["orig"].shape[1] - 16
    img_o, img_b = captured["orig"][:, txt_len:], captured["bent"][:, txt_len:]
    grid_o, _ = bendutils.tokens_to_grid(img_o, (1, 4, 4))
    grid_b, _ = bendutils.tokens_to_grid(img_b, (1, 4, 4))
    assert torch.allclose(grid_b, RotateModule(90)(grid_o), atol=1e-5)
    assert torch.equal(captured["bent"][:, :txt_len], captured["orig"][:, :txt_len])


def test_catalogue_marks_containers_and_modulelists():
    import json
    (text,) = mbn.BendableLayerCatalogue().catalogue(UNET, "", 2)
    entries = {e["path"]: e for e in json.loads(text)["layers"]}
    assert entries["input_blocks"]["hookable"] is False
    assert "middle_block.2" in entries["middle_block"]["note"]


if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
