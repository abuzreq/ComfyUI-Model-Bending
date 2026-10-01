"""
CPU tests for video-model (WAN) bending with tiny randomly initialised models built from ComfyUI's own WanModel
class (no weights needed). The attention bends are checked against independent reference implementations and
against exact equivalences (a permutation of the attention map along the video axis equals the same permutation
of the attention output), so token order, grid layout and head handling are verified without real WAN weights.

Run with ComfyUI's Python from anywhere:
    python_embeded/python.exe ComfyUI/custom_nodes/ComfyUI-Model-Bending/tests/test_video.py
"""
import copy
import json
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hooks as th  # noqa: E402  (loads the package and sets ComfyUI to CPU / PyTorch attention)

import comfy.ops  # noqa: E402
from comfy.ldm.wan.model import WanModel  # noqa: E402
from comfy.model_sampling import ModelSamplingDiscreteFlow  # noqa: E402
from model_bending import attention_bending as ab, bendutils, model_bending_nodes as mbn, probe  # noqa: E402
from model_bending.bending_modules import (  # noqa: E402
    ApplyToRandomSubsetModule, FlipModule, FrameRampModule, GatedBendingModule, GaussianBlurModule,
    MultiplyScalarModule, RotateModule, SharpenModule, TemporalBlurModule, TemporalShiftModule, TranslateModule,
    FrameReverseModule)


# ---------------------------------------------------------------------------
# Tiny WAN and a sampler loop that passes clip_fea for I2V
# ---------------------------------------------------------------------------

def tiny_wan(model_type="t2v", layers=4, seed=3):
    wan = WanModel(model_type=model_type, patch_size=(1, 2, 2), in_dim=4, dim=32, ffn_dim=64, freq_dim=16,
                   text_dim=16, out_dim=4, num_heads=2, num_layers=layers, operations=comfy.ops.disable_weight_init)
    th._init_random(wan, seed=seed)
    with torch.no_grad():  # sharpen the attention so the maps are far from uniform
        for block in wan.blocks:
            for attn in (block.self_attn, block.cross_attn):
                attn.norm_q.weight.fill_(1.0)
                attn.norm_k.weight.fill_(1.0)
                attn.q.weight.mul_(40.0)
                if hasattr(attn, "norm_k_img"):
                    attn.norm_k_img.weight.fill_(1.0)
    return wan


class WanBase(nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.diffusion_model = dm
        self.model_sampling = ModelSamplingDiscreteFlow(model_config=None)

    def apply_model(self, x, sigma, **c):
        t = self.model_sampling.timestep(sigma).float()
        return self.diffusion_model(x, t, c["c_crossattn"], clip_fea=c.get("clip_fea"),
                                    transformer_options=c.get("transformer_options", {}))


SIGMAS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])


def run(patcher, x, ctx, clip_fea=None, cfg_batch=True, sigmas=SIGMAS):
    base = patcher.model
    wrapper = patcher.model_options.get("model_function_wrapper")
    cond_or_uncond = [1, 0] if cfg_batch else [0]
    n = len(cond_or_uncond)
    xb = x.repeat(n, *[1] * (x.ndim - 1))
    cb = ctx.repeat(n, 1, 1)
    outs = []
    with torch.no_grad():
        for sigma in sigmas[:-1]:
            to = copy.copy(patcher.model_options.get("transformer_options", {}))
            sb = sigma.repeat(xb.shape[0])
            to.update({"sample_sigmas": sigmas, "sigmas": sb, "cond_or_uncond": cond_or_uncond})
            c = {"c_crossattn": cb, "transformer_options": to}
            if clip_fea is not None:
                c["clip_fea"] = clip_fea.repeat(n, 1, 1)
            if wrapper is not None:
                out = wrapper(base.apply_model, {"input": xb, "timestep": sb, "c": c, "cond_or_uncond": cond_or_uncond})
            else:
                out = base.apply_model(xb, sb, **c)
            outs.append(out)
    return outs


def wan_inputs(shape=(1, 4, 3, 6, 10), seed=0, ctx_len=8, real=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    ctx = torch.randn(1, ctx_len, 16, generator=g)
    ctx[:, real:] = 0.0  # padding rows, zeroed like ComfyUI's UMT5 output
    return x, ctx


WAN = th.FakePatcher(WanBase(tiny_wan()))
X, CTX = wan_inputs()       # grid (3, 3, 5): 3 latent frames of 3 x 5 tokens
BASE = run(WAN, X, CTX)


def close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


def all_close(xs, ys, atol=1e-5):
    return all(close(a, b, atol) for a, b in zip(xs, ys))


def attn_bend(patcher=WAN, module=None, attention="cross_text", blocks="*", tokens="all", renormalize="none",
              apply_to="both", **kw):
    m, report = ab.AttentionMapBending().patch(patcher, module, attention, blocks, tokens, renormalize, apply_to, **kw)
    return m, json.loads(report)


def hook_bend(path, module, patcher=WAN):
    dm = patcher.model.diffusion_model
    return th.patched(patcher, bendutils.make_bends_wrapper(dm, [bendutils.BendSpec(path, module)]))


# ---------------------------------------------------------------------------
# Token grid
# ---------------------------------------------------------------------------

def test_token_layout_for_wan():
    dm = WAN.model.diffusion_model
    assert bendutils.token_layout_for(dm, 45, (2, 4, 3, 6, 10)) == bendutils.TokenLayout((3, 3, 5))
    for t in (1, 4, 13, 21):
        assert bendutils.token_layout_for(dm, t * 30 * 52, (1, 16, t, 60, 104)).grid == (t, 30, 52)
    assert bendutils.token_layout_for(dm, 4 * 4 * 5, (1, 4, 4, 7, 9)).grid == (4, 4, 5)  # odd sizes pad up
    # generic hooks refuse token counts that do not match (e.g. the text embedding)
    assert bendutils.token_layout_for(dm, 512, (1, 16, 3, 6, 10), allow_extra=False) is None
    # reference tokens (ref_conv) are prepended
    ref = types.SimpleNamespace(patch_size=(1, 2, 2), ref_conv=object())
    assert bendutils.token_layout_for(ref, 15 + 45, (1, 4, 3, 6, 10)) == bendutils.TokenLayout((3, 3, 5), prefix=15)
    # the guessing fallback no longer confuses p=1,tp=4 with WAN's 1x2x2 patches when T % 4 == 0
    assert bendutils.infer_token_grid(4 * 30 * 52, (1, 16, 4, 60, 104)) == (4, 30, 52)
    assert bendutils.infer_token_grid(13 * 30 * 52, (1, 16, 13, 60, 104)) == (13, 30, 52)
    # tokens <-> video round trip keeps order: token (f, y, x) = f*h*w + y*w + x
    x = torch.randn(2, 15 + 45, 8)
    layout = bendutils.TokenLayout((3, 3, 5), prefix=15)
    v, pre, post = bendutils.tokens_to_video(x, layout)
    assert v.shape == (2, 8, 3, 3, 5) and post is None
    assert torch.equal(v[1, :, 2, 1, 4], x[1, 15 + 2 * 15 + 1 * 5 + 4])
    assert torch.equal(bendutils.video_to_tokens(v, pre, post), x)


def test_process_path_strips_wan_subclasses():
    assert bendutils.process_path("WAN22.diffusion_model.blocks.0.cross_attn") == "blocks.0.cross_attn"
    assert bendutils.process_path("WAN21_Vace.diffusion_model.blocks.3") == "blocks.3"


# ---------------------------------------------------------------------------
# Cross-attention bending: reference implementation on synthetic tensors
# ---------------------------------------------------------------------------

def ref_attention(q, k, v, heads, **_):
    b, lq, hd = q.shape
    d = hd // heads
    qh, kh, vh = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    p = torch.softmax(qh @ kh.transpose(-1, -2) * d ** -0.5, dim=-1)
    return (p @ vh).transpose(1, 2).reshape(b, lq, hd)


def reference_bent(q, k, v, heads, grid, fn, rows, keys=None, heads_sel=None, frames=None, renorm="none", weight=1.0):
    """Loop-by-loop reference: each selected key column is reshaped to (F, H, W) and bent per frame."""
    b, lq, hd = q.shape
    d = hd // heads
    f, h, w = grid
    qh, kh, vh = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    p = torch.softmax(qh @ kh.transpose(-1, -2) * d ** -0.5, dim=-1)
    out = p.clone()
    lk = k.shape[1]
    for r in rows:
        for hh in (heads_sel if heads_sel is not None else range(heads)):
            m = p[r, hh].clone()                       # (lq, lk)
            cols = list(keys) if keys is not None else list(range(lk))
            new = m.clone()
            for c in cols:
                img = m[:, c].reshape(f, 1, h, w)
                bent = fn(img)                        # per frame, (f, 1, h, w)
                bent = img + weight * (bent - img)
                for fr in range(f):
                    if frames is None or fr in frames:
                        new[:, c].view(f, h, w)[fr] = bent[fr, 0]
            if renorm == "keys":
                total = new.sum(-1, keepdim=True)
                dead = total.abs() < 1e-6
                new = torch.where(dead, m, new / torch.where(dead, torch.ones_like(total), total))
            elif renorm == "per_token_mass":
                for c in cols:
                    om, nm = m[:, c].sum(), new[:, c].sum()
                    if nm.abs() > 1e-12:
                        new[:, c] = new[:, c] * om / nm
            out[r, hh] = new
    return (out @ vh).transpose(1, 2).reshape(b, lq, hd)


def make_entry(module, grid=(3, 3, 5), cou=(1, 0), **kw):
    dm = types.SimpleNamespace(patch_size=(1, 2, 2))
    e = ab.AttentionEntry(dm, "cross_text", {0}, kw.pop("tokens", ab.TokenSpec("all")), module=module, **kw)
    e.latent_shape = (len(cou), 4, grid[0], grid[1] * 2, grid[2] * 2)
    e.cond_or_uncond = list(cou)
    e.active, e.weight = True, kw.get("strength", 1.0)
    return e


def call_override(entries, q, k, v, heads=2):
    for e in entries:
        e.site, e.calls = ("cross", 0, False), 0
    return ab.attention_override(ref_attention, q, k, v, heads=heads,
                                 transformer_options={ab.ENTRIES_KEY: entries})


def synthetic(b=2, lq=45, lk=7, heads=2, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, lq, heads * d, generator=g) * 3
    k = torch.randn(b, lk, heads * d, generator=g)
    v = torch.randn(b, lk, heads * d, generator=g)
    return q, k, v


def test_cross_bend_matches_reference_loops():
    q, k, v = synthetic()
    grid = (3, 3, 5)
    rot = RotateModule(33, padding="zeros")
    cases = [
        dict(module=FlipModule("horizontal")),
        dict(module=rot, renorm="keys"),
        dict(module=rot, renorm="per_token_mass"),
        dict(module=SharpenModule(1.5, 0.8), renorm="keys", tokens=ab.TokenSpec("indices", [1, 4])),
        dict(module=TranslateModule(0.4, -0.3), heads=[1], frames=[0, 2]),
        dict(module=GaussianBlurModule(0.7), apply_to="cond", strength=0.5),
    ]
    for case in cases:
        module = case["module"]
        e = make_entry(module, renorm=case.get("renorm", "none"), tokens=case.get("tokens", ab.TokenSpec("all")),
                       heads=case.get("heads"), frames=case.get("frames"), apply_to=case.get("apply_to", "both"),
                       strength=case.get("strength", 1.0))
        got = call_override([e], q, k, v)
        rows = e.rows(2)
        want = reference_bent(q, k, v, 2, grid, module, rows, keys=e.tokens.indices, heads_sel=case.get("heads"),
                              frames=case.get("frames"), renorm=case.get("renorm", "none"),
                              weight=case.get("strength", 1.0))
        assert close(got, want, atol=1e-5), (type(module).__name__, (got - want).abs().max())
        if case.get("apply_to") == "cond":  # row 0 is uncond: untouched, bit for bit
            assert torch.equal(got[0], ref_attention(q, k, v, 2)[0])


def test_cross_bend_chunking_and_prompt_tokens():
    q, k, v = synthetic(lk=9)
    e = make_entry(RotateModule(50), renorm="keys", tokens=ab.TokenSpec("prompt"))
    e.real_counts = [6, 4]
    full = call_override([e], q, k, v)
    old = ab.CHUNK_ELEMENTS
    try:
        ab.CHUNK_ELEMENTS = 1  # one head at a time
        chunked = call_override([e], q, k, v)
    finally:
        ab.CHUNK_ELEMENTS = old
    assert close(full, chunked, atol=1e-6)
    # 'prompt' bends only the non-padding keys of each row
    for r, n in enumerate([6, 4]):
        want = reference_bent(q[r:r + 1], k[r:r + 1], v[r:r + 1], 2, (3, 3, 5), RotateModule(50), [0],
                              keys=list(range(n)), renorm="keys")
        assert close(full[r:r + 1], want, atol=1e-5)


def test_keys_renorm_rows_sum_to_one_without_nan():
    q, k, v = synthetic()
    e = make_entry(RotateModule(45, padding="zeros"), renorm="keys")
    e.capture = ab.AttentionCapture()
    out = call_override([e], q, k, v)
    assert torch.isfinite(out).all()
    # a 'scale up' that pushes mass out of the frame still leaves valid rows
    e2 = make_entry(mbn.ScaleModule(3.0, padding="zeros") if hasattr(mbn, "ScaleModule") else RotateModule(90), renorm="keys")
    assert torch.isfinite(call_override([e2], q, k, v)).all()


# ---------------------------------------------------------------------------
# On the tiny WAN model
# ---------------------------------------------------------------------------

def test_identity_bend_matches_sdpa():
    for renorm in ("none", "keys", "per_token_mass"):
        m, report = attn_bend(module=MultiplyScalarModule(1.0), renormalize=renorm)
        assert report["experimental"] is True and report["blocks"] == [0, 1, 2, 3]
        assert all_close(run(m, X, CTX), BASE), renorm


def test_cross_flip_equals_flipping_the_attention_output():
    """A permutation of the map along the video axis is the same permutation of the attention output."""
    for module in (FlipModule("horizontal"), FlipModule("vertical"), RotateModule(37, padding="border"),
                   TranslateModule(0.3, -0.2, padding="border")):
        m, _ = attn_bend(module=module, blocks="1")
        ref = hook_bend("blocks.1.cross_attn", module)
        got, want = run(m, X, CTX), run(ref, X, CTX)
        assert all_close(got, want, atol=1e-4), type(module).__name__
        assert not all_close(got, BASE)


def test_apply_to_steps_and_frames():
    m, _ = attn_bend(module=FlipModule("horizontal"), apply_to="cond")
    outs = run(m, X, CTX)
    assert all(torch.equal(o[0], b[0]) and not torch.equal(o[1], b[1]) for o, b in zip(outs, BASE))
    m, _ = attn_bend(module=FlipModule("horizontal"), steps="1")
    outs = run(m, X, CTX)
    assert torch.equal(outs[0], BASE[0]) and not torch.equal(outs[1], BASE[1]) and torch.equal(outs[2], BASE[2])
    m, _ = attn_bend(module=FlipModule("horizontal"), t_start=0.6, t_end=0.0)  # sigma 1.0, 0.75 skipped
    outs = run(m, X, CTX)
    assert torch.equal(outs[0], BASE[0]) and torch.equal(outs[1], BASE[1]) and not torch.equal(outs[2], BASE[2])
    # a gated module is weighted by diffusion time too
    gated = GatedBendingModule(FlipModule("horizontal"), t_start=0.6, t_end=0.0)
    m, _ = attn_bend(module=gated)
    outs = run(m, X, CTX)
    assert torch.equal(outs[0], BASE[0]) and not torch.equal(outs[3], BASE[3])
    # frames: only latent frame 0 bent
    m, _ = attn_bend(module=FlipModule("horizontal"), frames="0", blocks="3")
    ref_all, _ = attn_bend(module=FlipModule("horizontal"), blocks="3")
    assert not all_close(run(m, X, CTX), run(ref_all, X, CTX))


def test_no_leak_and_chaining():
    m, _ = attn_bend(module=FlipModule("horizontal"))
    run(m, X, CTX)
    assert all(torch.equal(a, b) for a, b in zip(run(WAN, X, CTX), BASE))
    # a pre-existing override (e.g. a SageAttention patch) still serves the calls that are not bent
    calls = {"n": 0}

    def other(func, *args, **kwargs):
        calls["n"] += 1
        return func(*args, **kwargs)
    pre = WAN.clone()
    pre.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = other
    m, _ = attn_bend(pre, module=FlipModule("horizontal"), blocks="1")
    outs = run(m, X, CTX)
    assert calls["n"] > 0
    ref, _ = attn_bend(module=FlipModule("horizontal"), blocks="1")
    assert all_close(outs, run(ref, X, CTX))
    # two attention bends compose on the same call
    m1, _ = attn_bend(module=FlipModule("horizontal"), blocks="1")
    m2, _ = attn_bend(m1, module=FlipModule("horizontal"), blocks="1")
    assert all_close(run(m2, X, CTX), BASE, atol=1e-4)  # flipping twice is the identity


def test_tokens_select_keys():
    m_all, _ = attn_bend(module=FlipModule("vertical"), tokens="all")
    m_idx, rep = attn_bend(module=FlipModule("vertical"), tokens="0-7")
    assert rep["tokens"] == list(range(8))
    assert all_close(run(m_all, X, CTX), run(m_idx, X, CTX))
    m_one, _ = attn_bend(module=FlipModule("vertical"), tokens="2")
    m_prompt, _ = attn_bend(module=FlipModule("vertical"), tokens="prompt")
    one, prompt, full = run(m_one, X, CTX), run(m_prompt, X, CTX), run(m_all, X, CTX)
    assert not all_close(one, full) and not all_close(prompt, full) and not all_close(one, BASE)


def test_i2v_image_and_text_calls():
    wan = th.FakePatcher(WanBase(tiny_wan("i2v", layers=2, seed=4)))
    x, ctx = wan_inputs(seed=2)
    clip = torch.randn(1, 257, 1280)
    base = run(wan, x, ctx, clip_fea=clip)
    seen = []
    orig = ab._cross_attention

    def spy(base_fn, q, k, v, rest, kwargs, live):
        seen.extend((kind, k.shape[1]) for _, kind in live)
        return orig(base_fn, q, k, v, rest, kwargs, live)
    ab._cross_attention = spy
    try:
        m_img, _ = attn_bend(wan, module=FlipModule("horizontal"), attention="cross_image", blocks="0")
        out_img = run(m_img, x, ctx, clip_fea=clip, sigmas=SIGMAS[:2])
    finally:
        ab._cross_attention = orig
    assert ("cross_image", 257) in seen and ("cross_text", 8) in seen
    m_txt, _ = attn_bend(wan, module=FlipModule("horizontal"), attention="cross_text", blocks="0")
    out_txt = run(m_txt, x, ctx, clip_fea=clip, sigmas=SIGMAS[:2])
    assert not close(out_img[0], base[0]) and not close(out_txt[0], base[0]) and not close(out_img[0], out_txt[0])
    # bending the image call equals flipping its share of the output: the text call is untouched
    m_id, _ = attn_bend(wan, module=MultiplyScalarModule(1.0), attention="cross_image", blocks="0")
    assert all_close(run(m_id, x, ctx, clip_fea=clip), base)


def test_self_attention_modes():
    # self_query with a permutation equals permuting the self-attention output
    m, _ = attn_bend(module=FlipModule("horizontal"), attention="self_query", blocks="2")
    ref = hook_bend("blocks.2.self_attn", FlipModule("horizontal"))
    assert all_close(run(m, X, CTX), run(ref, X, CTX), atol=1e-4)
    # with renormalize='keys' a zero-padded rotation is divided by its coverage (corners keep the original)
    m, _ = attn_bend(module=RotateModule(30), attention="self_query", blocks="2", renormalize="keys")
    assert all(torch.isfinite(o).all() for o in run(m, X, CTX))
    # self_key: identity leaves the output unchanged, a flip moves where content is read from
    m, _ = attn_bend(module=MultiplyScalarModule(1.0), attention="self_key")
    assert all_close(run(m, X, CTX), BASE)
    m, _ = attn_bend(module=FlipModule("vertical"), attention="self_key", blocks="1")
    assert not all_close(run(m, X, CTX), BASE)
    # self_key with flip equals flipping the self-attention values (v projection output) on the grid
    ref = hook_bend("blocks.1.self_attn.v", FlipModule("vertical"))
    assert all_close(run(m, X, CTX), run(ref, X, CTX), atol=1e-4)


def test_attention_capture_and_read():
    cap_model, cap, report = ab.AttentionMapCapture().attach(WAN, "cross_text", "0-1", "0-7")
    outs = run(cap_model, X, CTX)
    assert all(torch.equal(a, b) for a, b in zip(outs, BASE))  # capture never changes the output
    assert sorted({s for s, _ in cap.maps}) == [0, 1, 2, 3] and sorted({b for _, b in cap.maps}) == [0, 1]
    maps = cap.maps[(0, 0)].float()
    assert maps.shape == (8, 3, 3, 5)
    assert torch.allclose(maps.sum(0), torch.ones(3, 3, 5), atol=1e-2)  # all keys of a softmax row sum to 1
    frames, rep = ab.ReadAttentionMaps().read(cap, {"samples": X}, "sum", "mean", "all", "per_video")
    # 3 latent frames -> 1 + 2*4 video frames; 4 steps side by side; 16 px per token (patch 2 x VAE 8)
    assert frames.shape == (9, 48, 80 * 4, 3) and json.loads(rep)["experimental"] is True
    frames, _ = ab.ReadAttentionMaps().read(cap, {"samples": X}, "3", "1", "2", "per_frame", "gray", False)
    assert frames.shape == (3, 48, 80, 3) and frames.min() >= 0 and frames.max() <= 1
    # capture after a bend sees the bent map
    bent, _ = attn_bend(module=FlipModule("horizontal"), blocks="0")
    cap_model2, cap2, _ = ab.AttentionMapCapture().attach(bent, "cross_text", "0", "0-7")
    run(cap_model2, X, CTX, sigmas=SIGMAS[:2])
    a, b = cap.maps[(0, 0)].float(), cap2.maps[(0, 0)].float()
    assert torch.allclose(b, torch.flip(a, dims=(-1,)), atol=2e-3)


class FakeClip:
    """Real ComfyUI tokenizer (T5, the same SentencePiece family as WAN's UMT5) behind clip.tokenize."""

    def __init__(self):
        from comfy import sd1_clip
        from comfy.text_encoders.sd3_clip import T5XXLTokenizer
        self.tokenizer = sd1_clip.SD1Tokenizer(clip_name="t5xxl", tokenizer=T5XXLTokenizer)

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)


def test_word_tokens_from_prompt():
    clip = FakeClip()
    prompt = "a white horse gallops along the beach at sunset"
    table = ab.prompt_token_table(clip, prompt)
    texts = [t for _, _, _, t in table]
    report = {}
    spec = ab.resolve_tokens("horse, the beach", clip, prompt, report)
    picked = [texts[i].strip().lower() for i in spec.indices]
    assert "horse" in "".join(picked) and "beach" in "".join(picked), (picked, texts)
    assert report["prompt_tokens"] and "words_not_found" not in report
    try:
        ab.resolve_tokens("unicorn", clip, prompt, {})
        raise AssertionError("expected a ValueError for a missing word")
    except ValueError:
        pass
    try:
        ab.resolve_tokens("horse", None, None, {})
        raise AssertionError("expected a ValueError without a text encoder")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Generic hooks, DiT blocks, modules and nodes on video tensors
# ---------------------------------------------------------------------------

def test_generic_hooks_lay_tokens_on_the_grid():
    # rotating a block output on the grid differs from the historical "token matrix as an image"
    m = hook_bend("blocks.1", RotateModule(90))
    outs = run(m, X, CTX)
    assert all(torch.isfinite(o).all() for o in outs) and not all_close(outs, BASE)
    # the text embedding (512-ish tokens, not a grid) is still bent as a plain tensor without crashing
    run(hook_bend("text_embedding", MultiplyScalarModule(0.5)), X, CTX, sigmas=SIGMAS[:2])


def test_dit_block_bending_on_wan():
    node = mbn.DiTBlockBending()
    m, report = node.patch(WAN, FlipModule("horizontal"), "double:1", "img", True)
    assert '"double:1"' in report
    via_hook = hook_bend("blocks.1", FlipModule("horizontal"))
    assert all_close(run(m, X, CTX), run(via_hook, X, CTX))
    m_txt, _ = node.patch(WAN, MultiplyScalarModule(0.0), "double:1", "txt", False)
    assert all(torch.equal(a, b) for a, b in zip(run(m_txt, X, CTX), BASE))  # WAN has no text stream: no-op


def test_temporal_modules():
    x = torch.randn(2, 3, 4, 5, 6)
    ramp = FrameRampModule(FlipModule("horizontal"), 0.0, 1.0)(x)
    assert torch.equal(ramp[:, :, 0], x[:, :, 0])
    assert torch.allclose(ramp[:, :, -1], torch.flip(x[:, :, -1], (-1,)), atol=1e-6)
    shift = TemporalShiftModule(1, "border")(x)
    assert torch.equal(shift[:, :, 1:], x[:, :, :-1]) and torch.equal(shift[:, :, 0], x[:, :, 0])
    assert torch.equal(TemporalShiftModule(-1, "wrap")(x), torch.roll(x, -1, 2))
    assert torch.equal(TemporalBlurModule(0.0)(x), x)
    blurred = TemporalBlurModule(1.0)(x)
    assert blurred.shape == x.shape and not torch.equal(blurred, x)
    assert torch.equal(FrameReverseModule()(x), torch.flip(x, (2,)))
    # ordinary modules still act frame by frame
    assert torch.equal(FlipModule("vertical")(x), torch.flip(x, (-2,)))
    # temporal bends on the WAN block output (tokens laid out as video)
    for module in (TemporalShiftModule(1), TemporalBlurModule(1.0), FrameRampModule(RotateModule(45), 0, 1, "triangle")):
        outs = run(hook_bend("blocks.2", module), X, CTX, sigmas=SIGMAS[:2])
        assert torch.isfinite(outs[0]).all() and not close(outs[0], BASE[0])


def test_subset_and_squeeze_on_video_shapes():
    x5 = torch.randn(1, 6, 2, 4, 4)
    out = ApplyToRandomSubsetModule(MultiplyScalarModule(0.0), 0.5, seed=1, dim="channel")(x5)
    assert out.shape == x5.shape and (out == 0).any() and (out != 0).any()
    x3 = torch.randn(2, 10, 6)
    assert ApplyToRandomSubsetModule(MultiplyScalarModule(2.0), 0.5, seed=1, dim="channel")(x3).shape == x3.shape
    one = torch.randn(1, 4, 1, 6, 6)  # B*T == 1: kornia ops used to squeeze the batch away
    from model_bending.bending_modules import ErosionModule, SobelModule
    for module in (ErosionModule(3), SobelModule(True)):
        assert module(one).shape == one.shape


def test_feature_map_node_shows_every_frame():
    tokens = torch.randn(1, 45, 8)
    maps = mbn.IntermediateOutputNode._to_rgb_maps(tokens, (2, 4, 3, 6, 10), dm=WAN.model.diffusion_model)
    assert maps.shape == (3, 48, 80, 3)
    video = torch.randn(1, 4, 3, 6, 10)
    assert mbn.IntermediateOutputNode._to_rgb_maps(video, (1, 4, 3, 6, 10), downscale=16).shape == (3, 96, 160, 3)


def test_probe_skips_huge_spatial_means():
    old = probe.SPATIAL_MAX_ELEMENTS
    probe.SPATIAL_MAX_ELEMENTS = 100
    try:
        m, handle = probe.attach_probe(WAN, "blocks.1", store="spatial_means")
        run(m, X, CTX, sigmas=SIGMAS[:3])
        act = handle.snapshot()
        assert "blocks.1" in act.channel_means and "blocks.1" not in act.spatial_means
    finally:
        probe.SPATIAL_MAX_ELEMENTS = old


def test_bends_json_new_ops_and_attention_bends():
    from model_bending import nodes
    # attention_bends alone matches the node
    js = json.dumps({"attention_bends": [{"module_type": "flip", "module_args": {"direction": "horizontal"},
                                          "attention": "cross_text", "blocks": "1", "renormalize": "none"}]})
    m, report, resolved = nodes.ApplyBendsFromJSON().patch(WAN, js)
    ref, _ = attn_bend(module=FlipModule("horizontal"), blocks="1")
    assert all_close(run(m, X, CTX), run(ref, X, CTX))
    rep = json.loads(report)
    assert rep["attention_bends"][0]["blocks"] == [1] and rep["attention_bends"][0]["experimental"] is True
    assert json.loads(resolved)["attention_bends"][0]["module_type"] == "flip"
    # layer bends with the new ops (frame_ramp wraps an inner op) and attention bends together
    js = json.dumps({"bends": [{"path": "blocks.2", "module_type": "frame_ramp",
                                "module_args": {"w_start": 0, "w_end": 1, "curve": "smooth"},
                                "inner": {"module_type": "translate", "module_args": {"dx": 0.25}}},
                               {"path": "blocks.0.ffn", "module_type": "temporal_shift", "module_args": {"frames": 1}}],
                     "attention_bends": [{"module_type": "sharpen", "module_args": {"amount": 2}, "steps": "0",
                                          "attention": "self_query", "blocks": "3", "t": [1.0, 0.0]}]})
    m, report, _ = nodes.ApplyBendsFromJSON().patch(WAN, js, strict=True)
    outs = run(m, X, CTX)
    assert all(torch.isfinite(o).all() for o in outs) and not all_close(outs, BASE)
    # bad attention items are reported, not applied
    js = json.dumps({"attention_bends": [{"module_type": "flip", "attention": "sideways"},
                                         {"module_type": "flip", "tokens": "horse"}]})
    m, report, _ = nodes.ApplyBendsFromJSON().patch(WAN, js)
    warnings = json.loads(report)["warnings"]
    assert any("sideways" in w for w in warnings) and any("horse" in w for w in warnings)
    assert all(torch.equal(a, b) for a, b in zip(run(m, X, CTX), BASE))


def test_real_comfy_sampler_path():
    """ComfyUI's own WAN21 BaseModel, ModelPatcher (clone, option merging) and sampler with CFG, on a tiny WAN."""
    import comfy.model_patcher
    import comfy.sample
    import comfy.supported_models
    cfg = comfy.supported_models.WAN21_T2V({"image_model": "wan2.1", "model_type": "t2v", "patch_size": (1, 2, 2),
                                            "in_dim": 16, "dim": 64, "ffn_dim": 128, "freq_dim": 32, "text_dim": 16,
                                            "out_dim": 16, "num_heads": 2, "num_layers": 3})
    cfg.set_inference_dtype(torch.float32, None)
    model = cfg.get_model({}, device="cpu")
    th._init_random(model.diffusion_model, seed=7)
    with torch.no_grad():
        for b in model.diffusion_model.blocks:
            for a in (b.self_attn, b.cross_attn):
                a.norm_q.weight.fill_(1)
                a.norm_k.weight.fill_(1)
                a.q.weight.mul_(40)
    cpu = torch.device("cpu")
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
    g = torch.Generator().manual_seed(0)
    latent = torch.zeros(1, 16, 3, 12, 20)
    ctx = torch.randn(1, 8, 16, generator=g)
    ctx[:, 5:] = 0
    pos, neg = [[ctx, {}]], [[torch.randn(1, 8, 16, generator=g), {}]]

    def sample(mp):
        noise = comfy.sample.prepare_noise(latent, 0)
        return comfy.sample.sample(mp, noise, 3, 5.0, "euler", "simple", pos, neg, latent, disable_pbar=True, seed=0)
    base = sample(patcher)
    bent, _ = ab.AttentionMapBending().patch(patcher, FlipModule("horizontal"), "cross_text", "1", "all", "none", "both")
    out = sample(bent)
    assert not torch.allclose(out, base, atol=1e-4)
    ident, _ = ab.AttentionMapBending().patch(patcher, MultiplyScalarModule(1.0), "cross_text", "*", "all", "none", "both")
    assert torch.allclose(sample(ident), base, atol=1e-4)
    assert torch.equal(sample(patcher), base)  # nothing leaks into the unpatched model
    cap_model, cap, _ = ab.AttentionMapCapture().attach(bent, "cross_text", "0-1", "0-4")
    assert torch.allclose(sample(cap_model), out, atol=1e-5)
    assert sorted({s for s, _ in cap.maps}) == [0, 1, 2] and cap.grid_scale == 16


def test_node_registration_marks_video_nodes_experimental():
    from model_bending import nodes
    for key in ("Attention Map Bending", "Attention Map Capture", "Read Attention Maps", "Frame Ramp (Bending)",
                "Temporal Shift Module (Bending)", "Temporal Blur Module (Bending)", "Frame Reverse Module (Bending)"):
        cls = nodes.NODE_CLASS_MAPPINGS[key]
        assert getattr(cls, "EXPERIMENTAL", False) and "experimental" in cls.CATEGORY
        assert "Experimental" in nodes.NODE_DISPLAY_NAME_MAPPINGS[key]
        cls.INPUT_TYPES()
    for key in ("Translate Module (Bending)", "Flip Module (Bending)", "Gaussian Blur Module (Bending)",
                "Sharpen Module (Bending)"):
        nodes.NODE_CLASS_MAPPINGS[key].INPUT_TYPES()


if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and f.__module__ == __name__]
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
