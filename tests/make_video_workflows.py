"""
Generates the example video-bending workflows in workflows/video/ from compact graph specs, and validates them.

Node inputs, widget order and defaults come from a running ComfyUI's /object_info, so the UI-format JSON matches
the installed node definitions exactly (hand-writing litegraph links and widget lists is error-prone).

    python_embeded/python.exe ComfyUI/custom_nodes/ComfyUI-Model-Bending/tests/make_video_workflows.py [--url http://127.0.0.1:8188]
    ... make_video_workflows.py --check     # validate the saved files only (types, links, widget counts, /prompt)

--check also posts each workflow (API format) to /prompt for server-side validation. With the WAN models missing,
the only accepted errors are unknown model filenames; if a prompt validates (models installed), it is removed
from the queue immediately so nothing runs.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PKG_DIR, "workflows", "video")
FRONTEND_ONLY = {"Note", "MarkdownNote"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Model files from Comfy-Org's repackaged releases (checked on Hugging Face).
WAN21_T2V = "wan2.1_t2v_1.3B_fp16.safetensors"
WAN21_I2V = "wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
WAN22_HIGH = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
WAN22_LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
UMT5 = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"

PROMPT = "A white horse gallops along a beach at sunset, waves crashing around its legs, cinematic, golden light"
NEGATIVE = "blurry, low quality, static, distorted, deformed, watermark, text, subtitles"

DOWNLOADS = (
    "Models (Comfy-Org repackaged, put them in ComfyUI/models/...):\n"
    "- diffusion_models: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models\n"
    "- text_encoders: umt5_xxl_fp8_e4m3fn_scaled.safetensors (same repo, split_files/text_encoders)\n"
    "- vae: wan_2.1_vae.safetensors (same repo, split_files/vae)\n")
EXPERIMENTAL = ("EXPERIMENTAL: the video bending nodes were validated on tiny random-weight WAN models only, "
                "not yet on real WAN weights. Please report what you see.")


def is_widget(spec):
    kind, opts = spec[0], (spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})
    if opts.get("forceInput"):
        return False
    return isinstance(kind, list) or kind in WIDGET_TYPES


def input_names(info, section):
    order = (info.get("input_order") or {}).get(section)
    return order if order is not None else list((info["input"].get(section) or {}).keys())


class Node:
    def __init__(self, wf, nid, ntype, info, title=None):
        self.wf, self.id, self.type, self.info, self.title = wf, nid, ntype, info, title
        self.values = {}
        self.links = {}  # input name -> (node, slot)

    def __getitem__(self, slot):
        return (self, slot)


class Workflow:
    def __init__(self, object_info):
        self.oi = object_info
        self.nodes = []
        self.notes = []

    def add(self, ntype, title=None, **kwargs):
        info = self.oi.get(ntype)
        if info is None:
            raise KeyError(f"node type {ntype!r} is not installed on the server")
        node = Node(self, len(self.nodes) + len(self.notes) + 1, ntype, info, title)
        specs = {**(info["input"].get("required") or {}), **(info["input"].get("optional") or {})}
        for name, value in kwargs.items():
            if name not in specs:
                raise KeyError(f"{ntype} has no input {name!r}")
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Node):
                node.links[name] = value
            else:
                node.values[name] = value
        self.nodes.append(node)
        return node

    def note(self, text, column=0, row=0, width=420, height=None):
        self.notes.append((len(self.nodes) + len(self.notes) + 1, text, column, row, width, height))

    # -- serialisation -------------------------------------------------------
    def _depths(self):
        depth = {}

        def d(n):
            if n.id not in depth:
                depth[n.id] = 1 + max((d(src) for src, _ in n.links.values()), default=-1)
            return depth[n.id]
        for n in self.nodes:
            d(n)
        return depth

    def to_ui(self):
        depth = self._depths()
        col_y = {}
        links, nodes_json, next_link = [], [], 1
        outputs_links = {}
        # inputs first so link ids are known
        per_node_inputs = {}
        for n in self.nodes:
            ins = []
            req, opt = input_names(n.info, "required"), input_names(n.info, "optional")
            specs = {**(n.info["input"].get("required") or {}), **(n.info["input"].get("optional") or {})}
            for name in req + opt:
                spec = specs[name]
                widget = is_widget(spec)
                if widget and name not in n.links:
                    continue
                typ = spec[0] if not isinstance(spec[0], list) else "COMBO"
                entry = {"localized_name": name, "name": name, "type": typ, "link": None}
                if widget:
                    entry["widget"] = {"name": name}
                if name in n.links:
                    src, slot = n.links[name]
                    out_type = src.info["output"][slot]
                    entry["type"] = out_type if not widget else typ
                    entry["link"] = next_link
                    links.append([next_link, src.id, slot, n.id, len(ins), out_type])
                    outputs_links.setdefault((src.id, slot), []).append(next_link)
                    next_link += 1
                ins.append(entry)
            per_node_inputs[n.id] = ins
        for n in self.nodes:
            req, opt = input_names(n.info, "required"), input_names(n.info, "optional")
            specs = {**(n.info["input"].get("required") or {}), **(n.info["input"].get("optional") or {})}
            values = []
            for name in req + opt:
                spec = specs[name]
                if not is_widget(spec):
                    continue
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if name in n.values:
                    v = n.values[name]
                elif "default" in opts:
                    v = opts["default"]
                elif isinstance(spec[0], list):
                    v = spec[0][0] if spec[0] else ""
                elif spec[0] == "COMBO":
                    v = (opts.get("options") or [""])[0]
                else:
                    v = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(spec[0], None)
                values.append(v)
                if opts.get("control_after_generate"):
                    values.append("fixed")
                if opts.get("image_upload"):
                    values.append("image")
            outs = []
            for slot, (otype, oname) in enumerate(zip(n.info["output"], n.info.get("output_name") or n.info["output"])):
                outs.append({"localized_name": oname, "name": oname, "type": otype,
                             "links": outputs_links.get((n.id, slot), [])})
            dcol = depth[n.id]
            y = col_y.get(dcol, 260)
            multiline = sum(1 for name in req + opt if is_widget(specs[name]) and len(specs[name]) > 1
                            and isinstance(specs[name][1], dict) and specs[name][1].get("multiline"))
            height = 40 + 26 * len(values) + 22 * max(len(per_node_inputs[n.id]), len(outs)) + 90 * multiline
            col_y[dcol] = y + height + 50
            entry = {"id": n.id, "type": n.type, "pos": [60 + dcol * 400, y], "size": [340, height], "flags": {},
                     "order": 0, "mode": 0, "inputs": per_node_inputs[n.id], "outputs": outs,
                     "properties": {"Node name for S&R": n.type}, "widgets_values": values}
            if n.title:
                entry["title"] = n.title
            nodes_json.append(entry)
        # execution order: by depth
        for i, e in enumerate(sorted(nodes_json, key=lambda e: (depth[e["id"]], e["id"]))):
            e["order"] = i
        for nid, text, column, row, width, height in self.notes:
            lines = sum(1 + len(line) // max(1, width // 8) for line in text.split("\n"))
            h = height or (40 + 18 * lines)
            nodes_json.append({"id": nid, "type": "Note", "pos": [60 + column * 400, -20 - h + row], "size": [width, h],
                               "flags": {}, "order": len(nodes_json), "mode": 0, "inputs": [], "outputs": [],
                               "properties": {}, "widgets_values": [text], "color": "#432", "bgcolor": "#653"})
        last_node = max([n["id"] for n in nodes_json] or [0])
        return {"last_node_id": last_node, "last_link_id": next_link - 1, "nodes": nodes_json, "links": links,
                "groups": [], "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [200, 400]}}, "version": 0.4}


# ---------------------------------------------------------------------------
# Shared WAN graph pieces
# ---------------------------------------------------------------------------

def wan_t2v_base(wf, unet=WAN21_T2V, width=640, height=368, length=25, prompt=PROMPT):
    model = wf.add("UNETLoader", unet_name=unet, weight_dtype="default")
    clip = wf.add("CLIPLoader", clip_name=UMT5, type="wan", device="default")
    vae = wf.add("VAELoader", vae_name=WAN_VAE)
    ms = wf.add("ModelSamplingSD3", model=model[0], shift=8.0)
    pos = wf.add("CLIPTextEncode", title="Positive prompt", clip=clip[0], text=prompt)
    neg = wf.add("CLIPTextEncode", title="Negative prompt", clip=clip[0], text=NEGATIVE)
    latent = wf.add("EmptyHunyuanLatentVideo", width=width, height=height, length=length, batch_size=1)
    return dict(model=ms, clip=clip, vae=vae, pos=pos, neg=neg, latent=latent)


def sample_and_save(wf, g, model, name, seed=42, steps=10, cfg=6.5, positive=None, negative=None, latent=None):
    ks = wf.add("KSampler", title=f"Sampler: {name}", model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name="uni_pc", scheduler="simple", positive=(positive or g["pos"][0]),
                negative=(negative or g["neg"][0]), latent_image=(latent or g["latent"][0]), denoise=1.0)
    dec = wf.add("VAEDecode", samples=ks[0], vae=g["vae"][0])
    wf.add("SaveAnimatedWEBP", title=f"Save: {name}", images=dec[0], filename_prefix=f"wan_bending/{name}",
           fps=16.0, lossless=False, quality=90, method="default")
    return ks


def attention_bend(wf, model, module, **kw):
    kw.setdefault("attention", "cross_text")
    kw.setdefault("blocks", "13-18")
    kw.setdefault("tokens", "all")
    kw.setdefault("renormalize", "keys")
    kw.setdefault("apply_to", "both")
    return wf.add("Attention Map Bending", model=model, bending_module=module[0], **kw)


# ---------------------------------------------------------------------------
# The workflows
# ---------------------------------------------------------------------------

def wf_attention_bending(oi):
    wf = Workflow(oi)
    g = wan_t2v_base(wf)
    sample_and_save(wf, g, g["model"][0], "baseline")
    rot = wf.add("Rotate Module (Bending)", angle_degrees=12.0, padding="border")
    early = attention_bend(wf, g["model"][0], rot, steps="0-2", title="Rotate, early steps 0-2")
    sample_and_save(wf, g, early[0], "rotate_early_steps")
    late = attention_bend(wf, g["model"][0], rot, steps="7-9", title="Rotate, late steps 7-9")
    sample_and_save(wf, g, late[0], "rotate_late_steps")
    scale = wf.add("Scale Module (Bending)", scale_factor=1.04, padding="border")
    sc = attention_bend(wf, g["model"][0], scale, title="Scale 1.04, all steps")
    sample_and_save(wf, g, sc[0], "scale_1.04")
    wf.note("Attention map bending on WAN 2.1 1.3B: the cross-attention maps between the video and the prompt are "
            "rotated / scaled frame by frame in the middle blocks 13-18.\n\n"
            "Compare: baseline, rotate during the early steps 0-2 (layout and coherence change), the same rotation "
            "during the late steps 7-9 (little change), and a 1.04x scale (only small scales stay coherent). "
            "Same seed everywhere.\n\nTry: blocks '*' (strongest), tokens 'horse' (+ clip and prompt "
            "inputs; much weaker), apply_to 'cond' (amplified by CFG), renormalize 'none'.\n\n" + EXPERIMENTAL
            + "\n\n" + DOWNLOADS, column=0, width=520)
    return wf


def wf_attention_capture(oi):
    wf = Workflow(oi)
    text = wf.add("PrimitiveStringMultiline", title="Prompt (shared)", value=PROMPT)
    g = wan_t2v_base(wf)
    g["pos"].links["text"] = text[0]
    cap = wf.add("Attention Map Capture", model=g["model"][0], attention="cross_text", blocks="13-18",
                 tokens="horse", clip=g["clip"][0], prompt=text[0], title="Capture 'horse' (unbent)")
    ks = sample_and_save(wf, g, cap[0], "capture_baseline")
    read = wf.add("Read Attention Maps", attention_maps=cap[1], latent=ks[0], token="sum", block="mean", step="all",
                  normalize="per_video", colormap="inferno", match_video_frames=True)
    wf.add("SaveAnimatedWEBP", title="Save: attention maps (unbent)", images=read[0],
           filename_prefix="wan_bending/attention_maps_baseline", fps=16.0, lossless=False, quality=90, method="default")
    rot = wf.add("Rotate Module (Bending)", angle_degrees=20.0, padding="border")
    bent = attention_bend(wf, g["model"][0], rot, steps="0-2")
    cap2 = wf.add("Attention Map Capture", model=bent[0], attention="cross_text", blocks="13-18", tokens="horse",
                  clip=g["clip"][0], prompt=text[0], title="Capture 'horse' (after the bend)")
    ks2 = sample_and_save(wf, g, cap2[0], "capture_rotated")
    read2 = wf.add("Read Attention Maps", attention_maps=cap2[1], latent=ks2[0], token="sum", block="mean", step="all",
                   normalize="per_video", colormap="inferno", match_video_frames=True)
    wf.add("SaveAnimatedWEBP", title="Save: attention maps (rotated)", images=read2[0],
           filename_prefix="wan_bending/attention_maps_rotated", fps=16.0, lossless=False, quality=90, method="default")
    wf.note("Where the video attends to the word 'horse', for every sampling step side "
            "by side (left = first step), averaged over blocks 13-18. The second branch captures after a rotation, "
            "so you see the bent maps and the video they produce.\n\nThe prompt is shared by the text encoder and the "
            "capture nodes (to find the word). Use 'prompt' as tokens to record every prompt token, then pick one with "
            "Read Attention Maps' token input (indices are in the capture report).\n\n" + EXPERIMENTAL + "\n\n"
            + DOWNLOADS, column=0, width=520)
    return wf


def wf_texture_ops(oi):
    wf = Workflow(oi)
    g = wan_t2v_base(wf)
    sample_and_save(wf, g, g["model"][0], "baseline")
    sharpen = wf.add("Sharpen Module (Bending)", amount=1.0, sigma=1.0)
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], sharpen, blocks="*", title="Sharpen")[0], "sharpen")
    blur = wf.add("Gaussian Blur Module (Bending)", sigma=1.0)
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], blur, blocks="*", title="Blur")[0], "blur")
    amp = wf.add("Multiply Scalar Module (Bending)", scalar=1.5)
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], amp, blocks="*", renormalize="none",
                                          title="Amplify 1.5 (renormalize none)")[0], "amplify_1.5")
    wf.note("Texture operations on every block: sharpening the attention maps gives more defined content and "
            "richer surface texture, blurring reduces texture while keeping the composition, and amplifying above 1 "
            "intensifies texture and contrast (below 1 dampens detail).\n\nAmplify must use "
            "renormalize 'none': renormalising divides the factor back out.\n\n" + EXPERIMENTAL + "\n\n" + DOWNLOADS,
            column=0, width=520)
    return wf


def wf_temporal(oi):
    wf = Workflow(oi)
    g = wan_t2v_base(wf)
    sample_and_save(wf, g, g["model"][0], "baseline")
    rot = wf.add("Rotate Module (Bending)", angle_degrees=30.0, padding="border")
    ramp = wf.add("Frame Ramp (Bending)", bending_module=rot[0], w_start=0.0, w_end=1.0, curve="smooth")
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], ramp, steps="0-4",
                                          title="Rotation growing over time")[0], "rotate_ramp_over_frames")
    shift = wf.add("Temporal Shift Module (Bending)", frames=2, padding="border")
    dit = wf.add("DiT Block Bending", model=g["model"][0], bending_module=shift[0], blocks="double:10-20",
                 stream="img", spatial=True, t_start=1.0, t_end=0.5)
    sample_and_save(wf, g, dit[0], "temporal_shift_blocks_10-20")
    tblur = wf.add("Temporal Blur Module (Bending)", sigma=1.5)
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], tblur, attention="self_query", blocks="10-20",
                                          renormalize="none", steps="0-4", title="Self-attention smeared in time")[0],
                    "self_attention_temporal_blur")
    wf.note("Temporal bending: transforms that change over the frames.\n"
            "- Frame Ramp: the rotation of the cross-attention maps grows from 0 (first frame) to full (last frame).\n"
            "- Temporal Shift on the block outputs: content is pushed 2 latent frames (8 video frames) later in "
            "blocks 10-20 during the noisy first half of sampling.\n"
            "- Temporal Blur on the self-attention output: motion smears and lingers.\n\n"
            "WAN packs 4 video frames into one latent frame (the first frame alone).\n\n" + EXPERIMENTAL + "\n\n"
            + DOWNLOADS, column=0, width=520)
    return wf


def wf_self_attention(oi):
    wf = Workflow(oi)
    g = wan_t2v_base(wf)
    sample_and_save(wf, g, g["model"][0], "baseline")
    tr = wf.add("Translate Module (Bending)", dx=0.25, dy=0.0, padding="border")
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], tr, attention="self_key", steps="0-4",
                                          title="self_key: read from 25% to the left")[0], "self_key_translate")
    flip = wf.add("Flip Module (Bending)", direction="horizontal")
    sample_and_save(wf, g, attention_bend(wf, g["model"][0], flip, attention="self_query", steps="0-2",
                                          title="self_query: results land mirrored")[0], "self_query_flip")
    wf.note("Self-attention bending. The full self-attention map is far too large to build (~10^9 entries per head), so:\n"
            "- self_key bends where each position reads from: the values are moved on the video grid, the layout "
            "that decides where to look is not.\n"
            "- self_query bends where each position's result lands: equivalent to bending the map along its query "
            "axis (exact for linear ops such as flip, rotate, translate, blur).\n\n" + EXPERIMENTAL + "\n\n"
            + DOWNLOADS, column=0, width=520)
    return wf


def wf_dit_blocks(oi):
    wf = Workflow(oi)
    g = wan_t2v_base(wf)
    sample_and_save(wf, g, g["model"][0], "baseline")
    rot = wf.add("Rotate Module (Bending)", angle_degrees=15.0, padding="border")
    gated = wf.add("Timestep Gated Bending", bending_module=rot[0], t_start=1.0, t_end=0.6, ramp="cosine",
                   ramp_width=0.1)
    dit = wf.add("DiT Block Bending", model=g["model"][0], bending_module=gated[0], blocks="double:13-18",
                 stream="img", spatial=True)
    sample_and_save(wf, g, dit[0], "dit_blocks_rotate_gated")
    mul = wf.add("Multiply Scalar Module (Bending)", scalar=1.3)
    mb = wf.add("Model Bending", model=g["model"][0], bending_module=mul[0], path="blocks.15.ffn")
    sample_and_save(wf, g, mb[0], "ffn_15_x1.3")
    wf.note("The existing layer/block bending nodes on WAN. Block outputs are laid out on the video grid "
            "(frames x height x width), so spatial ops act on the picture, frame by frame.\n"
            "- DiT Block Bending: rotate the outputs of blocks 13-18, faded out by diffusion time (t 1.0 -> 0.6).\n"
            "- Model Bending: any layer path, e.g. blocks.15.ffn, blocks.15.self_attn, blocks.15.cross_attn "
            "(see 'Bendable Layer Catalogue').\n\nWAN blocks have no text stream: stream 'txt' does nothing.\n\n"
            + DOWNLOADS, column=0, width=520)
    return wf


def wf_i2v(oi):
    wf = Workflow(oi)
    model = wf.add("UNETLoader", unet_name=WAN21_I2V, weight_dtype="default")
    clip = wf.add("CLIPLoader", clip_name=UMT5, type="wan", device="default")
    vae = wf.add("VAELoader", vae_name=WAN_VAE)
    cv = wf.add("CLIPVisionLoader", clip_name=CLIP_VISION)
    img = wf.add("LoadImage", image="example.png")
    enc = wf.add("CLIPVisionEncode", clip_vision=cv[0], image=img[0], crop="none")
    ms = wf.add("ModelSamplingSD3", model=model[0], shift=8.0)
    pos = wf.add("CLIPTextEncode", title="Positive prompt", clip=clip[0],
                 text="The scene comes alive: the camera slowly pushes in, gentle wind, natural motion")
    neg = wf.add("CLIPTextEncode", title="Negative prompt", clip=clip[0], text=NEGATIVE)
    i2v = wf.add("WanImageToVideo", positive=pos[0], negative=neg[0], vae=vae[0], clip_vision_output=enc[0],
                 start_image=img[0], width=640, height=368, length=25, batch_size=1)
    g = dict(vae=vae, pos=i2v, neg=i2v, latent=i2v)
    kw = dict(positive=i2v[0], negative=i2v[1], latent=i2v[2])
    sample_and_save(wf, g, ms[0], "i2v_baseline", **kw)
    rot = wf.add("Rotate Module (Bending)", angle_degrees=15.0, padding="border")
    img_bend = attention_bend(wf, ms[0], rot, attention="cross_image", blocks="17-24", steps="0-2",
                              title="Bend attention to the CLIP image")
    sample_and_save(wf, g, img_bend[0], "i2v_cross_image_rotate", **kw)
    txt_bend = attention_bend(wf, ms[0], rot, attention="cross_text", blocks="17-24", steps="0-2",
                              title="Bend attention to the prompt")
    sample_and_save(wf, g, txt_bend[0], "i2v_cross_text_rotate", **kw)
    wf.note("Image-to-video (WAN 2.1 I2V 14B, 40 blocks: the middle blocks are ~17-24). WAN 2.1 I2V attends to the "
            "CLIP image tokens (257) and to the prompt in two separate cross-attention calls; 'cross_image' bends "
            "the first, 'cross_text' the second.\n\nWAN 2.2 I2V has no CLIP image cross-attention: use cross_text.\n\n"
            + EXPERIMENTAL + "\n\nModels: wan2.1_i2v_480p_14B_fp8_scaled.safetensors (diffusion_models) and "
            "clip_vision_h.safetensors (clip_vision) from Comfy-Org/Wan_2.1_ComfyUI_repackaged.\n" + DOWNLOADS,
            column=0, width=520)
    return wf


def wf_wan22(oi):
    wf = Workflow(oi)
    high = wf.add("UNETLoader", title="High-noise expert", unet_name=WAN22_HIGH, weight_dtype="default")
    low = wf.add("UNETLoader", title="Low-noise expert", unet_name=WAN22_LOW, weight_dtype="default")
    clip = wf.add("CLIPLoader", clip_name=UMT5, type="wan", device="default")
    vae = wf.add("VAELoader", vae_name=WAN_VAE)
    ms_high = wf.add("ModelSamplingSD3", model=high[0], shift=8.0)
    ms_low = wf.add("ModelSamplingSD3", model=low[0], shift=8.0)
    pos = wf.add("CLIPTextEncode", title="Positive prompt", clip=clip[0], text=PROMPT)
    neg = wf.add("CLIPTextEncode", title="Negative prompt", clip=clip[0], text=NEGATIVE)
    latent = wf.add("EmptyHunyuanLatentVideo", width=640, height=368, length=25, batch_size=1)
    rot = wf.add("Rotate Module (Bending)", angle_degrees=12.0, padding="border")
    bent_high = attention_bend(wf, ms_high[0], rot, blocks="17-24", title="Bend the high-noise expert (layout)")

    def two_stage(high_model, name):
        k1 = wf.add("KSamplerAdvanced", title=f"High noise: {name}", model=high_model, add_noise="enable",
                    noise_seed=42, steps=20, cfg=3.5, sampler_name="euler", scheduler="simple", positive=pos[0],
                    negative=neg[0], latent_image=latent[0], start_at_step=0, end_at_step=10,
                    return_with_leftover_noise="enable")
        k2 = wf.add("KSamplerAdvanced", title=f"Low noise: {name}", model=ms_low[0], add_noise="disable",
                    noise_seed=42, steps=20, cfg=3.5, sampler_name="euler", scheduler="simple", positive=pos[0],
                    negative=neg[0], latent_image=k1[0], start_at_step=10, end_at_step=10000,
                    return_with_leftover_noise="disable")
        dec = wf.add("VAEDecode", samples=k2[0], vae=vae[0])
        wf.add("SaveAnimatedWEBP", title=f"Save: {name}", images=dec[0], filename_prefix=f"wan_bending/{name}",
               fps=16.0, lossless=False, quality=90, method="default")
    two_stage(ms_high[0], "wan22_baseline")
    two_stage(bent_high[0], "wan22_high_noise_rotate")
    wf.note("WAN 2.2 14B uses two models: a high-noise expert for the first half of sampling (layout, motion) and a "
            "low-noise expert for the rest (detail). Bend each MODEL separately; here only the high-noise expert is "
            "bent, in its middle blocks (40 blocks: ~17-24), so the layout changes and the details stay clean.\n\n"
            + EXPERIMENTAL + "\n\nModels: wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors and "
            "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors from "
            "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged (diffusion_models); text encoder and VAE "
            "as for WAN 2.1 (wan_2.1_vae).", column=0, width=520)
    return wf


WORKFLOWS = {
    "wan21_t2v_attention_bending.json": wf_attention_bending,
    "wan21_t2v_attention_capture.json": wf_attention_capture,
    "wan21_t2v_texture_ops.json": wf_texture_ops,
    "wan21_t2v_temporal_bending.json": wf_temporal,
    "wan21_t2v_self_attention_bending.json": wf_self_attention,
    "wan21_t2v_dit_block_bending.json": wf_dit_blocks,
    "wan21_i2v_attention_bending.json": wf_i2v,
    "wan22_t2v_14b_attention_bending.json": wf_wan22,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_ui(name, wf, oi):
    """Node types exist, links are consistent and typed, widget counts match the node definitions."""
    problems = []
    nodes = {n["id"]: n for n in wf["nodes"]}
    for n in wf["nodes"]:
        if n["type"] in FRONTEND_ONLY:
            continue
        info = oi.get(n["type"])
        if info is None:
            problems.append(f"{name}: unknown node type {n['type']}")
            continue
        specs = {**(info["input"].get("required") or {}), **(info["input"].get("optional") or {})}
        expected = 0
        for spec in specs.values():
            if is_widget(spec):
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                expected += 1 + bool(opts.get("control_after_generate")) + bool(opts.get("image_upload"))
        if len(n["widgets_values"]) != expected:
            problems.append(f"{name}: {n['type']} #{n['id']} has {len(n['widgets_values'])} widget values, expected {expected}")
        for req in input_names(info, "required"):
            spec = info["input"]["required"][req]
            if not is_widget(spec) and not any(i["name"] == req and i["link"] for i in n["inputs"]):
                problems.append(f"{name}: {n['type']} #{n['id']} required input {req!r} is not connected")
    for lid, src, sslot, dst, dslot, typ in wf["links"]:
        s, d = nodes.get(src), nodes.get(dst)
        if s is None or d is None:
            problems.append(f"{name}: link {lid} references a missing node")
            continue
        if s["outputs"][sslot]["type"] != typ:
            problems.append(f"{name}: link {lid} type {typ} != output type {s['outputs'][sslot]['type']}")
        dinput = d["inputs"][dslot]
        if dinput["link"] != lid:
            problems.append(f"{name}: link {lid} not registered on {d['type']} input {dslot}")
        if lid not in s["outputs"][sslot]["links"]:
            problems.append(f"{name}: link {lid} not registered on {s['type']} output {sslot}")
        if "widget" not in dinput and dinput["type"] != typ and dinput["type"] != "*":
            problems.append(f"{name}: link {lid} {typ} -> {d['type']}.{dinput['name']} ({dinput['type']})")
    return problems


def to_api(wf, oi):
    """UI workflow -> API prompt (what the frontend sends to /prompt)."""
    links = {lid: (src, sslot) for lid, src, sslot, _, _, _ in wf["links"]}
    api = {}
    for n in wf["nodes"]:
        if n["type"] in FRONTEND_ONLY:
            continue
        info = oi[n["type"]]
        specs = {**(info["input"].get("required") or {}), **(info["input"].get("optional") or {})}
        inputs, values = {}, list(n["widgets_values"])
        for name in input_names(info, "required") + input_names(info, "optional"):
            spec = specs[name]
            if is_widget(spec):
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                inputs[name] = values.pop(0)
                if opts.get("control_after_generate") or opts.get("image_upload"):
                    values.pop(0)
        for i in n["inputs"]:
            if i.get("link"):
                src, slot = links[i["link"]]
                inputs[i["name"]] = [str(src), slot]
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return api


MODEL_INPUTS = {"unet_name", "clip_name", "vae_name", "image"}


def check_prompt(name, api, url):
    """Server-side validation. Only missing model files (and the example image) are acceptable errors."""
    body = json.dumps({"prompt": api, "client_id": "make_video_workflows"}).encode()
    req = urllib.request.Request(url + "/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        # Models are installed and the prompt was queued: remove it so nothing runs.
        pid = result.get("prompt_id")
        if pid:
            urllib.request.urlopen(urllib.request.Request(url + "/queue", data=json.dumps({"delete": [pid]}).encode(),
                                                          headers={"Content-Type": "application/json"}), timeout=30)
            urllib.request.urlopen(urllib.request.Request(url + "/interrupt", data=b"{}",
                                                          headers={"Content-Type": "application/json"}), timeout=30)
        return []
    except urllib.error.HTTPError as e:
        result = json.loads(e.read() or b"{}")
    problems = []
    for nid, err in (result.get("node_errors") or {}).items():
        for e in err.get("errors", []):
            field = (e.get("extra_info") or {}).get("input_name")
            if e.get("type") == "value_not_in_list" and field in MODEL_INPUTS:
                continue
            problems.append(f"{name}: node {nid} ({err.get('class_type')}): {e.get('type')} {field}: {e.get('details')}")
    if not result.get("node_errors") and result.get("error"):
        problems.append(f"{name}: {result['error']}")
    return problems


def fetch_object_info(url):
    with urllib.request.urlopen(url + "/object_info", timeout=120) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--check", action="store_true", help="only validate the saved workflows")
    parser.add_argument("--no-prompt-check", action="store_true", help="skip the /prompt validation")
    args = parser.parse_args()
    oi = fetch_object_info(args.url)
    os.makedirs(OUT_DIR, exist_ok=True)
    problems = []
    for fname, build in WORKFLOWS.items():
        path = os.path.join(OUT_DIR, fname)
        if not args.check:
            ui = build(oi).to_ui()
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(ui, f, indent=1)
                f.write("\n")
        with open(path, encoding="utf-8") as f:
            ui = json.load(f)
        found = check_ui(fname, ui, oi)
        if not found and not args.no_prompt_check:
            found = check_prompt(fname, to_api(ui, oi), args.url)
        problems.extend(found)
        print(("OK   " if not found else "FAIL ") + fname)
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
