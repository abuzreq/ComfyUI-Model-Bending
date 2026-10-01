# Attention map bending and capture for video DiTs (WAN 2.1 / 2.2).
#
# The cross-attention map softmax(QK^T / sqrt(d)) between the video tokens and the text tokens is materialised,
# each text token's column is laid out on the latent (frames, h, w) grid, bent frame by frame with any
# BENDING_MODULE, optionally renormalised, and multiplied by V.
# Self-attention is bent without materialising its L x L map:
#   - "self_query" bends the map along the query axis. For a linear op M this is M.P.V = M.(P.V), i.e. the
#     attention output laid out on the grid and bent (divided by M.1 with renormalize="keys").
#   - "self_key" bends where each position reads from: the values are bent on the grid before attention.
#
# Everything here is EXPERIMENTAL: it was validated on tiny random-weight WAN models built from ComfyUI's own
# WanModel class (tests/test_video.py), not on real WAN weights.
#
# Mechanism: ComfyUI's wrap_attn (comfy/ldm/modules/attention.py) routes every attention call that carries
# transformer_options to transformer_options["optimized_attention_override"](func, q, k, v, heads=..., ...).
# One shared override (attention_override) serves every node: each node appends an AttentionEntry to
# transformer_options[ENTRIES_KEY], and per forward pass registers pre/post hooks on the targeted blocks'
# self_attn / cross_attn modules so the override knows which block and which call it is in.
import json
import math
import re

import torch
import torch.nn as nn

from .bendutils import (
    LOG_TAG,
    current_step,
    in_window,
    info,
    model_sampling_of,
    normalized_t,
    parse_step_str_to_ranges,
    t_window_from,
    token_layout_for,
    tokens_to_video,
    video_to_tokens,
    warn,
)

ENTRIES_KEY = "model_bending_attention_entries"
PREV_OVERRIDE_KEY = "model_bending_prev_attention_override"
OVERRIDE_KEY = "optimized_attention_override"

ATTENTION_KINDS = ("cross_text", "cross_image", "self_query", "self_key")
RENORM_MODES = ("keys", "none", "per_token_mass")
APPLY_TO = ("both", "cond", "uncond")
CATEGORY = "model_bending/video (experimental)"
EXPERIMENTAL_NOTE = ("Experimental: validated only on tiny random-weight WAN models in the test suite, "
                     "not yet on real video model weights.")

# Largest attention-probability chunk materialised at once (elements, float32). The cross-attention map of
# WAN 14B at 720p / 81 frames is ~75k x 512 per head and sample, so chunks go per sample and per few heads.
CHUNK_ELEMENTS = 1 << 26
MAX_CAPTURE_TOKENS = 32


def _diffusion_model(m):
    if hasattr(m, "model") and hasattr(m.model, "diffusion_model"):
        return m.model.diffusion_model
    return None


def attention_blocks(dm):
    """[(index, block)] of WAN-style blocks that own self_attn / cross_attn modules."""
    blocks = getattr(dm, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        return []
    return [(i, b) for i, b in enumerate(blocks)
            if isinstance(getattr(b, "self_attn", None), nn.Module) or isinstance(getattr(b, "cross_attn", None), nn.Module)]


def parse_index_list(spec, count, what):
    """'*' / '' / 'all' -> None (everything), '0-5, 9' -> sorted indices < count."""
    spec = (spec or "").strip()
    if spec.lower() in ("", "*", "all"):
        return None
    idx = parse_step_str_to_ranges(spec, max_steps=max(0, count - 1))
    if idx is None:
        raise ValueError(f"{LOG_TAG} invalid {what} {spec!r}; use e.g. '13-18' or '0,3,5-7'")
    return sorted(i for i in idx if 0 <= i < count)


# ---------------------------------------------------------------------------
# Prompt tokens
# ---------------------------------------------------------------------------

def _inner_tokenizer(clip):
    tok = getattr(clip, "tokenizer", None)
    name = getattr(tok, "clip", None)
    return getattr(tok, name, None) if isinstance(name, str) else None


def _decode(inner, ids):
    try:
        return inner.decode(ids, skip_special_tokens=False)
    except TypeError:
        return inner.decode(ids)
    except Exception:
        return "?"


def prompt_token_table(clip, prompt):
    """
    [(index, token_id, word_id, text)] for the prompt as the text encoder sees it (first encoder, first chunk),
    without the trailing padding. index is the position in the text context, i.e. the key index in cross-attention.
    """
    tokens = clip.tokenize(prompt, return_word_ids=True)
    key = next(iter(tokens))
    chunk = tokens[key][0]
    inner = _inner_tokenizer(clip)
    pad = getattr(inner, "pad_token", None)
    rows = []
    for i, item in enumerate(chunk):
        tok = item[0]
        wid = item[2] if len(item) > 2 else 0
        if isinstance(tok, int):
            text = _decode(inner, [tok]) if inner is not None else str(tok)
        else:
            text = "<embedding>"
        rows.append((i, tok, wid, text))
    while rows and rows[-1][2] == 0 and pad is not None and rows[-1][1] == pad:
        rows.pop()
    return rows


def _norm_word(s):
    return re.sub(r"[\W_]+", "", str(s).lower())


def _phrase_ids(clip, phrase):
    """Token ids of a phrase on its own, without start/end/padding tokens."""
    inner = _inner_tokenizer(clip)
    tokens = clip.tokenize(phrase)
    chunk = tokens[next(iter(tokens))][0]
    special = {getattr(inner, a, None) for a in ("start_token", "end_token", "pad_token")} - {None}
    return [item[0] for item in chunk if isinstance(item[0], int) and item[0] not in special]


def find_word_tokens(clip, table, words_spec):
    """
    Token indices of each comma-separated word or phrase in words_spec. A phrase is matched by its own token
    ids inside the prompt's (SentencePiece and CLIP BPE both encode a word the same way alone and mid-prompt);
    failing that, case-insensitively by the decoded text of consecutive tokens.
    """
    prompt_ids = [tok for _, tok, _, _ in table]
    texts = [_norm_word(text) for _, _, _, text in table]
    found, missing = [], []
    for phrase in (p.strip() for p in words_spec.split(",")):
        if not phrase:
            continue
        hit = False
        try:
            ids = _phrase_ids(clip, phrase)
        except Exception:
            ids = []
        if ids:
            for start in range(len(prompt_ids) - len(ids) + 1):
                if prompt_ids[start:start + len(ids)] == ids:
                    found.extend(table[start + j][0] for j in range(len(ids)))
                    hit = True
        if not hit:
            target = _norm_word(phrase)
            for start in range(len(texts)):
                acc, j = "", start
                while j < len(texts) and len(acc) < len(target) and target.startswith(acc + texts[j]):
                    acc += texts[j]
                    j += 1
                if acc and acc == target and texts[start]:
                    found.extend(table[i][0] for i in range(start, j))
                    hit = True
        if not hit:
            missing.append(phrase)
    return sorted(set(found)), missing


class TokenSpec:
    """Which keys of the attention map to bend: all, the non-padding prompt tokens, or explicit indices."""

    def __init__(self, mode, indices=None):
        self.mode = mode          # "all" | "prompt" | "indices"
        self.indices = indices

    def for_row(self, real_count, num_keys):
        if self.mode == "all":
            return None
        if self.mode == "prompt":
            if real_count is None or real_count >= num_keys:
                return None
            return list(range(real_count))
        return [i for i in self.indices if i < num_keys]


def resolve_tokens(tokens, clip=None, prompt=None, report=None):
    spec = (tokens or "").strip()
    low = spec.lower()
    if low in ("", "*", "all"):
        return TokenSpec("all")
    if low == "prompt":
        return TokenSpec("prompt")
    table = None
    if clip is not None and prompt:
        try:
            table = prompt_token_table(clip, prompt)
        except Exception as exc:  # tokenizers differ between encoders: never fail the graph over the report
            warn("attention bending: could not tokenize the prompt (%s)", exc)
    if report is not None and table is not None:
        report["prompt_tokens"] = [f"{i}: {text!r}" for i, _, _, text in table]
    if re.fullmatch(r"[\d\s,\-]+", spec):
        indices = parse_step_str_to_ranges(spec, max_steps=100000)
        if not indices:
            raise ValueError(f"{LOG_TAG} invalid token indices {spec!r}")
        return TokenSpec("indices", sorted(indices))
    if table is None:
        raise ValueError(f"{LOG_TAG} tokens={spec!r} names words: connect the text encoder (clip) and the prompt "
                         "so they can be found, or give token indices such as '0-3, 7'")
    indices, missing = find_word_tokens(clip, table, spec)
    if report is not None:
        report["word_tokens"] = indices
        if missing:
            report["words_not_found"] = missing
    if missing:
        warn("attention bending: word(s) %s not found in the prompt; see the report's prompt_tokens", missing)
    if not indices:
        raise ValueError(f"{LOG_TAG} none of {spec!r} was found in the prompt")
    return TokenSpec("indices", indices)


# ---------------------------------------------------------------------------
# One node's attention bend / capture
# ---------------------------------------------------------------------------

class AttentionEntry:
    """A bend (module) and/or capture at one kind of attention call in a set of blocks."""

    def __init__(self, dm, kind, blocks, tokens, module=None, gated=None, renorm="keys", apply_to="both",
                 heads=None, frames=None, steps=None, t_window=None, strength=1.0, capture=None, label="attention"):
        self.dm = dm
        self.kind = kind
        self.blocks = blocks                  # set of block indices
        self.tokens = tokens                  # TokenSpec
        self.module = module                  # BENDING_MODULE (ungated) or None for capture-only
        self.gated = gated                    # GatedBendingModule or None
        self.renorm = renorm
        self.apply_to = apply_to
        self.heads = heads                    # list or None
        self.frames = frames                  # list or None
        self.steps = steps                    # set or None
        self.t_window = t_window
        self.strength = float(strength)
        self.capture = capture                # AttentionCapture or None
        self.label = label
        # per forward pass
        self.active = False
        self.weight = 0.0
        self.step = None
        self.latent_shape = None
        self.cond_or_uncond = None
        self.real_counts = None
        self.site = None                      # (kind, block, has_img) while inside a hooked attention module
        self.calls = 0
        self.fired = set()
        self.warned = set()

    def __deepcopy__(self, memo):
        # Entries are live objects shared by the model's wrapper and its hooks: a cloned model must keep
        # pointing at the same entry (ComfyUI's clone copies lists and dicts, other tools may deepcopy).
        return self

    @property
    def bends(self):
        return self.module is not None and self.active and self.weight != 0.0

    def warn_once(self, key, msg, *args):
        if key not in self.warned:
            self.warned.add(key)
            warn(msg, *args)

    def call_kind(self):
        """Kind of the attention call being made now (advances the per-module call counter)."""
        if self.site is None:
            return None
        family, _, has_img = self.site
        n = self.calls
        self.calls += 1
        if family == "self":
            return "self"
        if has_img:  # WanI2VCrossAttention: the CLIP-image call comes first, then the text call
            return "cross_image" if n == 0 else "cross_text"
        return "cross_text"

    def rows(self, batch):
        """Batch rows this entry acts on, from the CFG layout (cond_or_uncond: 0 = cond, 1 = uncond)."""
        cu = self.cond_or_uncond or [0]
        if batch % len(cu) != 0:
            return list(range(batch))
        per = batch // len(cu)
        rows = []
        for j, c in enumerate(cu):
            if self.apply_to == "both" or (self.apply_to == "cond" and c == 0) or (self.apply_to == "uncond" and c == 1):
                rows.extend(range(j * per, (j + 1) * per))
        return rows

    def capture_row(self, batch):
        """The first conditional sample (attention maps are recorded for it only)."""
        cu = self.cond_or_uncond or [0]
        if 0 not in cu or batch % len(cu) != 0:
            return None
        return cu.index(0) * (batch // len(cu))

    def head_list(self, num_heads):
        return list(range(num_heads)) if self.heads is None else [h for h in self.heads if h < num_heads]

    def frame_mask(self, frames, device):
        if self.frames is None:
            return None
        mask = torch.zeros(frames, dtype=torch.bool, device=device)
        idx = [f for f in self.frames if f < frames]
        if idx:
            mask[idx] = True
        return mask

    # -- the bend itself ------------------------------------------------------
    def bend_video(self, vid):
        """Apply the module to a (B, C, F, h, w) float tensor with strength and frame selection."""
        bent = self.module(vid)
        if not isinstance(bent, torch.Tensor) or bent.shape != vid.shape:
            raise ValueError(f"{LOG_TAG} {self.label}: the bending module changed the map shape "
                             f"{tuple(vid.shape)} -> {tuple(bent.shape) if isinstance(bent, torch.Tensor) else type(bent)}")
        bent = torch.nan_to_num(bent.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.weight != 1.0:
            bent = vid + self.weight * (bent - vid)
        mask = self.frame_mask(vid.shape[2], vid.device)
        if mask is not None:
            bent = torch.where(mask.view(1, 1, -1, 1, 1), bent, vid)
        return bent

    def bend_map(self, p, layout, keys):
        """
        Bend attention probabilities p (1, hc, Lq, Lk) in place along the query (video) axis, for the key
        columns `keys` (None = all), then renormalise.
        """
        f, h, w = layout.grid
        n, pre = f * h * w, layout.prefix
        img = p[:, :, pre:pre + n, :]
        cols = img if keys is None else img[..., keys]           # (1, hc, n, s)
        _, hc, _, s = cols.shape
        vid = cols.permute(0, 1, 3, 2).reshape(1, hc * s, f, h, w)
        new = self.bend_video(vid).reshape(1, hc, s, n).permute(0, 1, 3, 2)
        if self.renorm == "keys":
            # Rows must sum to 1 again. A row whose mass was moved out of the frame (zero padding) keeps its
            # original values instead of becoming 0/0. The untouched columns are summed directly: 1 - sum(cols)
            # cancels catastrophically when a row's mass sits on the bent tokens.
            total = new.sum(-1)                                   # (1, hc, n)
            if keys is not None:
                others = sorted(set(range(img.shape[-1])) - set(keys))
                if others:
                    total = total + img[..., others].sum(-1)
            dead = total.abs() < 1e-6
            if bool(dead.any()):
                new = torch.where(dead.unsqueeze(-1), cols, new)
                total = torch.where(dead, torch.ones_like(total), total)
        elif self.renorm == "per_token_mass":
            old_mass, new_mass = cols.sum(2, keepdim=True), new.sum(2, keepdim=True)
            ok = new_mass.abs() > 1e-12
            new = torch.where(ok, new * old_mass / torch.where(ok, new_mass, torch.ones_like(new_mass)), new)
        if keys is None:
            img.copy_(new)
        else:
            img[..., keys] = new
        if self.renorm == "keys":
            img.div_(total.unsqueeze(-1))
        return p

    def bend_tokens(self, x, layout, channels):
        """Bend (1, L, C) tokens on their grid, restricted to `channels` (None = all)."""
        vid, pre, post = tokens_to_video(x.float(), layout)
        sub = vid if channels is None else vid[:, channels]
        bent = self.bend_video(sub)
        if channels is not None:
            out = vid.clone()
            out[:, channels] = bent
            bent = out
        return video_to_tokens(bent, pre, post).to(x.dtype)

    def coverage(self, layout, device):
        """M.1: how much of each position the bend keeps (for self_query renormalisation)."""
        f, h, w = layout.grid
        ones = torch.ones(1, 1, f, h, w, device=device)
        return self.bend_video(ones)[0, 0]                       # (f, h, w)


class AttentionCapture:
    """Head-mean attention maps of selected text tokens, recorded per step and block for the first cond sample."""

    def __deepcopy__(self, memo):
        return self

    def __init__(self, grid_scale=16, frame_repeat=4):
        self.grid_scale = grid_scale
        self.frame_repeat = frame_repeat
        self.reset(None)

    def reset(self, signature):
        self.signature = signature
        self.maps = {}          # (step, block) -> [S, F, h, w] float16 (head mean)
        self.tokens = None      # key indices of the stored maps
        self.token_labels = None
        self.last_step = -1

    def begin(self, transformer_options, step):
        schedule = (transformer_options or {}).get("sample_sigmas")
        signature = None
        if schedule is not None:
            signature = (len(schedule), round(float(schedule[0]), 5), round(float(schedule[-1]), 5))
        if signature != self.signature or (step is not None and step < self.last_step):
            self.reset(signature)
        if step is not None:
            self.last_step = max(self.last_step, step)

    def has_data(self):
        return bool(self.maps)

    def record(self, step, block, maps, keys):
        key = (step, block)
        if key in self.maps:  # 2nd-order samplers evaluate a step twice: keep the first
            return
        self.maps[key] = maps.to("cpu", torch.float16)
        if self.tokens is None:
            self.tokens = list(keys)


# ---------------------------------------------------------------------------
# The shared override
# ---------------------------------------------------------------------------

def _split_heads(t, heads, skip_reshape):
    if skip_reshape:
        return t
    b, l, hd = t.shape
    return t.view(b, l, heads, hd // heads).transpose(1, 2)


def _merge_heads(t):
    b, h, l, d = t.shape
    return t.transpose(1, 2).reshape(b, l, h * d)


def _attention_probs(q, k, scale):
    """softmax(q.k^T * scale) in float32; q (1, hc, Lq, d), k (1, hc, Lk, d)."""
    s = torch.matmul(q * scale, k.transpose(-1, -2))
    return s.float().softmax(dim=-1)


def _head_chunk(lq, lk):
    return max(1, CHUNK_ELEMENTS // max(1, lq * lk))


def attention_override(func, *args, **kwargs):
    to = kwargs.get("transformer_options") or {}
    entries = to.get(ENTRIES_KEY) or ()
    prev = to.get(PREV_OVERRIDE_KEY)

    def base(*a, **kw):
        return prev(func, *a, **kw) if prev is not None else func(*a, **kw)

    live = []
    for e in entries:
        kind = e.call_kind()
        if kind is not None:
            live.append((e, kind))
    if not live or len(args) < 3:
        return base(*args, **kwargs)
    if kwargs.get("mask") is not None:
        for e, _ in live:
            e.warn_once("mask", "%s: attention with a mask is not bent (not used by WAN)", e.label)
        return base(*args, **kwargs)

    q, k, v = args[:3]
    is_self = live[0][1] == "self"
    if is_self:
        return _self_attention(base, q, k, v, args[3:], kwargs, [e for e, _ in live])
    return _cross_attention(base, q, k, v, args[3:], kwargs, live)


def _self_attention(base, q, k, v, rest, kwargs, entries):
    heads = kwargs.get("heads")
    skip_reshape = kwargs.get("skip_reshape", False)
    skip_out = kwargs.get("skip_output_reshape", False)
    key_entries = [e for e in entries if e.kind == "self_key" and e.bends]
    query_entries = [e for e in entries if e.kind == "self_query" and e.bends]
    if not key_entries and not query_entries:
        return base(q, k, v, *rest, **kwargs)

    def as_tokens(t):  # (B, L, h*d) view of q/k/v or of the output
        return _merge_heads(t) if (t.ndim == 4) else t

    def from_tokens(t, like_heads):
        return _split_heads(t, heads, False) if like_heads else t

    num_tokens = (v.shape[2] if skip_reshape else v.shape[1])
    if key_entries:
        vt = as_tokens(v).clone()
        for e in key_entries:
            layout = token_layout_for(e.dm, num_tokens, e.latent_shape)
            if layout is None:
                e.warn_once("grid", "%s: could not lay out %d self-attention tokens on the video grid", e.label, num_tokens)
                continue
            d = vt.shape[-1] // heads
            ch = None if e.heads is None else [hh * d + j for hh in e.head_list(heads) for j in range(d)]
            for r in e.rows(vt.shape[0]):
                vt[r:r + 1] = e.bend_tokens(vt[r:r + 1], layout, ch)
            e.fired.add(e.site[1])
        v = from_tokens(vt, skip_reshape)
    out = base(q, k, v, *rest, **kwargs)
    if query_entries:
        out_heads = skip_out
        ot = as_tokens(out) if out_heads else out
        ot = ot.clone()
        for e in query_entries:
            layout = token_layout_for(e.dm, ot.shape[1], e.latent_shape)
            if layout is None:
                e.warn_once("grid", "%s: could not lay out %d self-attention tokens on the video grid", e.label, ot.shape[1])
                continue
            d = ot.shape[-1] // heads
            ch = None if e.heads is None else [hh * d + j for hh in e.head_list(heads) for j in range(d)]
            if e.renorm == "per_token_mass":
                e.warn_once("ptm", "%s: renormalize='per_token_mass' has no meaning for self_query; using 'none'", e.label)
            for r in e.rows(ot.shape[0]):
                orig = ot[r:r + 1]
                bent = e.bend_tokens(orig, layout, ch)
                if e.renorm == "keys":
                    cov = e.coverage(layout, ot.device)                  # (f, h, w)
                    f, h, w = layout.grid
                    cov_t = cov.reshape(1, f * h * w, 1)
                    pre = layout.prefix
                    img_b = bent[:, pre:pre + f * h * w].float()
                    img_o = orig[:, pre:pre + f * h * w].float()
                    dead = cov_t.abs() < 1e-6
                    fixed = torch.where(dead, img_o, img_b / torch.where(dead, torch.ones_like(cov_t), cov_t))
                    if ch is not None:  # only the selected heads' channels were bent
                        keep = torch.zeros(ot.shape[-1], dtype=torch.bool, device=ot.device)
                        keep[ch] = True
                        fixed = torch.where(keep, fixed, img_o)
                    bent = bent.clone()
                    bent[:, pre:pre + f * h * w] = fixed.to(bent.dtype)
                ot[r:r + 1] = bent
            e.fired.add(e.site[1])
        out = from_tokens(ot, out_heads) if out_heads else ot
    return out


def _cross_attention(base, q, k, v, rest, kwargs, live):
    heads = kwargs.get("heads")
    skip_reshape = kwargs.get("skip_reshape", False)
    skip_out = kwargs.get("skip_output_reshape", False)
    entries = [e for e, kind in live if kind == e.kind and (e.bends or (e.capture is not None and e.active))]
    if not entries:
        return base(q, k, v, *rest, **kwargs)

    qh, kh, vh = (_split_heads(t, heads, skip_reshape) for t in (q, k, v))
    b, nh, lq, d = qh.shape
    lk = kh.shape[2]
    scale = d ** -0.5
    benders = [e for e in entries if e.bends]
    capturers = [e for e in entries if e.capture is not None and e.active]

    # Which (row, head) pairs need the explicit map: rows/heads that are bent, plus the capture row.
    bend_rows = sorted({r for e in benders for r in e.rows(b)})
    bend_heads = sorted({hh for e in benders for hh in e.head_list(nh)})
    cap_rows = {e.capture_row(b) for e in capturers} - {None}
    rows = sorted(set(bend_rows) | cap_rows)
    full_cover = bend_rows == list(range(b)) and bend_heads == list(range(nh))
    out_base = None if full_cover else base(q, k, v, *rest, **kwargs)
    out = _split_heads(out_base, heads, skip_out) if out_base is not None and not skip_out else out_base
    out = out.clone() if out is not None else torch.empty_like(qh)

    layouts = {}
    for e in entries:
        layouts[id(e)] = token_layout_for(e.dm, lq, e.latent_shape)
        if layouts[id(e)] is None:
            e.warn_once("grid", "%s: could not lay out %d video tokens on the latent grid %s; not bent",
                        e.label, lq, e.latent_shape)

    chunk = _head_chunk(lq, lk)
    for r in rows:
        head_set = set(bend_heads) if r in bend_rows else set()
        for e in capturers:
            if e.capture_row(b) == r:
                head_set |= set(e.head_list(nh))
        head_idx = sorted(head_set)
        cap_sums = {id(e): None for e in capturers if e.capture_row(b) == r}
        for c0 in range(0, len(head_idx), chunk):
            hs = head_idx[c0:c0 + chunk]
            p = _attention_probs(qh[r:r + 1, hs], kh[r:r + 1, hs], scale)        # (1, hc, lq, lk)
            for e in entries:
                layout = layouts[id(e)]
                if layout is None:
                    continue
                local = [j for j, hh in enumerate(hs) if hh in set(e.head_list(nh))]
                if not local:
                    continue
                real = e.real_counts[r] if (e.kind == "cross_text" and e.real_counts and r < len(e.real_counts)) else None
                keys = e.tokens.for_row(real, lk)
                if keys is not None and not keys:
                    continue
                if e.bends and r in e.rows(b):
                    sub = p[:, local] if len(local) < len(hs) else p
                    sub = e.bend_map(sub, layout, keys)
                    if len(local) < len(hs):
                        p[:, local] = sub
                    e.fired.add(e.site[1])
                if id(e) in cap_sums:
                    f, h, w = layout.grid
                    pre = layout.prefix
                    ck = keys if keys is not None else list(range(lk))
                    ck = ck[:MAX_CAPTURE_TOKENS]
                    m = p[:, local, pre:pre + f * h * w][..., ck].sum(1)[0]      # (n, s) summed over heads
                    cap_sums[id(e)] = m if cap_sums[id(e)] is None else cap_sums[id(e)] + m
                    e._cap_keys = ck
            if r in bend_rows:
                write = [j for j, hh in enumerate(hs) if hh in bend_heads]
                if write:
                    o = torch.matmul(p[:, write].to(vh.dtype), vh[r:r + 1, [hs[j] for j in write]])
                    out[r:r + 1, [hs[j] for j in write]] = o.to(out.dtype)
        for e in capturers:
            if id(e) in cap_sums and cap_sums[id(e)] is not None:
                layout = layouts[id(e)]
                f, h, w = layout.grid
                n_heads = len(e.head_list(nh))
                maps = (cap_sums[id(e)] / max(1, n_heads)).t().reshape(-1, f, h, w)
                e.capture.record(e.step, e.site[1], maps, e._cap_keys)
                e.fired.add(e.site[1])
    if skip_out:
        return out
    return _merge_heads(out)


# ---------------------------------------------------------------------------
# Installing an entry on a model
# ---------------------------------------------------------------------------

def install_entry(m, entry, strict=False):
    """Add the entry to the model's attention-override list and wrap the model so it knows the current pass."""
    dm = entry.dm
    family = "self" if entry.kind.startswith("self") else "cross"
    attr = "self_attn" if family == "self" else "cross_attn"
    targets = []
    for i, block in attention_blocks(dm):
        if i in entry.blocks and isinstance(getattr(block, attr, None), nn.Module):
            mod = getattr(block, attr)
            targets.append((i, mod, hasattr(mod, "k_img")))
    if not targets:
        msg = (f"{type(dm).__name__} has no '{attr}' modules in blocks {sorted(entry.blocks)[:8]}...; attention "
               "bending supports WAN-family models (blocks.N.self_attn / blocks.N.cross_attn)")
        if strict:
            raise ValueError(f"{LOG_TAG} {msg}")
        warn(msg)
    if entry.kind == "cross_image" and targets and not any(t[2] for t in targets):
        msg = "attention='cross_image' needs an image-to-video model with CLIP image cross-attention (WAN 2.1 I2V)"
        if strict:
            raise ValueError(f"{LOG_TAG} {msg}")
        warn(msg)

    to = m.model_options.setdefault("transformer_options", {})
    current = to.get(OVERRIDE_KEY)
    if current is not attention_override:
        if current is not None:
            to[PREV_OVERRIDE_KEY] = current
        to[OVERRIDE_KEY] = attention_override
    to[ENTRIES_KEY] = list(to.get(ENTRIES_KEY) or []) + [entry]

    prev_wrapper = m.model_options.get("model_function_wrapper")
    needs_t = entry.t_window is not None or entry.gated is not None
    state = {"announced": False}

    def pre_hook(_i, _has_img):
        def hook(module, args, kwargs):
            entry.site = (family, _i, _has_img)
            entry.calls = 0
        return hook

    def post_hook(module, args, kwargs, output):
        entry.site = None

    def wrapper(apply_model, params):
        c = params["c"] or {}
        transformer_options = c.get("transformer_options") or {}
        step = current_step(transformer_options)
        t = normalized_t(model_sampling_of(apply_model), params["timestep"]) if needs_t else None
        if entry.capture is not None:
            entry.capture.begin(transformer_options, step)
        entry.step = step if step is not None else 0
        entry.latent_shape = tuple(params["input"].shape)
        entry.cond_or_uncond = params.get("cond_or_uncond") or transformer_options.get("cond_or_uncond")
        ctx = c.get("c_crossattn")
        entry.real_counts = None
        if isinstance(ctx, torch.Tensor) and ctx.ndim == 3:
            nonzero = ctx.detach().abs().amax(-1) > 0                          # (B, Lctx)
            pos = torch.arange(1, nonzero.shape[1] + 1, device=nonzero.device)
            entry.real_counts = [int(v) for v in (nonzero * pos).amax(-1).tolist()]
        active = True
        if needs_t and t is None:
            entry.warn_once("no_t", "%s: cannot resolve diffusion time; time-windowed bend skipped", entry.label)
            active = False
        if active and not in_window(step, t, entry.steps, entry.t_window):
            active = False
        weight = entry.strength
        if active and entry.gated is not None:
            weight *= entry.gated.weight(t)
        entry.active = active
        entry.weight = weight if active else 0.0
        do_hooks = active and (entry.module is not None and entry.weight != 0.0 or entry.capture is not None)
        handles = []
        if do_hooks:
            for i, mod, has_img in targets:
                handles.append(mod.register_forward_pre_hook(pre_hook(i, has_img), with_kwargs=True))
                handles.append(mod.register_forward_hook(post_hook, with_kwargs=True))
        try:
            if prev_wrapper is not None:
                out = prev_wrapper(apply_model, params)
            else:
                out = apply_model(params["input"], params["timestep"], **c)
        finally:
            for h in handles:
                h.remove()
            entry.site = None
            entry.active = False
        if do_hooks and not entry.fired:
            entry.warn_once("none", "%s: no %s call was bent in blocks %s; is this a WAN model, and do the "
                            "tokens/heads/frames select anything?", entry.label, entry.kind,
                            ",".join(str(i) for i, _, _ in targets[:8]))
        elif entry.fired and not state["announced"]:
            state["announced"] = True
            info("%s: %s bent in blocks %s (latent %s)", entry.label, entry.kind,
                 ",".join(str(i) for i in sorted(entry.fired)), entry.latent_shape)
        return out

    m.set_model_unet_function_wrapper(wrapper)
    return targets


_JSON_KEYS = {"module_type", "module_args", "inner", "attention", "blocks", "tokens", "renormalize", "apply_to",
              "heads", "frames", "steps", "t", "blend", "label"}


def apply_attention_bends_json(model, items, resolve_op, build_op, problems, resolved, strict=False):
    """
    Experimental: install the bends JSON's top-level "attention_bends" list, each item
    {"module_type", "module_args"[, "inner"], "attention": "cross_text", "blocks": "13-18", "tokens": "all",
     "renormalize": "keys", "apply_to": "both", "heads": "*", "frames": "*", "steps": "0-2", "t": [hi, lo],
     "blend": 1.0, "label"}. resolve_op(item, where) validates the op (shared with the bends format), build_op
    builds it. Prompt words cannot be resolved here (no text encoder): use token indices.
    """
    if not isinstance(items, list):
        problems.append("'attention_bends' must be a list; none applied")
        return model
    m = model.clone()
    dm = _diffusion_model(m)
    if dm is None:
        problems.append("attention_bends: no diffusion model found; none applied")
        return model
    for i, item in enumerate(items):
        where = f"attention bend #{i}: "
        if not isinstance(item, dict):
            problems.append(f"{where}is not an object; skipped")
            continue
        for key in item:
            if key not in _JSON_KEYS:
                problems.append(f"{where}unknown key {key!r} (ignored)")
        kind = item.get("attention", "cross_text")
        renorm = item.get("renormalize", "keys")
        apply_to = item.get("apply_to", "both")
        bad = [(n, v, allowed) for n, v, allowed in (("attention", kind, ATTENTION_KINDS),
                                                    ("renormalize", renorm, RENORM_MODES),
                                                    ("apply_to", apply_to, APPLY_TO)) if v not in allowed]
        if bad:
            for n, v, allowed in bad:
                problems.append(f"{where}{n}={v!r} must be one of {', '.join(allowed)}; skipped")
            continue
        try:
            op = resolve_op(item, where)
            module = build_op(op)
            tokens = resolve_tokens(str(item.get("tokens", "all"))) if kind.startswith("cross") else TokenSpec("all")
            step_idx = parse_index_list(str(item.get("steps", "*")), 10000, "steps")
            t = item.get("t")
            t_window = None
            if t is not None:
                if not (isinstance(t, (list, tuple)) and len(t) == 2):
                    raise ValueError(f"t must be [hi, lo], got {t!r}")
                t_window = t_window_from(float(t[0]), float(t[1]))
            blend = float(item.get("blend", 1.0))
            entry = AttentionEntry(
                dm, kind, _block_set(dm, str(item.get("blocks", "*"))), tokens, module=module, renorm=renorm,
                apply_to=apply_to, heads=parse_index_list(str(item.get("heads", "*")), 4096, "heads"),
                frames=parse_index_list(str(item.get("frames", "*")), 4096, "frames"),
                steps=set(step_idx) if step_idx is not None else None, t_window=t_window, strength=blend,
                label=item.get("label") or f"attention bend #{i}")
        except ValueError as exc:
            problems.append(f"{where}{str(exc).replace(LOG_TAG + ' ', '')}; skipped")
            continue
        targets = install_entry(m, entry, strict=strict)
        resolved.append({**op, "attention": kind, "blocks": [b for b, _, _ in targets],
                         "tokens": tokens.mode if tokens.mode != "indices" else tokens.indices,
                         "renormalize": renorm, "apply_to": apply_to, "heads": entry.heads or "*",
                         "frames": entry.frames or "*", "steps": sorted(entry.steps) if entry.steps else "*",
                         **({"t": list(t_window)} if t_window else {}), "blend": blend, "experimental": True})
    if strict and problems:
        raise ValueError(f"{LOG_TAG} attention_bends problems (strict):\n- " + "\n- ".join(problems))
    return m


def _common_inputs():
    return {
        "attention": (list(ATTENTION_KINDS), {"default": "cross_text", "tooltip": (
            "cross_text: video-to-prompt cross-attention. cross_image: video-to-CLIP-image "
            "cross-attention (WAN 2.1 I2V). self_query: bend where each position's self-attention result lands. "
            "self_key: bend where each position reads from (values moved, layout kept).")}),
        "blocks": ("STRING", {"default": "*", "tooltip": (
            "Block indices, e.g. '13-18' (the middle of WAN 2.1 1.3B's 30 blocks; 14B has 40). "
            "'*' bends every block, the strongest effect.")}),
        "tokens": ("STRING", {"default": "all", "tooltip": (
            "cross attention only: 'all' (every key, padding included), 'prompt' (only the prompt's own "
            "tokens, not the padding), indices '0-3, 7', or prompt words 'horse, red car' (needs clip + prompt). "
            "Single words give much weaker effects than 'all'.")}),
    }


def _window_inputs():
    return {
        "steps": ("STRING", {"default": "*", "tooltip": (
            "Sampling steps, e.g. '0-2'. Early steps change layout and coherence; late steps change "
            "little.")}),
        "t_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "t_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    }


def _block_set(dm, spec):
    blocks = attention_blocks(dm)
    count = (blocks[-1][0] + 1) if blocks else 0
    idx = parse_index_list(spec, max(count, 1), "blocks")
    return set(range(count)) if idx is None else set(idx)


class AttentionMapBending:
    DESCRIPTION = (
        "Bends the cross-attention maps of a video DiT (WAN 2.1 / 2.2) "
        "frame by frame with any bending module (rotate, scale, translate, flip, blur, sharpen, multiply = amplify, "
        "...), and self-attention through its outputs (self_query) or values (self_key). " + EXPERIMENTAL_NOTE)
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "bending_module": ("BENDING_MODULE",),
                **_common_inputs(),
                "renormalize": (list(RENORM_MODES), {"default": "keys", "tooltip": (
                    "keys: each video position's attention sums to 1 again (a valid softmax). per_token_mass: each "
                    "token keeps its total attention. none: raw (amplify only acts without renormalisation).")}),
                "apply_to": (list(APPLY_TO), {"default": "both", "tooltip": (
                    "CFG passes to bend. 'cond' alone is amplified by the guidance scale: stronger, glitchier.")}),
            },
            "optional": {
                **_window_inputs(),
                "heads": ("STRING", {"default": "*", "tooltip": "Attention heads, e.g. '0-3' (WAN 1.3B has 12, 14B has 40)"}),
                "frames": ("STRING", {"default": "*", "tooltip": "Latent frames to bend, e.g. '0-5' (WAN: 1 latent frame = 4 video frames)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01,
                                       "tooltip": "Blend between the unbent (0) and bent (1) map"}),
                "clip": ("CLIP", {"tooltip": "Text encoder, to find prompt words given in 'tokens'"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The positive prompt, to find words given in 'tokens'"}),
                "strict": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "report")
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, bending_module, attention, blocks, tokens, renormalize, apply_to, steps="*",
              t_start=1.0, t_end=0.0, heads="*", frames="*", strength=1.0, clip=None, prompt="", strict=False):
        m = model.clone()
        dm = _diffusion_model(m)
        if dm is None:
            raise ValueError(f"{LOG_TAG} Attention Map Bending: no diffusion model found")
        report = {"experimental": True, "model": type(dm).__name__, "attention": attention, "renormalize": renormalize,
                  "apply_to": apply_to}
        token_spec = resolve_tokens(tokens, clip, prompt, report) if attention.startswith("cross") else TokenSpec("all")
        if not attention.startswith("cross") and (tokens or "all").strip().lower() not in ("all", "*", ""):
            warn("Attention Map Bending: 'tokens' only applies to cross attention; ignored for %s", attention)
        gated = bending_module if getattr(bending_module, "is_gated", False) else None
        module = bending_module.inner if gated is not None else bending_module
        if renormalize == "keys" and attention != "self_key" and type(module).__name__ == "MultiplyScalarModule":
            warn("Attention Map Bending: multiplying (amplify) is undone by renormalize='keys'; use 'none'")
            report["warning"] = "amplify (multiply) has no effect with renormalize='keys'; use 'none'"
        step_idx = parse_index_list(steps, 10000, "steps")
        entry = AttentionEntry(
            dm, attention, _block_set(dm, blocks), token_spec, module=module, gated=gated, renorm=renormalize,
            apply_to=apply_to, heads=parse_index_list(heads, 4096, "heads"),
            frames=parse_index_list(frames, 4096, "frames"),
            steps=set(step_idx) if step_idx is not None else None,
            t_window=t_window_from(t_start, t_end), strength=strength, label="Attention Map Bending")
        targets = install_entry(m, entry, strict=strict)
        report.update({
            "blocks": [i for i, _, _ in targets],
            "tokens": token_spec.mode if token_spec.mode != "indices" else token_spec.indices,
            "heads": entry.heads or "all", "frames": entry.frames or "all",
            "steps": sorted(entry.steps) if entry.steps is not None else "all",
            "t": list(entry.t_window) if entry.t_window else None, "strength": strength,
            "module": type(module).__name__,
        })
        info("Attention Map Bending is experimental (validated on tiny random-weight WAN models only)")
        return (m, json.dumps(report, indent=2))


class AttentionMapCapture:
    DESCRIPTION = (
        "Records the cross-attention maps of chosen prompt tokens (head mean, first conditional sample) for every "
        "step and chosen block while a sampler runs, after any upstream attention bends. Read them as video frames "
        "with 'Read Attention Maps'. " + EXPERIMENTAL_NOTE)
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        common = _common_inputs()
        common["attention"] = (["cross_text", "cross_image"], {"default": "cross_text"})
        common["tokens"] = ("STRING", {"default": "prompt", "tooltip": (
            "Tokens to record: prompt words 'horse' (needs clip + prompt), indices '0-3', or 'prompt' "
            f"(each prompt token, at most {MAX_CAPTURE_TOKENS})")})
        common["blocks"] = ("STRING", {"default": "13-18", "tooltip": "Blocks to record, e.g. '13-18'"})
        return {
            "required": {"model": ("MODEL",), **common},
            "optional": {
                "steps": ("STRING", {"default": "*", "tooltip": "Sampling steps to record"}),
                "heads": ("STRING", {"default": "*"}),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "strict": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "ATTENTION_MAPS", "STRING")
    RETURN_NAMES = ("MODEL", "attention_maps", "report")
    FUNCTION = "attach"
    CATEGORY = CATEGORY

    def attach(self, model, attention, blocks, tokens, steps="*", heads="*", clip=None, prompt="", strict=False):
        m = model.clone()
        dm = _diffusion_model(m)
        if dm is None:
            raise ValueError(f"{LOG_TAG} Attention Map Capture: no diffusion model found")
        report = {"experimental": True, "model": type(dm).__name__, "attention": attention}
        spec = tokens if tokens.strip().lower() not in ("all", "*", "") else "prompt"
        if spec != tokens:
            warn("Attention Map Capture: tokens='all' always sums to 1 per position; recording 'prompt' instead")
        token_spec = resolve_tokens(spec, clip, prompt, report)
        from .model_bending_nodes import latent_downscale_ratio
        from .bendutils import dit_patch_size
        patch = dit_patch_size(dm) or (1, 2, 2)
        capture = AttentionCapture(grid_scale=patch[1] * latent_downscale_ratio(m), frame_repeat=4)
        capture.token_labels = None
        if clip is not None and prompt:
            try:
                capture.token_labels = {i: text for i, _, _, text in prompt_token_table(clip, prompt)}
            except Exception as exc:
                warn("Attention Map Capture: could not tokenize the prompt (%s)", exc)
        step_idx = parse_index_list(steps, 10000, "steps")
        entry = AttentionEntry(dm, attention, _block_set(dm, blocks), token_spec, module=None,
                               heads=parse_index_list(heads, 4096, "heads"),
                               steps=set(step_idx) if step_idx is not None else None,
                               capture=capture, label="Attention Map Capture")
        targets = install_entry(m, entry, strict=strict)
        report.update({"blocks": [i for i, _, _ in targets],
                       "tokens": token_spec.mode if token_spec.mode != "indices" else token_spec.indices})
        return (m, capture, json.dumps(report, indent=2))


_INFERNO = torch.tensor([[0.0, 0.0, 0.016], [0.258, 0.039, 0.406], [0.578, 0.148, 0.404],
                         [0.865, 0.317, 0.226], [0.988, 0.645, 0.040], [0.988, 1.0, 0.645]])


def apply_colormap(x, name="inferno"):
    """x in [0, 1], any shape -> (..., 3)."""
    if name == "gray":
        return x.unsqueeze(-1).repeat(*([1] * x.ndim), 3)
    lut = _INFERNO.to(x.device)
    pos = x.clamp(0, 1) * (len(lut) - 1)
    lo = pos.floor().long().clamp(max=len(lut) - 2)
    frac = (pos - lo.float()).unsqueeze(-1)
    return lut[lo] * (1 - frac) + lut[lo + 1] * frac


class ReadAttentionMaps:
    DESCRIPTION = ("Turns recorded attention maps into video frames (IMAGE batch): bright = the token attends "
                   "there. " + EXPERIMENTAL_NOTE)
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention_maps": ("ATTENTION_MAPS",),
                "latent": ("LATENT", {"tooltip": "Connect the output of the sampler that used the capture model, "
                                                 "so this node runs after sampling"}),
                "token": ("STRING", {"default": "sum", "tooltip": "'sum' of the recorded tokens, or a token index"}),
                "block": ("STRING", {"default": "mean", "tooltip": "'mean' over recorded blocks, or a block index"}),
                "step": ("STRING", {"default": "all", "tooltip": (
                    "'all' = every recorded step side by side, 'mean', or a step index")}),
                "normalize": (["per_video", "per_frame"], {"default": "per_video"}),
                "colormap": (["inferno", "gray"], {"default": "inferno"}),
                "match_video_frames": ("BOOLEAN", {"default": True, "tooltip": (
                    "Repeat each latent frame 4x (after the first) so the maps line up with WAN's decoded frames")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "report")
    FUNCTION = "read"
    CATEGORY = CATEGORY

    def read(self, attention_maps, latent, token, block, step, normalize, colormap="inferno", match_video_frames=True):
        cap = attention_maps
        if not cap.has_data():
            raise ValueError(f"{LOG_TAG} no attention maps were recorded: connect the capture node's MODEL to a "
                             "sampler and this node's latent input to that sampler's output")
        steps = sorted({s for s, _ in cap.maps})
        blocks = sorted({b for _, b in cap.maps})

        def pick(values, spec, what):
            spec = str(spec).strip().lower()
            if spec in ("mean", "all"):
                return values
            try:
                want = int(spec)
            except ValueError:
                raise ValueError(f"{LOG_TAG} {what} must be 'mean'/'all' or an index, got {spec!r}")
            if want not in values:
                raise ValueError(f"{LOG_TAG} {what} {want} was not recorded; recorded: {values}")
            return [want]

        use_blocks = pick(blocks, block, "block")
        use_steps = pick(steps, step, "step")
        tokens = cap.tokens or []
        tok = str(token).strip().lower()
        if tok == "sum":
            tok_idx = None
        else:
            try:
                key = int(tok)
            except ValueError:
                raise ValueError(f"{LOG_TAG} token must be 'sum' or a token index, got {token!r}")
            if key not in tokens:
                raise ValueError(f"{LOG_TAG} token {key} was not recorded; recorded: {tokens}")
            tok_idx = tokens.index(key)

        def step_map(s):
            ms = [cap.maps[(s, b)].float() for b in use_blocks if (s, b) in cap.maps]
            m = torch.stack(ms).mean(0)                                   # (S, F, h, w)
            return m.sum(0) if tok_idx is None else m[tok_idx]           # (F, h, w)

        if str(step).strip().lower() == "mean":
            panels = [torch.stack([step_map(s) for s in use_steps]).mean(0)]
        else:
            panels = [step_map(s) for s in use_steps]
        video = torch.cat(panels, dim=-1)                                 # steps side by side: (F, h, w * n)
        if normalize == "per_frame":
            lo = video.amin(dim=(1, 2), keepdim=True)
            hi = video.amax(dim=(1, 2), keepdim=True)
        else:
            lo, hi = video.min(), video.max()
        video = (video - lo) / (hi - lo).clamp_min(1e-12)
        scale = int(cap.grid_scale)
        video = torch.nn.functional.interpolate(video.unsqueeze(1), scale_factor=scale, mode="nearest")[:, 0]
        if match_video_frames and video.shape[0] > 1 and cap.frame_repeat > 1:
            video = torch.cat([video[:1], video[1:].repeat_interleave(cap.frame_repeat, dim=0)])
        images = apply_colormap(video, colormap)                          # (frames, H, W, 3)
        labels = cap.token_labels or {}
        report = {
            "experimental": True,
            "steps": steps, "blocks": blocks,
            "tokens": [f"{i}: {labels.get(i, '')!r}" if labels else i for i in tokens],
            "shown": {"token": token, "block": use_blocks, "step": use_steps},
            "frames": int(images.shape[0]),
        }
        return (images.float().cpu(), json.dumps(report, indent=2))


NODE_CLASS_MAPPINGS = {
    "Attention Map Bending": AttentionMapBending,
    "Attention Map Capture": AttentionMapCapture,
    "Read Attention Maps": ReadAttentionMaps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention Map Bending": "Attention Map Bending (Experimental)",
    "Attention Map Capture": "Attention Map Capture (Experimental)",
    "Read Attention Maps": "Read Attention Maps (Experimental)",
}
