# Bends JSON, version 1.2

This is the format the web UI copies with "Copy Bends", posts to `/web_bend_demo/selection`, and the
`Apply Bends from JSON` node reads. Other tools that produce or consume bends should use it.

Versions 1.1 and 1.2 only add optional keys to version 1, so every v1 document is a valid 1.2 document.
Version 1.2 (plugin 0.3.1) adds the single metadata key `kb`.

## Example

```json
{
  "version": 1.2,
  "selected_part": "diffusion_model",
  "steps_min": null,
  "steps_max": null,
  "max_denoising_steps": 200,
  "bends": [
    {
      "path": "output_blocks.*.1",
      "module_type": "multiply",
      "module_args": {"scalar": 1.5},
      "t": [1.0, 0.5],
      "blend": 0.8,
      "label": "boost attention early",
      "guard": {"nan": "zero", "max_std_ratio": 4},
      "kb": {"dataset": "abuzreq/model-bending-knowledge-base", "record": "e75fee762a99a75a3f82"}
    },
    {
      "path": "middle_block.1",
      "module_type": "subset",
      "module_args": {"percentage": 0.5, "dim": "channel", "seed": 0},
      "inner": {"module_type": "rotate", "module_args": {"angle_degrees": 90}},
      "steps": "0-3"
    }
  ]
}
```

## Top-level keys

| Key | Type | Meaning |
|---|---|---|
| `bends` | list | The bends. Required. |
| `steps_min`, `steps_max` | int or null | Default step range for bends without their own `steps`. Either end may be omitted. |
| `max_denoising_steps` | int | Upper bound used when a step range is open-ended. Default 200, clamped to 1–1000. |
| `selected_part` | string | Module that paths are relative to. Default `diffusion_model`. |
| `version` | number | Optional. A value above 1.2 produces a warning. |
| `attention_bends` | list | *Experimental.* Attention-map bends for video DiTs (WAN); see [below](#attention_bends-experimental). |

## Per-bend keys

| Key | Since | Meaning |
|---|---|---|
| `path` | 1 | Dotted layer path, relative to `selected_part`. Required. |
| `module_type` | 1 | Op name (see [Ops](#ops)). Defaults to `rotate` if missing. |
| `module_args` | 1 | Op arguments. Missing ones take the op's default. |
| `angle` | 1 (legacy) | Shorthand for `rotate` with that angle. |
| `steps` | 1.1 | Step list, 0-based: `"0-4,9"`, `"3-"`, `"-7"`, `"*"`. Overrides the top-level range. |
| `t` | 1.1 | `[hi, lo]` window in diffusion time: 1 = pure noise, 0 = clean image. |
| `blend` | 1.1 | 0–1. The result is `x + blend·(bend(x) − x)`. |
| `label` | 1.1 | Free text, echoed in the node's report. |
| `guard` | 1.1 | Safety options applied after the bend (see below). |
| `inner` | 1.1 | The op wrapped by `module_type: "subset"` or `"frame_ramp"`: `{"module_type", "module_args"}`. |
| `kb` | 1.2 | Where the bend came from in the bend knowledge base: `{"dataset", "record"}` or `{"dataset", "cell"}`. Not used to bend; kept in `resolved_json`. |

### `path`

- Each segment accepts shell wildcards (`*`, `?`, `[4-8]`), for example `output_blocks.*.1`. Matches are
  applied in model order. (1.1)
- A path to a container the model never calls directly (`middle_block`, `output_blocks.4`) is bent at its last
  child, which produces the same tensor.
- A path to a list (`input_blocks`) cannot be bent and is skipped with a warning.

### `t`

Diffusion time follows the noise level, so a window does not depend on the step count, scheduler, shift or
denoise strength. For eps and v-prediction models it is timestep ÷ 999. For flow models (Flux, SD3) it is
sigma. If time cannot be resolved for a forward pass, the bend is skipped for that pass.

Rough guide: 1–0.7 composition, 0.7–0.2 shapes and style, 0.2–0 detail.

### `guard`

| Key | Values | Effect |
|---|---|---|
| `nan` | `"zero"` (default), `"clamp"`, `"none"` | `zero` replaces NaN and Inf with 0. `clamp` maps NaN to 0 and Inf to the largest finite value. `none` leaves them. |
| `max_std_ratio` | number > 0 | Shrinks the result around its mean so its spread is at most that multiple of the unbent spread. |
| `preserve_norm` | boolean | Rescales each channel to its unbent L2 norm, so the bend changes direction and not energy. |

### `kb`

An object. Known fields: `dataset` (string, e.g. `abuzreq/model-bending-knowledge-base`) and either `record`
(a record id: hex, 8–40 characters) or `cell` (a cell key such as
`sd1|out.mid|res_block|Conv2d|scale|shrink|late|txt2img`). The node only checks that `kb` is an object, so the
knowledge base can add fields without a plugin release. A `kb` that is not an object is one warning and is
dropped. It has no effect on the render.

## Ops

| `module_type` | Arguments (default; hard limits) |
|---|---|
| `add_scalar` | `scalar` (0; ±100) |
| `add_noise` | `noise_std` (0; ±100), `seed` (42) |
| `multiply` | `scalar` (1; ±100) |
| `rotate` | `angle_degrees` (0; ±360) |
| `threshold` | `threshold` (0) |
| `scale` | `scale_factor` (1; ±100) |
| `erosion`, `dilation`, `gradient` | `kernel_size` (3; 1–10) |
| `sobel` | `normalized` (true) |
| `fourier` (1.1) | `cutoff_freq` (5; 0–10), `amp_factor` (2; ±10) |
| `subset` (1.1) | `percentage` (0.5; 0–1), `dim` (`batch`, `channel` or `spatial`), `seed` (0), plus `inner` |
| `translate` (1.1, plugin 0.3) | `dx`, `dy` (0; ±1, fractions of width / height), `padding` (`border`, `zeros` or `reflection`) |
| `flip` (1.1, plugin 0.3) | `direction` (`horizontal`, `vertical` or `both`) |
| `blur` (1.1, plugin 0.3) | `sigma` (1; 0–20, in latent cells / tokens) |
| `sharpen` (1.1, plugin 0.3) | `amount` (1; ±10), `sigma` (1; 0.05–20). Unsharp mask: `x + amount·(x − blur(x))` |
| `temporal_shift` (1.1, plugin 0.3) | `frames` (1; ±64 latent frames), `padding` (`border`, `wrap` or `zeros`). Video only. |
| `temporal_blur` (1.1, plugin 0.3) | `sigma` (1; 0–16 latent frames). Video only. |
| `frame_reverse` (1.1, plugin 0.3) | none. Video only. |
| `frame_ramp` (1.1, plugin 0.3) | `w_start` (0), `w_end` (1) (±4), `curve` (`linear`, `ease_in`, `ease_out`, `smooth`, `triangle`), plus `inner`: the inner op's strength changes over the frames. |

Ops act on each frame of a video (5-D) activation, except the temporal ones, which act across frames. On video
DiTs (WAN), token outputs are laid out on their frames × height × width grid first.

The hard limits are enforced only when the node's `clamp` input is `hard` or `safe`.

## `attention_bends` (experimental)

Bends attention maps of video DiTs (WAN 2.1 / 2.2), like the *Attention Map Bending* node. Each item:

```json
{
  "attention_bends": [
    {
      "module_type": "rotate", "module_args": {"angle_degrees": 12},
      "attention": "cross_text", "blocks": "13-18", "tokens": "all",
      "renormalize": "keys", "apply_to": "both",
      "heads": "*", "frames": "*", "steps": "0-2", "t": [1.0, 0.0], "blend": 1.0, "label": "early rotation"
    }
  ]
}
```

| Key | Default | Meaning |
|---|---|---|
| `module_type`, `module_args`, `inner` | | Any op from the table above. |
| `attention` | `cross_text` | `cross_text` (video ↔ prompt), `cross_image` (video ↔ CLIP image, WAN 2.1 I2V), `self_query`, `self_key`. |
| `blocks` | `*` | Block indices, e.g. `13-18`. |
| `tokens` | `all` | Cross attention only: `all`, `prompt` (without padding) or indices `0-3, 7`. Prompt words need the text encoder, so they only work in the node. |
| `renormalize` | `keys` | `keys`, `per_token_mass` or `none`. `multiply` needs `none`. |
| `apply_to` | `both` | CFG passes: `both`, `cond` or `uncond`. |
| `heads`, `frames`, `steps` | `*` | Index lists. `frames` are latent frames. |
| `t`, `blend`, `label` | | As for bends. |

The item may be the only content of the document (`bends` can be empty). The report and `resolved_json` list each
attention bend with its blocks and arguments made explicit.

## Rules for tools

- **Order.** Bends on the same layer apply last-to-first in list order.
- **Unknown keys and arguments** are ignored with a warning. With the node's `strict` input they raise.
  An unknown `module_type` always raises.
- **Canonical form.** The node's `resolved_json` output is the same document with every wildcard expanded,
  every container resolved and every argument filled in. Feeding it back gives the same result. Store and
  share this form when you need reproducibility.
- **Warnings.** All messages are logged with the `[model-bending]` prefix. The node lists them in its `report`
  output, and `POST /web_bend_demo/selection` returns them as `warnings`.
- **`kb`.** Keep it when you copy or re-save a bend. Drop it if you change the bend so that it no longer
  matches its source (another path, op or amount).

## `Apply Bends from JSON` extras

These are node inputs, not part of the JSON:

- `{{a}}` … `{{d}}` in the JSON text are replaced by the node's optional `a`–`d` inputs.
- `strict` turns every warning into an error.
- `clamp`: `hard` enforces the limits in the table above. `safe` narrows them with a `safe_ranges` JSON:

  ```json
  {
    "multiply": {"scalar": [0, 3]},
    "paths": {"output_blocks.*": {"multiply": {"scalar": [0.5, 1.5]}}}
  }
  ```

  Keys under `paths` are globs matched against the resolved layer path. Clamped values are listed in the
  report as requested → applied.

## Compatibility

| Reader | v1 document | 1.1 document | 1.2 document |
|---|---|---|---|
| This version (0.3.1) | Unchanged behaviour | Full support | Full support |
| Plugin 0.3.0 | Unchanged behaviour | Full support | `kb` is an unknown key: a warning, or an error with `strict`. Everything else applies. |
| Older plugin versions | Supported | New keys are ignored, so windows, blend and guards do not apply. Wildcard paths are skipped with a warning. `subset`, `fourier` and the ops added in plugin 0.3 raise "Unknown module_type". `attention_bends` is ignored with a warning. | As for 1.1; `kb` is also ignored with a warning. |

A tool that must work with 0.3.0 can remove `kb` from the text it gives the node and keep it in its own records.

## Limits to know

- The web UI keeps the 1.1 and 1.2 keys of a bend it receives, and sends them back when you move that bend's
  slider, but it has no controls to edit them. Moving the slider drops `kb`, since the bend then differs from
  its source. It writes three ops itself: `add_noise`, `multiply` and `rotate`.
- The web UI stores one bend per layer path. `Apply Bends from JSON` accepts several.
- `subset` with `percentage` 0 or 1 does nothing. On token (3-D) activations only `batch` and `channel` are meaningful.

## Where the code is

Readers and validation, in [`nodes.py`](../nodes.py):

| Name | Role |
|---|---|
| `BENDS_JSON_VERSION`, `_TOP_KEYS`, `_BEND_KEYS` | Version and accepted keys |
| `BEND_OPS` | The single table of ops, argument types, defaults and hard limits |
| `normalize_bends` | Normalises raw bend objects; shared by the parser and the web UI route |
| `parse_bends_json` | Parses and validates a document |
| `_resolve_op`, `_check_guard` | Argument typing with "did you mean" hints; guard validation |
| `_clamp_op` | `clamp` and `safe_ranges` |
| `apply_bends_to_model` | Turns bends into hooks and builds the report and the resolved form |
| `ApplyBendsFromJSON` | The node: placeholders, `strict`, `clamp`, `report`, `resolved_json` |
| `apply_attention_bends_json` ([`attention_bending.py`](../attention_bending.py)) | `attention_bends` |
| `api_get_selection`, `api_set_selection` | `GET` / `POST /web_bend_demo/selection` |

Semantics, in [`bendutils.py`](../bendutils.py):

| Name | Role |
|---|---|
| `expand_path_pattern` | Wildcard matching |
| `resolve_hook_target` | Container expansion and unhookable paths |
| `parse_step_str_to_ranges` | The `steps` syntax |
| `normalized_t` | The definition of `t` |
| `blend_and_guard` | `blend` and `guard` |

Writers, in the web UI:

| File | Role |
|---|---|
| [`web/js/image-manager.js`](../web/js/image-manager.js) | `_setupCopyBendsButton` builds the clipboard JSON. `getBends`, `setBend` and `setBends` are the bend store. `sendSelectionToComfyUI` posts it. |
| [`web/js/config.js`](../web/js/config.js) | `BENDING_TYPES`: the ops the web UI can create |

Worked examples: the `test_json_*` and `test_web_ui_selection_route_keeps_v11_keys` tests in
[`tests/test_hooks.py`](../tests/test_hooks.py), `test_bends_json_new_ops_and_attention_bends` in
[`tests/test_video.py`](../tests/test_video.py), and [`tests/test_web_bend_store.js`](../tests/test_web_bend_store.js).
