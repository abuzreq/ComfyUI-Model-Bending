# Experiment Scripts

Scripts for running **batch bending experiments**, exploring results, and sharing them without ComfyUI.

**Note:** You can find a sample of the data generated when running this flow: "run_bending_experiments --> export_experiments --> compute_metrics --> build_offline_explorer" in this [shared folder](https://drive.google.com/file/d/1u88qPI9ot3qx1QqygalbBGeK5PQIctSw/view?usp=sharing). Simply unzip and open the explorer_offline.html

## What these scripts do

| Script | Purpose |
|--------|---------|
| **run_bending_experiments.py** | Run many bend combinations (layers × bend types × settings) and save images and data. |
| **export_experiments.py** | Export a batch to a single folder (data + images + optional metrics) for sharing or offline use. |
| **serve_explorer_standalone.py** | Serve the experiment explorer in the browser without ComfyUI (from experiments or export folder). |
| **build_offline_explorer.py** | Build a single HTML file that works offline (open via file://, no server). |
| **compute_metrics.py** | Compute similarity/distance metrics (LPIPS, CLIP, etc.) between unbent and bent results. |
| **analyze_bending_impact.py** | Summarize metrics by group (e.g. by layer type or bend type) and output JSON/CSV. |

**Typical workflow:** Run experiments → (optional) export with metrics → open explorer in browser or build offline HTML.

---

## Prerequisites

- **ComfyUI** with this plugin installed, and a workflow that uses the **Interactive Bending WebUI** node.
- **Python 3** and `pip install requests` for the run script.
- For metrics: `pip install lpips`; for CLIP metrics: `pip install transformers`.

---

## run_bending_experiments.py

Runs systematic experiments: for each combination of layer, bend type, and settings, it queues the workflow to ComfyUI and saves the results. Output goes to  
`{ComfyUI output}/web_bend_demo_experiments/{batch_id}/`  
with images, latents (if your workflow saves them), and a manifest for the explorer.

### Requirements

- ComfyUI running.
- Workflow saved in **API format** (ComfyUI: Enable Dev Mode → Save (API Format)).
- Optional: a SaveLatent (or SaveLatentNumpy) node in the workflow to store latents for metrics.

### Basic usage

```bash
# Run experiments (batch ID is taken from the workflow’s WebUI node if you omit --batch-id)
python run_bending_experiments.py --workflow path/to/workflow_api.json

# Give the batch a name (use the same as the WebUI session_id for cache in the web UI)
python run_bending_experiments.py -w workflow_api.json --batch-id my_batch

# Quick test on a few layers
python run_bending_experiments.py -w workflow_api.json --batch-id test --layers-limit 5

# See what would run, without queueing
python run_bending_experiments.py -w workflow_api.json --dry-run
```

### Useful options

| Option | Description |
|--------|-------------|
| `--workflow`, `-w` | Path to workflow JSON (API format). |
| `--batch-id`, `-b` | Batch name; also used as folder name under `web_bend_demo_experiments/`. |
| `--layers-limit` | Only run on the first N distinct layers (good for testing). |
| `--bend-types` | Restrict bend types, e.g. `--bend-types rotate add_noise multiply`. |
| `--steps-ranges` | Timestep ranges, e.g. `'*' '0-20' '20-50'`. |
| `--skip-defaults` | Skip bend levels that are no-op. |
| `--no-include-default` | Do not run the unbent baseline (needed for metrics comparison). |
| `--queue-once` | Queue one prompt first so the model is registered, then run the batch. |
| `--export-to DIR` | After the run, export the batch to `DIR` in one step. |
| `--dry-run` | Print the plan only; no queueing. |

If you don’t set `--batch-id`, the script uses the `session_id` from the Interactive Bending WebUI node (or generates one). The dry run shows the batch ID that would be used.

**Web UI cache:** If the batch ID matches the WebUI node’s `session_id`, the web interface can use these results from cache. Enable `USE_EXPERIMENT_CACHE` in `web/js/config.js` if needed.

---

## export_experiments.py

Exports one batch into a **portable folder**: `data.json`, `images/`, optional `latents/`, and a copy of `explorer.html`. By default it also computes metrics (cosine, LPIPS, DINOv2, CLIP) against the unbent baseline and stores them in the data for the explorer.

```bash
# Export batch "my_batch" to ./my_export
python export_experiments.py --batch-id my_batch --output-dir ./my_export

# Limit to first N unique layers
python export_experiments.py -b my_batch -o ./my_export --layers-limit 5

# Export without computing metrics
python export_experiments.py -b my_batch -o ./my_export --no-metrics
```

**View the export without ComfyUI:**

```bash
python serve_explorer_standalone.py --export-dir ./my_export --port 8765
# Then open http://localhost:8765/
```

---

## serve_explorer_standalone.py

Serves the experiment explorer so you can browse and filter results **without running ComfyUI**. It can read from the experiments folder (ComfyUI output) or from an exported folder.

```bash
# From default experiments folder (ComfyUI output/web_bend_demo_experiments)
python serve_explorer_standalone.py --port 8765

# From a specific experiments folder
python serve_explorer_standalone.py --experiments-dir /path/to/output/web_bend_demo_experiments --port 8765

# From an exported folder (data.json + images/ + explorer.html)
python serve_explorer_standalone.py --export-dir ./my_export --port 8765
```

Then open **http://localhost:8765/** in your browser.

---

## build_offline_explorer.py

Builds a **single HTML file** that embeds the experiment data and loads images from a local `images/` folder using relative paths. You can open it directly in the browser (file://) with no server.

```bash
# From an export folder that has data.json and images/
python build_offline_explorer.py --dir ./my_export

# Custom output filename (default: explorer_offline.html)
python build_offline_explorer.py --dir ./my_export --output index.html
```

The HTML is written inside the export folder. Open it from that folder so the relative image paths work.

---

## compute_metrics.py

Computes distance/similarity metrics between the **unbent baseline** and each bent result (cosine on latents; LPIPS, DINOv2-base, CLIP on images). You can run it as part of export or on its own.

**Requirements:** `pip install lpips`. For CLIP: `pip install transformers`.

```bash
# On an export folder (updates data.json)
python compute_metrics.py --dir ./my_export

# On a batch folder (updates results.jsonl)
python compute_metrics.py --batch-dir /path/to/output/web_bend_demo_experiments/my_batch
```

The batch must include the default (unbent) run. The run script includes it by default unless you use `--no-include-default`.

---

## analyze_bending_impact.py

Loads experiment data (from an export or batch folder) and summarizes metrics **by group** (e.g. by block, bend type, timestep range). Outputs statistics (mean, std, min, max, count) per group and per metric.

Metrics must already be computed (e.g. by running `compute_metrics.py` or exporting with metrics).

```bash
# From export folder
python analyze_bending_impact.py --dir ./my_export

# From batch folder
python analyze_bending_impact.py --batch-dir /path/to/web_bend_demo_experiments/my_batch

# Write results to files
python analyze_bending_impact.py --dir ./my_export --output analysis.json --csv analysis.csv

# Choose which groups and metrics
python analyze_bending_impact.py --dir ./my_export --demarcations block_region bend_module_type --metrics lpips_distance clip_distance --csv report.csv

# Only write files, no console summary
python analyze_bending_impact.py --dir ./my_export -o analysis.json -q
```

**Groups (demarcations):** e.g. `block_region`, `block_name`, `container_type`, `layer_type`, `bend_module_type`, `steps_range`, `type_path`. **Metrics:** `cosine_distance`, `lpips_distance`, `dinov2_distance`, `clip_distance`.

---

## grid_from_selection/

Contains **run_grid_from_selection.py** and example inputs. It runs a single bending selection across several layer paths and prompts, then produces a **LaTeX figure grid** (e.g. rows = prompts, columns = layers).

```bash
cd grid_from_selection
python run_grid_from_selection.py -s bend.json -w /path/to/workflow_api.json --paths-file paths.txt --prompts-file prompts.txt -o ./grid_out
```

See **grid_from_selection/README.md** for details and file formats.
