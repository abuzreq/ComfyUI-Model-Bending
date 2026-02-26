#!/usr/bin/env python3
"""
Export an experiment batch to a portable folder: data.json + images/ + explorer.
Use without ComfyUI:
  python serve_explorer_standalone.py --export-dir /path/to/exported_folder --port 8765
  then open http://localhost:8765/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from run_bending_experiments import get_experiments_base_dir


def export_batch(
    batch_id: str,
    output_dir: str,
    layers_limit: int = 0,
    no_explorer: bool = False,
    compute_metrics: bool = True,
) -> Path:
    """Export a batch to a portable folder. Returns the output directory path."""
    exp_base = get_experiments_base_dir()
    safe_batch = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
    batch_dir = exp_base / safe_batch
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch not found: {batch_dir}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_out = out_dir / "images"
    images_out.mkdir(exist_ok=True)

    # Load results
    results_path = batch_dir / "results.jsonl"
    results = []
    if results_path.is_file():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if layers_limit > 0:
        order = []
        seen = set()
        for r in results:
            lp = r.get("layer_path", "")
            if lp not in seen:
                seen.add(lp)
                order.append(lp)
        allowed = set(order[: layers_limit])
        results = [r for r in results if r.get("layer_path", "") in allowed]

    manifest = {}
    manifest_path = batch_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            pass

    # Copy images and set relative image_url
    images_src = batch_dir / "images"
    for r in results:
        fn = r.get("image_filename", "")
        subfolder = r.get("image_subfolder", "")
        if subfolder == "images" and fn:
            src = batch_dir / "images" / fn
            if src.is_file():
                dst = images_out / fn
                shutil.copy2(src, dst)
                r["image_url"] = f"images/{fn}"
            else:
                r["image_url"] = ""
        else:
            # ComfyUI output reference - copy if we have the file
            r["image_url"] = ""
            if fn and images_src.is_dir():
                # Try lookup_key.png in batch images
                lk = r.get("lookup_key", "")
                if lk:
                    src = images_src / f"{lk}.png"
                    if src.is_file():
                        dst = images_out / f"{lk}.png"
                        shutil.copy2(src, dst)
                        r["image_url"] = f"images/{lk}.png"

    # Copy latents and set relative latent_url
    latents_src = batch_dir / "latents"
    latents_out = out_dir / "latents"
    if latents_src.is_dir():
        latents_out.mkdir(exist_ok=True)
        for r in results:
            fn = r.get("latent_filename", "")
            subfolder = r.get("latent_subfolder", "")
            if subfolder == "latents" and fn:
                src = latents_src / fn
                if src.is_file():
                    dst = latents_out / fn
                    shutil.copy2(src, dst)
                    r["latent_url"] = f"latents/{fn}"
                else:
                    r["latent_url"] = ""
            else:
                r["latent_url"] = ""

    data = {"ok": True, "manifest": manifest, "results": results}
    data_path = out_dir / "data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(results)} results to {out_dir}")
    if compute_metrics:
        try:
            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from compute_metrics import run_on_export_dir
            run_on_export_dir(out_dir)
        except ImportError as e:
            print(f"  Skipping metrics (pip install lpips): {e}")
        except Exception as e:
            print(f"  Metrics failed: {e}")
    latent_count = len(list(latents_out.iterdir())) if latents_out.exists() else 0
    print(f"  data.json, images/ ({len(list(images_out.iterdir()))} files), latents/ ({latent_count} files)")

    if not no_explorer:
        explorer_src = Path(__file__).resolve().parent.parent / "web" / "explorer.html"
        if explorer_src.is_file():
            shutil.copy2(explorer_src, out_dir / "explorer.html")
            print(f"  explorer.html")
        else:
            _write_minimal_explorer(out_dir)
            print(f"  explorer.html (generated)")
    print("To view without ComfyUI: run from scripts folder:")
    print("  python serve_explorer_standalone.py --export-dir " + str(out_dir.resolve()) + " --port 8765")
    print("  Open http://localhost:8765/")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Export experiment batch to portable folder")
    ap.add_argument("--batch-id", "-b", required=True, help="Batch ID to export")
    ap.add_argument("--output-dir", "-o", required=True, help="Output directory (created if needed)")
    ap.add_argument("--layers-limit", type=int, default=0, help="Max unique layers to export (0=all)")
    ap.add_argument("--no-explorer", action="store_true", help="Do not copy standalone explorer HTML")
    ap.add_argument("--no-metrics", action="store_true", help="Do not compute cosine/LPIPS metrics vs default")
    args = ap.parse_args()
    try:
        export_batch(
            args.batch_id,
            args.output_dir,
            layers_limit=args.layers_limit,
            no_explorer=args.no_explorer,
            compute_metrics=not args.no_metrics,
        )
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


def _write_minimal_explorer(out_dir: Path) -> None:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Bending Experiments Explorer</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #1a1a1a; color: #e0e0e0; }
    h1 { margin: 0 0 16px; font-size: 1.4rem; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; padding: 12px; background: #2a2a2a; border-radius: 8px; }
    .toolbar label { display: flex; align-items: center; gap: 6px; }
    .toolbar select, .toolbar input[type="text"] { padding: 6px 10px; border-radius: 4px; background: #333; color: #fff; border: 1px solid #555; min-width: 120px; }
    .stats { font-size: 0.9rem; color: #aaa; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
    .card { background: #2a2a2a; border-radius: 8px; overflow: hidden; border: 1px solid #444; }
    .card img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; cursor: pointer; }
    .card-body { padding: 10px; font-size: 0.8rem; line-height: 1.4; }
    .card-tags { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
    .tag { background: #444; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.9); display: none; align-items: center; justify-content: center; z-index: 9999; cursor: pointer; }
    .lightbox.visible { display: flex; }
    .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; }
    .lightbox-caption { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; max-width: 90%; text-align: center; }
    .error, .loading { padding: 24px; text-align: center; color: #888; }
    .group-header { font-size: 0.95rem; font-weight: 600; margin: 20px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #444; color: #90caf9; }
    .metrics { font-size: 0.75rem; color: #81c784; }
  </style>
</head>
<body>
  <h1>Bending Experiments Explorer</h1>
  <div class="toolbar">
    <label>Filter region: <select id="filter-region"><option value="">All</option><option value="input">Input</option><option value="middle">Middle</option><option value="output">Output</option></select></label>
    <label>Filter container: <select id="filter-container"><option value="">All</option></select></label>
    <label>Filter bend: <select id="filter-bend"><option value="">All</option><option value="rotate">Rotate</option><option value="add_noise">Add Noise</option><option value="multiply">Multiply</option></select></label>
    <label>Search path: <input type="text" id="search-path" placeholder="e.g. input_blocks" /></label>
    <label>Group by: <select id="group-by"><option value="">None</option><option value="block_region">Block region</option><option value="container_type">Container</option><option value="bend_module_type">Bend type</option><option value="layer_path">Layer path</option></select></label>
    <span id="flat-sort-wrap">
      <label>Sort by: <select id="sort-by"><option value="layer_path">Layer path</option><option value="block_region">Block region</option><option value="container_type">Container</option><option value="bend_module_type">Bend type</option><option value="bend_arg">Bend arg value</option><option value="cosine_distance">Cosine dist</option><option value="lpips_distance">LPIPS dist</option><option value="dinov2_distance">DINOv2 dist</option><option value="clip_distance">CLIP dist</option></select></label>
      <label>Order: <select id="sort-order"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
    </span>
    <span id="grouped-sort-wrap" style="display:none;">
      <label>Group sort: <select id="group-sort-agg"><option value="sum">Sum</option><option value="max">Max</option><option value="std">Std</option></select> of <select id="group-sort-metric"><option value="cosine_distance">Cosine dist</option><option value="lpips_distance">LPIPS dist</option><option value="dinov2_distance">DINOv2 dist</option><option value="clip_distance">CLIP dist</option></select></label>
      <label>Group order: <select id="group-sort-order"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
      <label>Within group: <select id="within-sort-by"><option value="layer_path">Layer path</option><option value="block_region">Block region</option><option value="container_type">Container</option><option value="bend_module_type">Bend type</option><option value="bend_arg">Bend arg value</option><option value="cosine_distance">Cosine dist</option><option value="lpips_distance">LPIPS dist</option><option value="dinov2_distance">DINOv2 dist</option><option value="clip_distance">CLIP dist</option></select></label>
      <label>Within order: <select id="within-sort-order"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
    </span>
    <span class="stats" id="stats">0 results</span>
  </div>
  <div id="content"><div class="loading">Loading data.json...</div></div>
  <div class="lightbox" id="lightbox"><img src="" alt="" /><div class="lightbox-caption" id="lightbox-caption"></div></div>
  <script>
    let allResults = [];
    function debounce(fn, ms) { let t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }
    function naturalCompare(a, b) {
      const sa = String(a ?? ""), sb = String(b ?? ""), re = /(\\d+)|(\\D+)/g;
      let ma, mb;
      for (;;) { ma = re.exec(sa); mb = re.exec(sb); if (!ma && !mb) return 0; if (!ma) return -1; if (!mb) return 1;
        if (ma[1] !== undefined && mb[1] !== undefined) { const na = parseInt(ma[1], 10), nb = parseInt(mb[1], 10); if (na !== nb) return na - nb; }
        else { const c = ma[0].localeCompare(mb[0]); if (c !== 0) return c; }
      }
    }
    const BEND_ARG_KEYS = { multiply: "scalar", rotate: "angle_degrees", add_noise: "noise_std" };
    function getBendArgValue(r) {
      const key = BEND_ARG_KEYS[r.bend_module_type];
      if (!key || !r.module_args) return null;
      const v = r.module_args[key];
      return v != null ? v : null;
    }
    function compareItems(a, b, sortBy, dir) {
      let cmp = 0;
      if (sortBy === "bend_arg") {
        const va = getBendArgValue(a), vb = getBendArgValue(b);
        const na = va != null ? Number(va) : NaN, nb = vb != null ? Number(vb) : NaN;
        if (!Number.isNaN(na) && !Number.isNaN(nb)) cmp = na - nb;
        else cmp = naturalCompare(va, vb);
      } else if (sortBy === "cosine_distance" || sortBy === "lpips_distance" || sortBy === "dinov2_distance" || sortBy === "clip_distance") {
        const va = a[sortBy] != null ? Number(a[sortBy]) : Infinity, vb = b[sortBy] != null ? Number(b[sortBy]) : Infinity;
        cmp = va - vb;
      } else cmp = naturalCompare(a[sortBy], b[sortBy]);
      return dir * cmp;
    }
    function groupAggregate(items, metric, agg) {
      const vals = items.map(r => r[metric]).filter(v => v != null && !Number.isNaN(Number(v))).map(Number);
      if (vals.length === 0) return NaN;
      if (agg === "sum") return vals.reduce((a, b) => a + b, 0);
      if (agg === "max") return Math.max(...vals);
      if (agg === "std") {
        if (vals.length < 2) return 0;
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const variance = vals.reduce((s, x) => s + (x - mean) ** 2, 0) / vals.length;
        return Math.sqrt(variance);
      }
      return NaN;
    }
    async function load() {
      try {
        const r = await fetch("data.json");
        if (!r.ok) throw new Error("data.json not found. Run this from an exported folder.");
        const data = await r.json();
        allResults = data.results || [];
        const containers = [...new Set(allResults.map(x => x.container_type).filter(Boolean))].sort((a, b) => naturalCompare(a, b));
        document.getElementById("filter-container").innerHTML = "<option value=\\"\\">All</option>" + containers.map(c => "<option value=\"" + c + "\">" + c + "</option>").join("");
        render();
      } catch (e) {
        document.getElementById("content").innerHTML = "<div class=\"error\">" + e.message + "</div>";
      }
    }
    function filterAndSort() {
      const region = document.getElementById("filter-region").value;
      const container = document.getElementById("filter-container").value;
      const bend = document.getElementById("filter-bend").value;
      const search = (document.getElementById("search-path").value || "").toLowerCase();
      let list = allResults.filter(r => {
        if (r.is_default) return false;
        if (region && r.block_region !== region) return false;
        if (container && r.container_type !== container) return false;
        if (bend && r.bend_module_type !== bend) return false;
        if (search && !(r.layer_path || "").toLowerCase().includes(search)) return false;
        return true;
      });
      const sortBy = document.getElementById("sort-by").value;
      const order = document.getElementById("sort-order").value;
      const dir = order === "desc" ? -1 : 1;
      list.sort((a, b) => {
        let cmp = 0;
        if (sortBy === "bend_arg") {
          const va = getBendArgValue(a), vb = getBendArgValue(b);
          const na = va != null ? Number(va) : NaN, nb = vb != null ? Number(vb) : NaN;
          if (!Number.isNaN(na) && !Number.isNaN(nb)) cmp = na - nb;
          else cmp = naturalCompare(va, vb);
        } else if (sortBy === "cosine_distance" || sortBy === "lpips_distance" || sortBy === "dinov2_distance" || sortBy === "clip_distance") {
          const va = a[sortBy] != null ? Number(a[sortBy]) : Infinity;
          const vb = b[sortBy] != null ? Number(b[sortBy]) : Infinity;
          cmp = va - vb;
        } else cmp = naturalCompare(a[sortBy], b[sortBy]);
        return dir * cmp;
      });
      return list;
    }
    function render() {
      const list = filterAndSort();
      const groupBy = document.getElementById("group-by").value;
      const flatWrap = document.getElementById("flat-sort-wrap");
      const groupedWrap = document.getElementById("grouped-sort-wrap");
      if (groupBy) { flatWrap.style.display = "none"; groupedWrap.style.display = ""; } else { flatWrap.style.display = ""; groupedWrap.style.display = "none"; }
      document.getElementById("stats").textContent = list.length + " results";
      const content = document.getElementById("content");
      if (list.length === 0) { content.innerHTML = "<div class=\"loading\">No results match filters.</div>"; return; }
      const card = (r) => {
        const args = r.module_args ? JSON.stringify(r.module_args) : "";
        const steps = (r.steps_min != null || r.steps_max != null) ? " steps " + (r.steps_min ?? "?") + "-" + (r.steps_max ?? "?") : "";
        const cos = r.cosine_distance != null ? "cos \u0394 " + r.cosine_distance.toFixed(4) : "";
        const lpips = r.lpips_distance != null ? "LPIPS " + r.lpips_distance.toFixed(4) : "";
        const dinov2 = r.dinov2_distance != null ? "DINOv2 " + r.dinov2_distance.toFixed(4) : "";
        const clip = r.clip_distance != null ? "CLIP " + r.clip_distance.toFixed(4) : "";
        const metrics = [cos, lpips, dinov2, clip].filter(Boolean).join(" · ");
        return "<div class=\"card\"><img src=\"" + (r.image_url || "") + "\" alt=\"\" loading=\"lazy\" data-url=\"" + (r.image_url || "") + "\" data-caption=\"" + (r.layer_path || "") + " | " + (r.bend_module_type || "") + " " + args + steps + "\" /><div class=\"card-body\"><strong>" + (r.layer_path || "").split(".").pop() + "</strong><br/>" + (r.block_region || "") + " · " + (r.container_type || "") + "<br/>" + (r.bend_module_type || "") + " " + args + (steps ? "<br/>" + steps : "") + (metrics ? "<br/><span class=\"metrics\">" + metrics + "</span>" : "") + "<div class=\"card-tags\"><span class=\"tag\">" + (r.block_region || "-") + "</span><span class=\"tag\">" + (r.container_type || "-") + "</span><span class=\"tag\">" + (r.bend_module_type || "-") + "</span></div></div></div>";
      };
      let html = "";
      if (groupBy) {
        const groups = {};
        list.forEach(r => { const k = r[groupBy] ?? "(empty)"; if (!groups[k]) groups[k] = []; groups[k].push(r); });
        const agg = document.getElementById("group-sort-agg").value;
        const metric = document.getElementById("group-sort-metric").value;
        const groupOrder = document.getElementById("group-sort-order").value;
        const groupDir = groupOrder === "desc" ? -1 : 1;
        const withinBy = document.getElementById("within-sort-by").value;
        const withinOrder = document.getElementById("within-sort-order").value;
        const withinDir = withinOrder === "desc" ? -1 : 1;
        const sortedKeys = Object.keys(groups).sort((a, b) => {
          const va = groupAggregate(groups[a], metric, agg);
          const vb = groupAggregate(groups[b], metric, agg);
          const na = Number.isNaN(va) ? -Infinity : va;
          const nb = Number.isNaN(vb) ? -Infinity : vb;
          if (na !== nb) return groupDir * (na - nb);
          return naturalCompare(a, b);
        });
        sortedKeys.forEach(k => { groups[k].sort((a, b) => compareItems(a, b, withinBy, withinDir)); });
        sortedKeys.forEach(k => {
          html += "<div class=\"group-header\">" + groupBy + ": " + k + "</div><div class=\"grid\">";
          groups[k].forEach(r => { html += card(r); });
          html += "</div>";
        });
      } else html = "<div class=\"grid\">" + list.map(card).join("") + "</div>";
      content.innerHTML = html;
      content.querySelectorAll(".card img").forEach(img => {
        img.onclick = () => { document.getElementById("lightbox").querySelector("img").src = img.dataset.url || ""; document.getElementById("lightbox-caption").textContent = img.dataset.caption || ""; document.getElementById("lightbox").classList.add("visible"); };
      });
    }
    document.getElementById("lightbox").onclick = () => document.getElementById("lightbox").classList.remove("visible");
    ["filter-region", "filter-container", "filter-bend", "group-by", "sort-by", "sort-order", "group-sort-agg", "group-sort-metric", "group-sort-order", "within-sort-by", "within-sort-order"].forEach(id => document.getElementById(id).onchange = render);
    document.getElementById("search-path").oninput = debounce(render, 200);
    load();
  </script>
</body>
</html>
"""
    (out_dir / "explorer.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
