#!/usr/bin/env python3
"""
Build a single HTML page that embeds experiment data and uses relative image paths.
Open the generated file directly (file://) for faster loading: no HTTP, no fetch(),
images load from the local images/ folder.

Usage:
  python build_offline_explorer.py --dir /path/to/export_folder
  python build_offline_explorer.py --dir ./my_export --output index.html

Output: writes explorer_offline.html (or --output) into the export folder.
Requires: export folder with data.json and images/ (e.g. from export_experiments.py).
"""

import argparse
import json
import sys
from pathlib import Path


def ensure_relative_image_url(r: dict, images_dir: Path) -> None:
    """Set result image_url to relative path images/filename if the file exists."""
    url = r.get("image_url") or ""
    fn = r.get("image_filename") or r.get("lookup_key", "")
    if not fn and url:
        # Extract filename from path like images/foo.png or http://.../images/foo.png
        parts = url.replace("\\", "/").split("/")
        fn = parts[-1] if parts else ""
    if not fn:
        return
    if not fn.endswith(".png"):
        fn = fn + ".png"
    rel = f"images/{fn}"
    if (images_dir / fn).is_file():
        r["image_url"] = rel
    elif url and not url.startswith(("http", "//", "file:")):
        r["image_url"] = url if "/" in url else rel


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build offline explorer HTML with embedded data and relative image paths"
    )
    ap.add_argument("--dir", required=True, help="Export folder containing data.json and images/")
    ap.add_argument("--output", default="explorer_offline.html", help="Output HTML filename (default: explorer_offline.html)")
    args = ap.parse_args()

    export_dir = Path(args.dir).resolve()
    data_path = export_dir / "data.json"
    images_dir = export_dir / "images"

    if not data_path.is_file():
        print(f"Error: {data_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Warning: {images_dir} not found; image paths may break.", file=sys.stderr)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    for r in results:
        ensure_relative_image_url(r, images_dir)

    # Embed as JSON in a script tag (type=application/json avoids </script> issues)
    data_json = json.dumps(data, ensure_ascii=False)
    # Break any </script> inside JSON strings so it doesn't close the tag
    data_json = data_json.replace("</script>", "<\\/script>")

    html = _build_html(data_json)
    out_path = export_dir / args.output
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(results)} results, data inlined).")
    print("Open the file directly (file://) for fastest loading; images load from ./images/.")


def _build_html(data_json: str) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Bending Experiments Explorer (Offline)</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #1a1a1a; color: #e0e0e0; }
    h1 { margin: 0 0 16px; font-size: 1.4rem; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; padding: 12px; background: #2a2a2a; border-radius: 8px; }
    .toolbar label { display: flex; align-items: center; gap: 6px; }
    .toolbar select, .toolbar input[type="text"] { padding: 6px 10px; border-radius: 4px; background: #333; color: #fff; border: 1px solid #555; min-width: 140px; }
    .stats { font-size: 0.9rem; color: #aaa; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
    .content-scroll { overflow: auto; max-height: calc(100vh - 160px); position: relative; }
    .virtual-sizer { pointer-events: none; }
    .virtual-viewport { position: absolute; left: 0; right: 0; top: 0; display: grid; gap: 12px; grid-template-columns: repeat(var(--cols, 4), 1fr); grid-auto-rows: 268px; }
    .card { background: #2a2a2a; border-radius: 8px; overflow: hidden; border: 1px solid #444; contain: layout style paint; min-height: 260px; }
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
  <h1>Bending Experiments Explorer (Offline)</h1>
  <div class="toolbar">
    <label>Filter region: <select id="filter-region"><option value="">All</option><option value="input">Input</option><option value="middle">Middle</option><option value="output">Output</option></select></label>
    <label>Filter container: <select id="filter-container"><option value="">All</option></select></label>
    <label>Filter bend: <select id="filter-bend"><option value="">All</option><option value="rotate">Rotate</option><option value="add_noise">Add Noise</option><option value="multiply">Multiply</option></select></label>
    <label>Filter arg: <select id="filter-module-arg"><option value="">All</option></select></label>
    <label>Filter steps: <select id="filter-steps"><option value="">All</option></select></label>
    <label>Search path: <input type="text" id="search-path" placeholder="e.g. input_blocks" /></label>
    <label>Group by: <select id="group-by"><option value="">None</option><option value="block_region">Block region</option><option value="block_name">Block name</option><option value="container_type">Container</option><option value="bend_module_type">Bend type</option><option value="steps_range">Steps range</option><option value="layer_path">Layer path</option></select></label>
    <span id="flat-sort-wrap">
      <label>Sort by: <select id="sort-by"><option value="layer_path">Layer path</option><option value="block_region">Block region</option><option value="block_name">Block name</option><option value="container_type">Container</option><option value="bend_module_type">Bend type</option><option value="bend_arg">Bend arg value</option><option value="steps_range">Steps range</option><option value="cosine_distance">Cosine dist</option><option value="lpips_distance">LPIPS dist</option><option value="dinov2_distance">DINOv2 dist</option><option value="clip_distance">CLIP dist</option></select></label>
      <label>Order: <select id="sort-order"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
    </span>
    <span id="grouped-sort-wrap" style="display:none;">
      <label>Group sort: <select id="group-sort-agg"><option value="sum">Sum</option><option value="max">Max</option><option value="std">Std</option></select> of <select id="group-sort-metric"><option value="cosine_distance">Cosine dist</option><option value="lpips_distance">LPIPS dist</option><option value="dinov2_distance">DINOv2 dist</option><option value="clip_distance">CLIP dist</option></select></label>
      <label>Group order: <select id="group-sort-order"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
      <label>Within group: <select id="within-sort-by"><option value="layer_path">Layer path</option><option value="block_region">Block region</option><option value="block_name">Block name</option><option value="container_type">Container</option><option value="bend_module_type">Bend type</option><option value="bend_arg">Bend arg value</option><option value="steps_range">Steps range</option><option value="cosine_distance">Cosine dist</option><option value="lpips_distance">LPIPS dist</option><option value="dinov2_distance">DINOv2 dist</option><option value="clip_distance">CLIP dist</option></select></label>
      <label>Within order: <select id="within-sort-order"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
    </span>
    <span class="stats" id="stats">0 results</span>
  </div>
  <div id="content"><div class="loading">Loading...</div></div>
  <div class="lightbox" id="lightbox"><img src="" alt="" /><div class="lightbox-caption" id="lightbox-caption"></div></div>
  <script type="application/json" id="experiment-data">""" + data_json + """</script>
  <script>
(function () {
  const dataEl = document.getElementById("experiment-data");
  const data = JSON.parse(dataEl.textContent);
  let allResults = data.results || [];
  let currentList = [];
  let currentCols = 4;
  let scrollRaf = null;
  const BEND_ARG_KEYS = { multiply: "scalar", rotate: "angle_degrees", add_noise: "noise_std" };
  const VIRTUAL_THRESHOLD = 80;
  const GROUPED_MAX_ITEMS = 500;
  const ROW_HEIGHT = 280;
  const BUFFER_ROWS = 2;

  function debounce(fn, ms) { let t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }
  function getColumns(el) { const w = el.clientWidth || el.offsetWidth || 800; return Math.max(1, Math.floor((w + 12) / 212)); }
  function escapeHtml(s) { if (s == null) return ""; return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function escapeAttr(s) { if (s == null) return ""; return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function naturalCompare(a, b) {
    var sa = String(a != null ? a : ""), sb = String(b != null ? b : ""), re = /(\\d+)|(\\D+)/g, ma, mb;
    for (;;) { ma = re.exec(sa); mb = re.exec(sb); if (!ma && !mb) return 0; if (!ma) return -1; if (!mb) return 1;
      if (ma[1] !== undefined && mb[1] !== undefined) { var na = parseInt(ma[1], 10), nb = parseInt(mb[1], 10); if (na !== nb) return na - nb; }
      else { var c = ma[0].localeCompare(mb[0]); if (c !== 0) return c; }
    }
  }
  function getBendArgValue(r) { var key = BEND_ARG_KEYS[r.bend_module_type]; if (!key || !r.module_args) return null; var v = r.module_args[key]; return v != null ? v : null; }
  function compareItems(a, b, sortBy, dir) {
    var cmp = 0;
    if (sortBy === "steps_range") { var sa = (a.steps_min ?? -1) * 10000 + (a.steps_max ?? -1), sb = (b.steps_min ?? -1) * 10000 + (b.steps_max ?? -1); cmp = sa - sb; }
    else if (sortBy === "bend_arg") { var va = getBendArgValue(a), vb = getBendArgValue(b); var na = va != null ? Number(va) : NaN, nb = vb != null ? Number(vb) : NaN; if (!Number.isNaN(na) && !Number.isNaN(nb)) cmp = na - nb; else cmp = naturalCompare(va, vb); }
    else if (sortBy === "cosine_distance" || sortBy === "lpips_distance" || sortBy === "dinov2_distance" || sortBy === "clip_distance") { var va = a[sortBy] != null ? Number(a[sortBy]) : Infinity, vb = b[sortBy] != null ? Number(b[sortBy]) : Infinity; cmp = va - vb; }
    else cmp = naturalCompare(a[sortBy], b[sortBy]);
    return dir * cmp;
  }
  function groupAggregate(items, metric, agg) {
    var vals = items.map(function(r) { return r[metric]; }).filter(function(v) { return v != null && !Number.isNaN(Number(v)); }).map(Number);
    if (vals.length === 0) return NaN;
    if (agg === "sum") return vals.reduce(function(a, b) { return a + b; }, 0);
    if (agg === "max") return Math.max.apply(null, vals);
    if (agg === "std") { if (vals.length < 2) return 0; var mean = vals.reduce(function(a, b) { return a + b; }, 0) / vals.length; var variance = vals.reduce(function(s, x) { return s + (x - mean) * (x - mean); }, 0) / vals.length; return Math.sqrt(variance); }
    return NaN;
  }

  function updateFilterModuleArg() {
    const bend = document.getElementById("filter-bend").value;
    const sel = document.getElementById("filter-module-arg");
    let opts = '<option value="">All</option>';
    if (!bend) { sel.innerHTML = opts; return; }
    const key = BEND_ARG_KEYS[bend];
    if (!key) { sel.innerHTML = opts; return; }
    const filtered = allResults.filter(function(r) { return r.bend_module_type === bend; });
    const vals = filtered.map(function(r) { return (r.module_args && r.module_args[key]) != null ? String(r.module_args[key]) : ""; }).filter(Boolean);
    vals.sort(function(a, b) { var na = parseFloat(a), nb = parseFloat(b); if (!isNaN(na) && !isNaN(nb)) return na - nb; return String(a).localeCompare(String(b)); });
    var uniq = []; var seen = {}; vals.forEach(function(v) { if (!seen[v]) { seen[v] = true; uniq.push(v); } });
    opts += uniq.map(function(v) { return '<option value="' + escapeAttr(v) + '">' + escapeHtml(v) + '</option>'; }).join("");
    sel.innerHTML = opts;
  }
  function updateFilterSteps() {
    const sel = document.getElementById("filter-steps");
    var pairs = {};
    allResults.forEach(function(r) {
      var sm = r.steps_min != null ? r.steps_min : "?", sx = r.steps_max != null ? r.steps_max : "?";
      if (sm !== "?" || sx !== "?") pairs[sm + "-" + sx] = true;
    });
    var sorted = Object.keys(pairs).sort(function(a, b) {
      var a1 = a.split("-").map(function(x) { return x === "?" ? -1 : parseInt(x, 10); });
      var b1 = b.split("-").map(function(x) { return x === "?" ? -1 : parseInt(x, 10); });
      if (a1[0] !== b1[0]) return a1[0] - b1[0]; return a1[1] - b1[1];
    });
    sel.innerHTML = '<option value="">All</option>' + sorted.map(function(p) { return '<option value="' + escapeAttr(p) + '">' + escapeHtml(p) + '</option>'; }).join("");
  }

  function buildCard(r) {
    var args = r.module_args ? JSON.stringify(r.module_args) : "";
    var steps = (r.steps_min != null || r.steps_max != null) ? " steps " + (r.steps_min ?? "?") + "-" + (r.steps_max ?? "?") : "";
    var url = r.image_url || "";
    var cap = escapeAttr((r.layer_path || "") + " | " + (r.bend_module_type || "") + " " + args + steps);
    var cos = r.cosine_distance != null ? "cos \\u0394 " + r.cosine_distance.toFixed(4) : "";
    var lpips = r.lpips_distance != null ? "LPIPS " + r.lpips_distance.toFixed(4) : "";
    var dinov2 = r.dinov2_distance != null ? "DINOv2 " + r.dinov2_distance.toFixed(4) : "";
    var clip = r.clip_distance != null ? "CLIP " + r.clip_distance.toFixed(4) : "";
    var metrics = [cos, lpips, dinov2, clip].filter(Boolean).join(" \\u00b7 ");
    var layerShort = escapeHtml((r.layer_path || "").split(".").pop());
    var blockRegion = escapeHtml(r.block_region || ""), containerType = escapeHtml(r.container_type || ""), bendType = escapeHtml(r.bend_module_type || "");
    return '<div class="card"><img src="' + escapeAttr(url) + '" alt="" loading="lazy" data-url="' + escapeAttr(url) + '" data-caption="' + cap + '" /><div class="card-body"><strong>' + layerShort + '</strong><br/>' + blockRegion + ' \u00b7 ' + containerType + '<br/>' + bendType + ' ' + escapeHtml(args) + (steps ? '<br/>' + escapeHtml(steps) : '') + (metrics ? '<br/><span class="metrics">' + escapeHtml(metrics) + '</span>' : '') + '<div class="card-tags"><span class="tag">' + (blockRegion || '-') + '</span><span class="tag">' + (containerType || '-') + '</span><span class="tag">' + (bendType || '-') + '</span></div></div></div>';
  }

  function filterAndSort() {
    var region = document.getElementById("filter-region").value, container = document.getElementById("filter-container").value, bend = document.getElementById("filter-bend").value;
    var moduleArg = document.getElementById("filter-module-arg").value, stepsRange = document.getElementById("filter-steps").value;
    var search = (document.getElementById("search-path").value || "").toLowerCase();
    var list = allResults.filter(function(r) {
      if (r.is_default) return false;
      if (region && r.block_region !== region) return false;
      if (container && r.container_type !== container) return false;
      if (bend && r.bend_module_type !== bend) return false;
      if (moduleArg) { var key = BEND_ARG_KEYS[bend] || "scalar"; var v = r.module_args && r.module_args[key]; if (String(v) !== moduleArg) return false; }
      if (stepsRange) { var sm = r.steps_min != null ? r.steps_min : "?", sx = r.steps_max != null ? r.steps_max : "?"; if (sm + "-" + sx !== stepsRange) return false; }
      if (search && !(r.layer_path || "").toLowerCase().includes(search)) return false;
      return true;
    });
    var sortBy = document.getElementById("sort-by").value, order = document.getElementById("sort-order").value, dir = order === "desc" ? -1 : 1;
    list.sort(function(a, b) {
      var cmp = 0;
      if (sortBy === "steps_range") { var sa = (a.steps_min ?? -1) * 10000 + (a.steps_max ?? -1), sb = (b.steps_min ?? -1) * 10000 + (b.steps_max ?? -1); cmp = sa - sb; }
      else if (sortBy === "bend_arg") { var va = getBendArgValue(a), vb = getBendArgValue(b); var na = va != null ? Number(va) : NaN, nb = vb != null ? Number(vb) : NaN; if (!Number.isNaN(na) && !Number.isNaN(nb)) cmp = na - nb; else cmp = naturalCompare(va, vb); }
      else if (sortBy === "cosine_distance" || sortBy === "lpips_distance" || sortBy === "dinov2_distance" || sortBy === "clip_distance") {
        var va = a[sortBy] != null ? Number(a[sortBy]) : Infinity, vb = b[sortBy] != null ? Number(b[sortBy]) : Infinity; cmp = va - vb;
      } else cmp = naturalCompare(a[sortBy], b[sortBy]);
      return dir * cmp;
    });
    return list;
  }

  function updateVirtualView() {
    var content = document.getElementById("content"), viewport = document.getElementById("virtual-viewport"), sizer = document.getElementById("virtual-sizer");
    if (!viewport || !sizer || !content || currentList.length === 0) return;
    var cols = currentCols, numRows = Math.ceil(currentList.length / cols);
    var scrollTop = content.scrollTop, clientHeight = content.clientHeight;
    var startRow = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER_ROWS), endRow = Math.min(numRows, Math.ceil((scrollTop + clientHeight) / ROW_HEIGHT) + BUFFER_ROWS);
    var startIdx = startRow * cols, endIdx = Math.min(currentList.length, endRow * cols), slice = currentList.slice(startIdx, endIdx);
    viewport.style.top = (startRow * ROW_HEIGHT) + "px";
    viewport.style.setProperty("--cols", cols);
    viewport.innerHTML = slice.map(buildCard).join("");
    viewport.querySelectorAll(".card img").forEach(function(img) {
      img.onclick = function() { document.getElementById("lightbox").querySelector("img").src = img.dataset.url || ""; document.getElementById("lightbox-caption").textContent = img.dataset.caption || ""; document.getElementById("lightbox").classList.add("visible"); };
    });
  }
  function onScrollVirtual() {
    if (scrollRaf != null) return;
    scrollRaf = requestAnimationFrame(function() { scrollRaf = null; updateVirtualView(); });
  }
  function attachLightboxToCards(container) {
    if (!container) return;
    container.querySelectorAll(".card img").forEach(function(img) {
      img.onclick = function() { document.getElementById("lightbox").querySelector("img").src = img.dataset.url || ""; document.getElementById("lightbox-caption").textContent = img.dataset.caption || ""; document.getElementById("lightbox").classList.add("visible"); };
    });
  }

  function render() {
    var list = filterAndSort(), groupBy = document.getElementById("group-by").value;
    var flatWrap = document.getElementById("flat-sort-wrap"), groupedWrap = document.getElementById("grouped-sort-wrap");
    if (groupBy) { flatWrap.style.display = "none"; groupedWrap.style.display = ""; } else { flatWrap.style.display = ""; groupedWrap.style.display = "none"; }
    document.getElementById("stats").textContent = list.length + " results";
    var content = document.getElementById("content");
    content.className = "";
    content.onscroll = null;
    if (list.length === 0) { content.innerHTML = '<div class="loading">No results match filters.</div>'; return; }
    if (groupBy) {
      var groups = {};
      list.forEach(function(r) {
        var k = groupBy === "steps_range" ? ((r.steps_min != null || r.steps_max != null) && (r.steps_min !== "?" || r.steps_max !== "?") ? (r.steps_min ?? "?") + "-" + (r.steps_max ?? "?") : "(not set)") : (r[groupBy] ?? "(empty)");
        if (!groups[k]) groups[k] = []; groups[k].push(r);
      });
      var agg = document.getElementById("group-sort-agg").value, metric = document.getElementById("group-sort-metric").value;
      var groupOrder = document.getElementById("group-sort-order").value, groupDir = groupOrder === "desc" ? -1 : 1;
      var withinBy = document.getElementById("within-sort-by").value, withinOrder = document.getElementById("within-sort-order").value, withinDir = withinOrder === "desc" ? -1 : 1;
      var sortedKeys = Object.keys(groups).sort(function(a, b) {
        var va = groupAggregate(groups[a], metric, agg), vb = groupAggregate(groups[b], metric, agg);
        var na = Number.isNaN(va) ? -Infinity : va, nb = Number.isNaN(vb) ? -Infinity : vb;
        if (na !== nb) return groupDir * (na - nb);
        if (groupBy === "steps_range" && a !== "(not set)" && b !== "(not set)") { var a1 = a.split("-").map(function(x) { return x === "?" ? -1 : parseInt(x, 10); }), b1 = b.split("-").map(function(x) { return x === "?" ? -1 : parseInt(x, 10); }); if (a1[0] !== b1[0]) return a1[0] - b1[0]; return a1[1] - b1[1]; }
        return naturalCompare(a, b);
      });
      sortedKeys.forEach(function(k) { groups[k].sort(function(a, b) { return compareItems(a, b, withinBy, withinDir); }); });
      var totalRendered = 0, cap = list.length > GROUPED_MAX_ITEMS ? GROUPED_MAX_ITEMS : list.length, html = "";
      for (var i = 0; i < sortedKeys.length; i++) {
        var k = sortedKeys[i], groupItems = groups[k], remaining = cap - totalRendered;
        if (remaining <= 0) break;
        var toShow = Math.min(groupItems.length, remaining);
        html += '<div class="group-header">' + escapeHtml(groupBy) + ': ' + escapeHtml(k) + '</div><div class="grid">';
        for (var j = 0; j < toShow; j++) html += buildCard(groupItems[j]);
        html += "</div>";
        totalRendered += toShow;
      }
      if (list.length > GROUPED_MAX_ITEMS) html = '<div class="group-header" style="color:#ffb74d">Showing first ' + GROUPED_MAX_ITEMS + ' of ' + list.length + ' results. Clear Group by or narrow filters.</div>' + html;
      content.innerHTML = html;
      attachLightboxToCards(content);
      return;
    }
    if (list.length <= VIRTUAL_THRESHOLD) {
      content.innerHTML = '<div class="grid">' + list.map(buildCard).join("") + '</div>';
      attachLightboxToCards(content);
      return;
    }
    currentList = list;
    content.className = "content-scroll";
    currentCols = getColumns(content);
    var numRows = Math.ceil(list.length / currentCols);
    content.innerHTML = '<div class="virtual-sizer" id="virtual-sizer"></div><div class="virtual-viewport" id="virtual-viewport"></div>';
    var sizer = document.getElementById("virtual-sizer"), viewport = document.getElementById("virtual-viewport");
    sizer.style.height = (numRows * ROW_HEIGHT) + "px";
    viewport.style.setProperty("--cols", currentCols);
    viewport.style.gridAutoRows = ROW_HEIGHT + "px";
    content.onscroll = onScrollVirtual;
    updateVirtualView();
  }
  function onResize() {
    var content = document.getElementById("content");
    if (!content.classList.contains("content-scroll") || currentList.length === 0) return;
    var sizer = document.getElementById("virtual-sizer"), viewport = document.getElementById("virtual-viewport");
    if (!sizer || !viewport) return;
    currentCols = getColumns(content);
    var numRows = Math.ceil(currentList.length / currentCols);
    sizer.style.height = (numRows * ROW_HEIGHT) + "px";
    viewport.style.setProperty("--cols", currentCols);
    updateVirtualView();
  }

  var containers = [];
  var seen = {};
  allResults.forEach(function(x) { var c = x.container_type; if (c && !seen[c]) { seen[c] = true; containers.push(c); } });
  containers.sort(naturalCompare);
  document.getElementById("filter-container").innerHTML = '<option value="">All</option>' + containers.map(function(c) { return '<option value="' + escapeAttr(c) + '">' + escapeHtml(c) + '</option>'; }).join("");
  updateFilterModuleArg();
  updateFilterSteps();
  document.getElementById("lightbox").onclick = function() { document.getElementById("lightbox").classList.remove("visible"); };
  document.getElementById("filter-bend").onchange = function() { updateFilterModuleArg(); render(); };
  ["filter-region", "filter-container", "filter-module-arg", "filter-steps", "group-by", "sort-by", "sort-order", "group-sort-agg", "group-sort-metric", "group-sort-order", "within-sort-by", "within-sort-order"].forEach(function(id) { var el = document.getElementById(id); if (el) el.onchange = render; });
  document.getElementById("search-path").oninput = debounce(render, 200);
  window.addEventListener("resize", debounce(onResize, 100));
  render();
})();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
