#!/usr/bin/env python3
"""
Serve the experiment explorer without ComfyUI.
Reads from web_bend_demo_experiments (or an exported folder) and serves data + images.

Usage:
  # From experiments dir (ComfyUI output/web_bend_demo_experiments)
  python serve_explorer_standalone.py --experiments-dir /path/to/output/web_bend_demo_experiments --port 8765

  # From an exported folder (has data.json + images/ + explorer.html)
  python serve_explorer_standalone.py --export-dir /path/to/exported_folder --port 8765

Then open http://localhost:8765/
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
except ImportError:
    print("Python 3 required")
    sys.exit(1)


def _experiments_base_dir() -> Path:
    out = os.environ.get("COMFYUI_OUTPUT", "")
    if not out:
        base = Path(__file__).resolve().parent.parent.parent.parent
        out = base / "output"
    return Path(out) / "web_bend_demo_experiments"


class ExplorerHandler(SimpleHTTPRequestHandler):
    """Serves explorer HTML, data.json per batch, and images from experiments or export dir."""

    export_dir = None  # set to Path for export mode
    experiments_dir = None  # set to Path for experiments mode

    def __init__(self, *args, **kwargs):
        if self.export_dir is not None:
            self.directory = str(self.export_dir)
        else:
            self.directory = str(self.experiments_dir or Path(__file__).parent.parent / "web")
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        path = self.path.split("?")[0].lstrip("/")
        # Export-dir mode: API + static
        if self.export_dir is not None:
            if path == "" or path == "index.html":
                self._serve_export_explorer()
                return
            if path == "api/list":
                self._serve_export_list()
                return
            if path.startswith("api/data/"):
                self._serve_export_data(path[9:].strip("/"))
                return
            if path.startswith("api/image/"):
                parts = path[10:].strip("/").split("/", 1)
                if len(parts) == 2:
                    self._serve_export_image(parts[0], parts[1])
                    return
            return SimpleHTTPRequestHandler.do_GET(self)
        # Experiments mode: serve list, data, images, and explorer HTML
        if path == "" or path == "index.html":
            self.serve_explorer_with_api()
            return
        if path == "api/list":
            self.serve_list()
            return
        if path.startswith("api/data/"):
            batch_id = path[9:].strip("/")
            self.serve_data(batch_id)
            return
        if path.startswith("api/image/"):
            parts = path[10:].strip("/").split("/", 1)
            if len(parts) == 2:
                self.serve_image(parts[0], parts[1])
                return
        if path.startswith("api/latent/"):
            parts = path[11:].strip("/").split("/", 1)
            if len(parts) == 2:
                self.serve_latent(parts[0], parts[1])
                return
        return SimpleHTTPRequestHandler.do_GET(self)

    def serve_explorer_with_api(self):
        """Serve explorer HTML that uses api/list and api/data/{batch_id}."""
        api_html = Path(__file__).resolve().parent.parent / "web" / "explorer.html"
        if not api_html.is_file():
            self.send_error(404)
            return
        html = api_html.read_text(encoding="utf-8")
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def serve_list(self):
        base = self.experiments_dir or ""
        batches = []
        for name in sorted(os.listdir(base)):
            sub = os.path.join(base, name)
            if not os.path.isdir(sub):
                continue
            manifest_path = os.path.join(sub, "manifest.json")
            info = {"batch_id": name, "result_count": 0}
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        info.update(json.load(f))
                except json.JSONDecodeError:
                    pass
            results_path = os.path.join(sub, "results.jsonl")
            if os.path.isfile(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    info["result_count"] = sum(1 for line in f if line.strip())
            batches.append(info)
        body = json.dumps({"ok": True, "batches": batches}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_data(self, batch_id):
        if not batch_id:
            self.send_error(400)
            return
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        sub = os.path.join(self.experiments_dir, safe)
        if not os.path.isdir(sub):
            self.send_error(404)
            return
        results = []
        results_path = os.path.join(sub, "results.jsonl")
        if os.path.isfile(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            fn = r.get("image_filename", "")
                            subfolder = r.get("image_subfolder", "")
                            if subfolder == "images" and fn:
                                r["image_url"] = f"/api/image/{safe}/{fn}"
                            else:
                                r["image_url"] = ""
                            # Add latent URL if available
                            lat_fn = r.get("latent_filename", "")
                            lat_subfolder = r.get("latent_subfolder", "")
                            if lat_fn and lat_subfolder == "latents":
                                r["latent_url"] = f"/api/latent/{safe}/{lat_fn}"
                            results.append(r)
                        except json.JSONDecodeError:
                            pass
        manifest = {}
        manifest_path = os.path.join(sub, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                pass
        body = json.dumps({"ok": True, "manifest": manifest, "results": results}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_image(self, batch_id, filename):
        if ".." in filename or "/" in filename or "\\" in filename:
            self.send_error(400)
            return
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        path = os.path.join(self.experiments_dir, safe, "images", filename)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_latent(self, batch_id, filename):
        if ".." in filename or "/" in filename or "\\" in filename:
            self.send_error(400)
            return
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in batch_id)
        path = os.path.join(self.experiments_dir, safe, "latents", filename)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_export_explorer(self):
        """Serve explorer.html for export-dir mode (from web or export dir)."""
        export_path = Path(self.export_dir) / "explorer.html"
        if export_path.is_file():
            html = export_path.read_text(encoding="utf-8")
        else:
            src = Path(__file__).resolve().parent.parent / "web" / "explorer.html"
            html = src.read_text(encoding="utf-8") if src.is_file() else ""
        if not html:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_export_list(self):
        """Return single batch for export folder (from data.json)."""
        data_path = Path(self.export_dir) / "data.json"
        if not data_path.is_file():
            body = json.dumps({"ok": True, "batches": []}).encode("utf-8")
        else:
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results = data.get("results", [])
                manifest = data.get("manifest", {})
                batch_id = manifest.get("batch_id", "export")
                body = json.dumps({"ok": True, "batches": [{"batch_id": batch_id, "result_count": len(results)}]}).encode("utf-8")
            except (json.JSONDecodeError, OSError):
                body = json.dumps({"ok": True, "batches": []}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_export_data(self, batch_id):
        """Return data.json for export folder (image_url already set by export script)."""
        data_path = Path(self.export_dir) / "data.json"
        if not data_path.is_file():
            self.send_error(404)
            return
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.send_error(500)
            return
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_export_image(self, batch_id, filename):
        """Serve image from export folder images/."""
        if ".." in filename or "/" in filename or "\\" in filename:
            self.send_error(400)
            return
        path = Path(self.export_dir) / "images" / filename
        if not path.is_file():
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Serve experiment explorer without ComfyUI")
    ap.add_argument("--experiments-dir", help="Path to web_bend_demo_experiments folder")
    ap.add_argument("--export-dir", help="Path to an exported folder (data.json + images/ + explorer.html)")
    ap.add_argument("--port", "-p", type=int, default=8765, help="Port to serve on")
    args = ap.parse_args()

    if args.export_dir:
        export_dir = Path(args.export_dir).resolve()
        if not export_dir.is_dir():
            print(f"Not a directory: {export_dir}")
            sys.exit(1)
        ExplorerHandler.export_dir = export_dir
        ExplorerHandler.experiments_dir = None
        print(f"Serving exported folder at http://localhost:{args.port}/")
        print("Open http://localhost:{args.port}/")
        server = HTTPServer(("", args.port), ExplorerHandler)
    else:
        experiments_dir = Path(args.experiments_dir or _experiments_base_dir()).resolve()
        if not experiments_dir.is_dir():
            print(f"Experiments dir not found: {experiments_dir}")
            sys.exit(1)
        ExplorerHandler.experiments_dir = str(experiments_dir)
        ExplorerHandler.export_dir = None
        print(f"Serving experiments at http://localhost:{args.port}/")
        print("Open http://localhost:{args.port}/")
        server = HTTPServer(("", args.port), ExplorerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
