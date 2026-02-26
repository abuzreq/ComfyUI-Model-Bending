# Web UI Setup

This folder contains the interactive bending web interface. It is served by ComfyUI at `{ComfyUI_URL}/web_bend_demo/` when the plugin is installed.

## Configuration

Edit **`js/config.js`**:

| Variable | Purpose |
|----------|---------|
| `COMFYUI_BASE_URL` | ComfyUI base URL (e.g. `http://127.0.0.1:8188`). Default is for local Comfy with `--listen`. |
| `VIEWCOMFY_API_URL` | (Optional) Remote ViewComfy API URL for remote generation. |
| `VIEWCOMFY_CLIENT_ID` | (Optional) ViewComfy client ID. |
| `VIEWCOMFY_CLIENT_SECRET` | (Optional) ViewComfy client secret. |
| `USE_EXPERIMENT_CACHE` | (Optional) If enabled, the UI looks up cached experiment images before queueing. Requires pre-run batches in `web_bend_demo_experiments/`. |

Do not commit real ViewComfy credentials.

## Serving the UI

- **With ComfyUI**: Install the plugin in `custom_nodes`, start ComfyUI (e.g. `python main.py --listen`), then open `http://127.0.0.1:8188/web_bend_demo/` (or your Comfy URL + `/web_bend_demo/`).
- **Standalone (no backend)**: You can serve this folder with any static server (e.g. `python -m http.server 8000`) to open the pages, but layer listing and bending require a connected ComfyUI workflow.

## Folder structure

```
web/
├── index.html          # Main bending UI
├── explorer.html       # Experiment explorer (filter/sort batches)
├── js/
│   ├── config.js       # Configuration (edit this)
│   ├── main.js         # App logic, diagram, API calls
│   ├── unet-visualizer.js
│   ├── image-manager.js
│   ├── utils.js, svg-connector.js, viewcomfyapi.js
│   └── vendor/         # D3, socket.io, etc.
├── style/
│   └── styles.css
├── assets/
└── web_bend_demo.js    # ComfyUI extension (adds Open button to node)
```

## Experiment cache and explorer

- Run batches with `scripts/run_bending_experiments.py`; results go to ComfyUI output under `web_bend_demo_experiments/`.
- With `USE_EXPERIMENT_CACHE` enabled in `config.js`, the UI uses cached images when available.
- Explorer: open `{ComfyUI_URL}/web_bend_demo/explorer.html` to filter/sort/group results.
- Offline: export a batch with `scripts/export_experiments.py`, then serve with `scripts/serve_explorer_standalone.py --export-dir ./my_export --port 8765`.

For plugin overview, nodes, and usage, see the root [README.md](../README.md).
