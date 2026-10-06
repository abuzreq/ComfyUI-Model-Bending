# Changelog

## 0.3.1

- Bends JSON **1.2**: a bend may carry a `kb` object that says where it came from in the bend knowledge base
  (`{"dataset", "record"}` or `{"dataset", "cell"}`). The node accepts it (also with `strict` on), never uses it
  to bend, and keeps it in `resolved_json`. A `kb` that is not an object is dropped with a warning. The web UI
  keeps `kb` through "Copy Bends" and drops it when a slider changes the bend. Before 0.3.1, `kb` was an unknown
  key: a warning, or an error with `strict`. See [docs/bends-json.md](docs/bends-json.md).

## 0.3.0

Time windows, DiT block bending, activation probes and steering vectors, bends JSON 1.1, experimental video
(WAN) bending and new ops. See the README for the full list.
