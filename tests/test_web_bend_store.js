// Runs the web UI's real config.js + image-manager.js in a sandbox and checks that its bend store keeps the
// version 1.1 keys of the bends JSON (docs/bends-json.md). Run: node tests/test_web_bend_store.js
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const src = (f) => fs.readFileSync(path.join(__dirname, "..", f), "utf8");
const ctx = { console, document: { getElementById: () => null, querySelector: () => null, addEventListener() {} }, window: {} };
vm.createContext(ctx);
const code = src("web/js/config.js") + "\n" + src("web/js/image-manager.js").replace("const imageManager = new ImageManager();", "")
  + "\nthis.ImageManager = ImageManager; this.BENDS_JSON_VERSION = BENDS_JSON_VERSION;";
vm.runInContext(code, ctx);
const im = Object.create(ctx.ImageManager.prototype);
im.bends = new Map();

const incoming = [
  { path: "middle_block.1", module_type: "multiply", module_args: { scalar: 2 }, t: [1, 0.5], blend: 0.8, guard: { max_std_ratio: 4 } },
  { path: "output_blocks.*.1", module_type: "fourier", module_args: { cutoff_freq: 3 }, steps: "0-2" },
  { path: "input_blocks.4.1", module_type: "subset", module_args: { percentage: 0.5, dim: "channel" },
    inner: { module_type: "rotate", module_args: { angle_degrees: 90 } } },
  { path: "input_blocks.1.0", angle: 90, label: "legacy" },
];
im.setBends(incoming);
const out = im.getBends();
const plain = (v) => JSON.parse(JSON.stringify(v));
assert.deepStrictEqual(plain(out.slice(0, 3)), incoming.slice(0, 3), "1.1 keys and unknown ops survive setBends -> getBends");
assert.deepStrictEqual(plain(out[3]), { path: "input_blocks.1.0", module_type: "rotate", module_args: { angle_degrees: 90 }, label: "legacy" });

// moving a slider on a bend keeps its window; returning to the default removes the bend
im.setBend("middle_block.1", "multiply", { scalar: 0.5 });
assert.deepStrictEqual(plain(im.getBends()[0]), { path: "middle_block.1", module_type: "multiply", module_args: { scalar: 0.5 }, t: [1, 0.5], blend: 0.8, guard: { max_std_ratio: 4 } });
// changing a subset bend to a UI op drops "inner" only
im.setBend("input_blocks.4.1", "rotate", { angle_degrees: 180 });
assert.strictEqual(im.getBend("input_blocks.4.1").extras.inner, undefined);
im.setBend("middle_block.1", "multiply", { scalar: 1 });
assert.strictEqual(im.getBend("middle_block.1"), null);
// plain v1 bends are emitted exactly as before
im.setBends([{ path: "a.b", module_type: "rotate", module_args: { angle_degrees: 90 } }]);
assert.deepStrictEqual(plain(im.getBends()), [{ path: "a.b", module_type: "rotate", module_args: { angle_degrees: 90 } }]);
assert.strictEqual(ctx.BENDS_JSON_VERSION, 1.1);
console.log("web UI bend store: all checks passed");
