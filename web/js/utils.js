// Utility functions
class Utils {
    /**
     * Get absolute position of an element
     * @param {Element} element - The DOM element
     * @returns {Object} Position object with top, left, bottom, right
     */
    static getAbsolutePosition(element) {
        const rect = element.getBoundingClientRect();
        return {
            top: rect.top + window.scrollY,
            left: rect.left + window.scrollX,
            bottom: rect.bottom + window.scrollY,
            right: rect.right + window.scrollX,
        };
    }

    /**
     * Convert hex/rgb color to rgba
     * @param {string} color - Color string (hex or rgb)
     * @param {number} alpha - Alpha value
     * @returns {string} RGBA color string
     */
    static rgba(color, alpha) {
        if (String(color).startsWith("rgb")) {
            return color.replace(")", `, ${alpha})`).replace("rgb(", "rgba(");
        }
        const hex = color.replace("#", "");
        const n = parseInt(
            hex.length === 3
                ? hex.split("").map((h) => h + h).join("")
                : hex,
            16
        );
        const r = (n >> 16) & 255,
              g = (n >> 8) & 255,
              b = n & 255;
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    /** Known container types (innermost in type path used for highlight) */
    static KNOWN_CONTAINERS = ["SpatialTransformer", "Upsample", "Downsample", "ResBlock", "TimestepEmbedSequential", "CrossAttention"];

    /**
     * Group consecutive subs that share the same container into expandable panels.
     * @param {Array} subs - Array of sub objects with .container
     * @returns {Array<{ container: string, subs: Array, collapsed: boolean }>}
     */
    static buildContainerGroups(subs) {
        if (!subs || !subs.length) return [];
        const out = [];
        let cur = { container: subs[0].container, subs: [subs[0]], collapsed: true };
        for (let i = 1; i < subs.length; i++) {
            if (subs[i].container === cur.container) {
                cur.subs.push(subs[i]);
            } else {
                out.push(cur);
                cur = { container: subs[i].container, subs: [subs[i]], collapsed: true };
            }
        }
        out.push(cur);
        return out;
    }

    /**
     * Pick container from type path: rightmost known container, else parent (second-to-last part), else "Other".
     */
    static containerFromTypePath(typePath) {
        if (!typePath || typeof typePath !== "string") return "Other";
        const parts = typePath.split(".");
        for (let i = parts.length - 1; i >= 0; i--) {
            if (Utils.KNOWN_CONTAINERS.includes(parts[i])) return parts[i];
        }
        if (parts.length >= 2) return parts[parts.length - 2];
        return parts[0] || "Other";
    }

    /**
     * Group blocks by their base path
     * @param {Array} blocks - Array of block paths
     * @param {Array} blockTypes - Array of block types
     * @param {Array} [typePaths] - Optional type paths (grandparent.parent.leaf); used for category + container
     * @returns {Object} Grouped blocks
     */
    static groupBlocks(blocks, blockTypes, typePaths) {
        const grouped = {};
        const stemLast1 = (p) => (p.split(".").pop() || p);
        blocks.forEach((fullPath, i) => {
            const parts = fullPath.split(".");
            let base = parts.slice(0, 2).join(".");
            if (parts[0] === "middle_block" || parts[0] === "mid_block") base = "middle_block";
            if (parts[0] === "out") base = "out";
            if (!grouped[base]) grouped[base] = [];
            const sub = parts.slice(2).join(".") || "(root)";
            const typePath = typePaths && typePaths[i] != null ? typePaths[i] : null;
            const typeStr = blockTypes[i] || "";
            const category = typePath ? stemLast1(typePath) : stemLast1(typeStr);
            const container = typePath ? Utils.containerFromTypePath(typePath) : "Other";
            grouped[base].push({
                name: sub,
                type: typeStr,
                full: fullPath,
                typePath: typePath || typeStr,
                category,
                container,
            });
        });
        return grouped;
    }

    /**
     * Natural-order comparison for group keys (e.g. "0" < "1" < "0.attn1" < "1.ff").
     */
    static _naturalSortKeys(a, b) {
        const partsA = (a + "").split(".");
        const partsB = (b + "").split(".");
        for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
            const pa = partsA[i] ?? "";
            const pb = partsB[i] ?? "";
            const na = parseInt(pa, 10);
            const nb = parseInt(pb, 10);
            const aNum = !Number.isNaN(na) && String(na) === pa;
            const bNum = !Number.isNaN(nb) && String(nb) === pb;
            if (aNum && bNum) {
                if (na !== nb) return na - nb;
            } else {
                if (pa !== pb) return pa < pb ? -1 : 1;
            }
        }
        return 0;
    }

    /**
     * Returns true if any key in grouped matches SD-style patterns (input_blocks, down_blocks, etc.).
     */
    static _hasSDStructure(grouped) {
        const sdPatterns = ["input_blocks", "down_blocks", "output_blocks", "up_blocks", "middle_block", "mid_block"];
        return Object.keys(grouped).some((key) =>
            sdPatterns.some((p) => key === p || key.startsWith(p + "."))
        );
    }

    /**
     * Returns true if any key in grouped is or starts with single_blocks or double_blocks (Flux structure).
     */
    static _hasFluxStructure(grouped) {
        return Object.keys(grouped).some((key) =>
            key === "single_blocks" || key.startsWith("single_blocks.") ||
            key === "double_blocks" || key.startsWith("double_blocks.")
        );
    }

    /**
     * Categorize blocks into input, middle, and output.
     * SD: input_blocks/down_blocks → input, output_blocks/up_blocks → output, middle_block → middle.
     * Flux: single_blocks* → input (left leg), double_blocks* → output (right leg), other keys → output; middle = [].
     * Generic: natural sort, first half → input, second half → output, middle = [].
     * @param {Object} grouped - Grouped blocks
     * @param {{ modelType?: string }} [options] - Optional. modelType "sd" | "flux" | "unknown".
     * @returns {Object} Categorized blocks { input, middle, output }
     */
    static categorizeBlocks(grouped, options = {}) {
        const input = [],
              middle = [],
              output = [];

        const modelType = options.modelType;
        const useSD = modelType === "sd" || (modelType !== "flux" && modelType !== "unknown" && Utils._hasSDStructure(grouped));
        const useFlux = modelType === "flux" || (modelType !== "sd" && Utils._hasFluxStructure(grouped));

        console.log("Categorizing blocks, grouped keys:", Object.keys(grouped).slice(0, 10), "modelType:", modelType, "useSD:", useSD, "useFlux:", useFlux);

        if (useSD) {
            Object.entries(grouped).forEach(([key, subs]) => {
                const block = { name: key, subs };
                if (key.startsWith("input_blocks") || key.startsWith("down_blocks")) {
                    input.push(block);
                } else if (key.startsWith("output_blocks") || key.startsWith("up_blocks")) {
                    output.push(block);
                } else if (key === "middle_block" || key === "mid_block") {
                    middle.push(block);
                } else {
                    console.log("Uncategorized block key (SD mode):", key);
                }
            });

            const sortByIndex = (a, b) => {
                const getIndex = (name) => {
                    const match = name.match(/(\d+)/);
                    return match ? parseInt(match[1]) : 9999;
                };
                return getIndex(a.name) - getIndex(b.name);
            };
            input.sort(sortByIndex);
            output.sort(sortByIndex);
        } else if (useFlux) {
            // Flux: single_blocks* → input (left leg), double_blocks* → output (right leg), other → output
            const singleBlocks = [];
            const doubleBlocks = [];
            const other = [];
            Object.entries(grouped).forEach(([key, subs]) => {
                const block = { name: key, subs };
                if (key === "single_blocks" || key.startsWith("single_blocks.")) {
                    singleBlocks.push(block);
                } else if (key === "double_blocks" || key.startsWith("double_blocks.")) {
                    doubleBlocks.push(block);
                } else {
                    other.push(block);
                }
            });
            const sortNatural = (a, b) => Utils._naturalSortKeys(a.name, b.name);
            singleBlocks.sort(sortNatural);
            doubleBlocks.sort(sortNatural);
            other.sort(sortNatural);
            singleBlocks.forEach((b) => input.push(b));
            doubleBlocks.forEach((b) => output.push(b));
            // other.forEach((b) => output.push(b));
        } else {
            // Generic: use all groups, natural sort, first half → input, second half → output
            const entries = Object.entries(grouped).map(([key, subs]) => ({ name: key, subs }));
            entries.sort((a, b) => Utils._naturalSortKeys(a.name, b.name));
            const half = Math.ceil(entries.length / 2);
            entries.slice(0, half).forEach((block) => input.push(block));
            entries.slice(half).forEach((block) => output.push(block));
        }
        console.log("input", input);
        console.log("middle", middle);
        console.log("output", output);
        console.log(`Categorized: ${input.length} input, ${middle.length} middle, ${output.length} output blocks`);

        return { input, middle, output };
    }

    /**
     * Display name for a block: hide "input_blocks." / "output_blocks." etc., show only what's inside.
     * @param {string} name - Block name (e.g. "input_blocks.0", "output_blocks.1", "middle_block")
     * @returns {string} Label for UI
     */
    static blockDisplayName(name) {
        return name;
    }

    /**
     * Extract block type name from full type string
     * @param {string} type - Full type string
     * @returns {string} Simplified type name
     */
    static extractBlockTypeName(type) {
        const start = type.indexOf("TimestepEmbedSequential") + 24;
        const end = type.indexOf("Conv2d") - 1;
        return type.substring(start, end);
    }

    /**
     * Check if block has skip connections
     * @param {Object} block - Block object
     * @returns {boolean} True if block has skip connections
     */
    static hasSkipConnection(block) {
        return block.subs.some(s => s.name.includes("skip_connection"));
    }

    /**
     * Snap angle to nearest 90-degree increment
     * @param {number} angle - Raw angle
     * @returns {number} Snapped angle
     */
    static snapAngle(angle) {
        return Math.round(angle / 90) * 90;
    }

    /**
     * Clamp value between min and max
     * @param {number} value - Value to clamp
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @returns {number} Clamped value
     */
    static clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
} 