// UNet Visualizer Module
class UNetVisualizer {
  constructor() {
    this.selectedSubGroup = null;
    this.selectedGroup = null;
    this.openedBlockGroup = null;
    this.dragStartY = 0;
    this.dragStartAngle = 0;
    this.hasDragged = false;
    this.blockPositions = {};
    this.activeEffects = new Map(); // Track active effects
  }

  static formatBendValue(bendType, value) {
    const t = CONFIG.BENDING_TYPES?.[bendType];
    if (!t) return String(value);
    if (bendType === "rotate") return value + "°";
    if (bendType === "add_noise") return value === 0 ? "0" : "σ=" + value;
    if (bendType === "multiply") return "×" + value;
    return String(value);
  }

  static getSliderConfig(bendType) {
    const t = CONFIG.BENDING_TYPES?.[bendType];
    const levels = t?.levels || [0];
    const defVal = t?.defaultValue;
    const defIdx = levels.indexOf(defVal);
    return { levels, defaultIndex: defIdx >= 0 ? defIdx : 0 };
  }

  static updateSliderTicks(sliderGroup, bendType, sliderWidth) {
    sliderGroup.selectAll("line.slider-tick").remove();
    const cfg = UNetVisualizer.getSliderConfig(bendType);
    const levels = cfg.levels;
    const defaultIdx = cfg.defaultIndex;
    levels.forEach((_, i) => {
      const n = Math.max(1, levels.length - 1);
      const tx = (i / n) * sliderWidth;
      const isDefault = i === defaultIdx;
      sliderGroup.insert("line", "circle")
        .attr("class", isDefault ? "slider-tick slider-tick-default" : "slider-tick")
        .attr("x1", tx).attr("x2", tx)
        .attr("y1", 4).attr("y2", -4)
        .attr("stroke", isDefault ? "#555" : "#aaa")
        .attr("stroke-width", isDefault ? 1.5 : 1);
    });
  }

  static updateSliderValueDisplay(valueEl, bendType, value) {
    if (!valueEl || valueEl.empty()) return;
    valueEl.text(UNetVisualizer.formatBendValue(bendType, value));
  }

  /**
   * Set up expanded area for visualization
   * @param {string} areaId - ID of the area element
   * @param {string} elementId - ID of the triggering element
   * @param {boolean} indicateInProgress - Whether to show progress indicator
   * @returns {Object} D3 SVG selection
   */
  setupExpandedArea(
    areaId,
    elementId,
    forceHide = false,
    indicateInProgress = true
  ) {
    console.log(
      "setupExpandedArea",
      areaId,
      elementId,
      forceHide,
      indicateInProgress
    );
    const wrapper = document.getElementById(areaId);
    if (!wrapper) {
      console.error("setupExpandedArea: wrapper not found for", areaId);
      return null;
    }
    const diagramContainer = document.getElementById("diagram-container");
    if (diagramContainer) diagramContainer.style.overflow = "visible";

    wrapper.innerHTML = indicateInProgress
      ? '<div style="position: relative; top: 50%; left: 50%;">More to come...</div>'
      : "";
    wrapper.style.visibility = "hidden";

    document
      .querySelectorAll(".clickable > rect, .clickable > path")
      .forEach((el) => {
        el.style.fill = "#ffffff";
      });
    this.updateInfo(null);
    const legendContainer = document.querySelector(".legend-container");
    if (legendContainer) legendContainer.style.visibility = "hidden";

    if (elementId) {
      this.updateInfo(elementId);
      const diagramEl = document.querySelector(`#${elementId} > rect, #${elementId} > path`);
      if (diagramEl) diagramEl.style.fill = CONFIG.COLORS.BLOCK_FILL;
      if (CONFIG.BENDABLE.includes(elementId)) {
        wrapper.style.visibility = "visible";
        if (diagramContainer) diagramContainer.style.overflow = "hidden";
      }
    }

    const svg = d3
      .select(wrapper)
      .append("svg")
      .attr("width", "100%")
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("viewBox", "0 0 1250 850");

    this.addRubberTextFilter(svg);

    const zoomContainer = svg.append("g").attr("class", "zoom-container");
    const zoom = d3.zoom()
      .scaleExtent([0.5, 2])
      .filter((event) => {
        const t = event.target;
        if (!t || !t.closest) return true;
        if (t.closest("g[name]")) return false;
        if (t.closest(".svg-slider")) return false;
        if (t.closest("foreignObject")) return false;
        if (t.closest(".block-reset-btn")) return false;
        return true;
      })
      .on("zoom", (event) => zoomContainer.attr("transform", event.transform));
    svg.call(zoom);
    svg.call(zoom.transform, d3.zoomIdentity.translate(0, 100));

    return zoomContainer;
  }

  updateInfo(elementId) {
    document.querySelectorAll(`#info-panel > span`).forEach((el) => {
      el.style.display = "none";
    });
    console.log(
      elementId,
      document.querySelectorAll(`#info-panel > span`),
      document.querySelector(`#info-panel > #info-${elementId}`)
    );
    if (elementId !== null) {
      const infoEl = document.querySelector(`#info-panel > #info-${elementId}`);
      if (infoEl !== null) {
        infoEl.style.display = "block";
      }
    } else {
      console.log("show general");
      document.querySelector(`#info-panel > #info-general`).style.display =
        "block";
    }
  }

  /**
   * Add rubber text filter to SVG
   * @param {Object} svg - D3 SVG selection
   */
  addRubberTextFilter(svg) {
    const defs = svg.append("defs");

    const filter = defs
      .append("filter")
      .attr("id", "fx-rubber")
      .attr("filterUnits", "userSpaceOnUse")
      .attr("x", "-20%")
      .attr("y", "-40%")
      .attr("width", "140%")
      .attr("height", "180%");

    filter
      .append("feTurbulence")
      .attr("type", "fractalNoise")
      .attr("baseFrequency", "0.015")
      .attr("numOctaves", "2")
      .attr("seed", "3")
      .attr("result", "lightenedBlur");

    filter
      .append("feDisplacementMap")
      .attr("id", "rubber-disp-text")
      .attr("in", "SourceGraphic")
      .attr("in2", "noise")
      .attr("scale", "0")
      .attr("xChannelSelector", "R")
      .attr("yChannelSelector", "G");

    //filter.append("feGaussianBlur").attr("stdDeviation", "0.25");

    // (optional) animate wobble “liveliness” by jittering baseFrequency a bit
    let t = 0;
    const turb = document.querySelector("#fx-rubber feTurbulence");
    function tick() {
      t += 0.015;
      const base = 0.02 + Math.sin(t) * 0.003; // subtle
      turb.setAttribute("baseFrequency", base.toFixed(4));
      requestAnimationFrame(tick);
    }
    tick();
  }

  /**
   * Update rubber effect scale
   * @param {number} scale - Scale value for the displacement
   */
  updateRubberEffect(scale) {
    const displacement = document.querySelector("#rubber-disp-text");
    if (displacement) {
      displacement.setAttribute("scale", 12 * scale);
    }
  }

  /**
   * Reset all block sliders and dropdowns. Call after clearAllBends.
   */
  resetAllSliders() {
    if (!this.unetSvg) return;
    const sliderWidth = (3 * CONFIG.BLOCK_WIDTH) / 4 - 42 - 8 + 30 - 28;
    this.unetSvg.selectAll(".sub-group").each(function (d) {
      const g = d3.select(this);
      if (d) d.bendType = "rotate", d.sliderIndex = 0;
      const sel = g.select(".bend-type-select").node();
      if (sel) sel.value = "rotate";
      g.select(".svg-slider circle").attr("cx", 0);
      UNetVisualizer.updateSliderTicks(g.select(".svg-slider"), "rotate", sliderWidth);
      UNetVisualizer.updateSliderValueDisplay(g.select(".slider-value-text"), "rotate", 0);
    });
    if (typeof this.updateBlockHighlights === "function") this.updateBlockHighlights();
  }

  /**
   * Sync sliders and dropdowns to match bends. Handles {path, module_type, module_args} and legacy {path, angle}.
   * @param {Array<{path: string, module_type?: string, module_args?: Object, angle?: number}>} bends
   */
  syncSlidersFromBends(bends) {
    if (!this.unetSvg) return;
    const map = new Map();
    (bends || []).forEach((b) => {
      if (b.module_type && b.module_args) {
        const t = CONFIG.BENDING_TYPES?.[b.module_type];
        const k = t?.module_args_key;
        const val = k ? b.module_args[k] : undefined;
        map.set(b.path, { module_type: b.module_type, value: val });
      } else if (typeof b.angle === "number") {
        map.set(b.path, { module_type: "rotate", value: b.angle });
      }
    });
    const sliderWidth = (3 * CONFIG.BLOCK_WIDTH) / 4 - 42 - 8 + 30 - 28;
    this.unetSvg.selectAll(".sub-group").each(function (d) {
      if (!d) return;
      const g = d3.select(this);
      const entry = map.get(d.full);
      if (!entry) {
        d.bendType = "rotate";
        d.sliderIndex = 0;
        const sel = g.select(".bend-type-select").node();
        if (sel) sel.value = "rotate";
        g.select(".svg-slider circle").attr("cx", 0);
        UNetVisualizer.updateSliderTicks(g.select(".svg-slider"), "rotate", sliderWidth);
        UNetVisualizer.updateSliderValueDisplay(g.select(".slider-value-text"), "rotate", 0);
        return;
      }
      const t = CONFIG.BENDING_TYPES?.[entry.module_type];
      const levels = t?.levels || [0];
      const idx = levels.indexOf(entry.value);
      d.bendType = entry.module_type;
      d.sliderIndex = idx >= 0 ? idx : 0;
      const val = levels[d.sliderIndex] ?? levels[0];
      const sel = g.select(".bend-type-select").node();
      if (sel) sel.value = entry.module_type;
      const numSteps = Math.max(1, levels.length - 1);
      const x = (d.sliderIndex / numSteps) * sliderWidth;
      g.select(".svg-slider circle").attr("cx", x);
      UNetVisualizer.updateSliderTicks(g.select(".svg-slider"), d.bendType, sliderWidth);
      UNetVisualizer.updateSliderValueDisplay(g.select(".slider-value-text"), d.bendType, val);
    });
    if (typeof this.updateBlockHighlights === "function") this.updateBlockHighlights();
  }

  /**
   * Apply or remove rubber effect and bent styling on all sub-groups based on bends.
   */
  updateRubberEffectOnBentLayers() {
    if (!this.unetSvg || !window.comfySessionId) return;
    const svg = this.unetSvg;
    const bends = imageManager.getBends();
    const bentPaths = new Set(bends.map((b) => b.path));
    const intensity = imageManager.getBendingIntensity();

    svg.selectAll(".sub-group").each(function (d) {
      const sg = d3.select(this);
      const isBent = d && bentPaths.has(d.full);
      sg.classed("bent", !!isBent);
      const label = sg.select(".sub-label").node();
      const accent = sg.select(".sub-accent").node();
      const rect = sg.select(".subgroup-rect").node();
      const sliderLine = sg.select(".svg-slider > line").node();
      [label, accent, rect, sliderLine].forEach((el) => {
        if (!el) return;
        if (isBent) el.setAttribute("filter", "url(#fx-rubber)");
        else el.removeAttribute("filter");
      });
    });
    this.updateRubberEffect(intensity);
  }

  /**
   * Update block rect fill based on bend state. Modified blocks use MODIFIED_BLOCK.
   * Also highlights container headers when a sub inside them is bent; persists when closed.
   */
  updateBlockHighlights() {
    if (!this.unetSvg || !window.comfySessionId) return;
    const svg = this.unetSvg;
    const opened = this.openedBlockGroup;
    const bends = imageManager.getBends();
    const bentPaths = new Set(bends.map((b) => b.path));

    this.updateRubberEffectOnBentLayers();

    svg.selectAll("g[name]").each(function () {
      const g = d3.select(this);
      const name = g.attr("name");
      if (!name) return;
      const rect = g.select(".block-main-rect");
      if (rect.empty()) return;
      const isOpen = opened && opened.node() === g.node();
      if (isOpen) {
        rect.style("fill", CONFIG.COLORS.BLOCK_HIGHLIGHT);
      } else {
        const modified = bends.some(
          (b) => b.path === name || b.path.startsWith(name + ".")
        );
        rect.style("fill", modified ? CONFIG.COLORS.MODIFIED_BLOCK : CONFIG.COLORS.BLOCK_FILL);
      }
    });

    svg.selectAll(".container-group").each(function () {
      const groupG = d3.select(this);
      const grp = groupG.datum();
      if (!grp || !grp.subs) return;
      const hasBentSub = grp.subs.some((s) => bentPaths.has(s.full));
      const headerRect = groupG.select(".container-header-rect");
      if (headerRect.empty()) return;
      headerRect.style(
        "fill",
        hasBentSub ? CONFIG.COLORS.MODIFIED_BLOCK : "#eee"
      );
    });
  }

  /**
   * Expand UNet visualization
   */
  async expandUnet() {
    const svg = this.setupExpandedArea("unet-expanded", "unet", false, false);
    if (!svg) {
      console.error("Failed to setup expanded area - svg is null");
      return;
    }

    console.log("Expanding UNet, preparing block data...");
    const blockData = await this.prepareBlockData();
    console.log("Block data prepared:", blockData);
    console.log("Input blocks:", blockData.input?.length || 0);
    console.log("Middle blocks:", blockData.middle?.length || 0);
    console.log("Output blocks:", blockData.output?.length || 0);
    
    const { input, middle, output } = blockData;
    
    if ((!input || input.length === 0) && (!middle || middle.length === 0) && (!output || output.length === 0)) {
      console.error("No blocks to display! All arrays are empty.");
      // Show error message in the expanded area
      svg.append("text")
        .attr("x", 100)
        .attr("y", 100)
        .attr("fill", "red")
        .text("No model structure data available. Please refresh the model structure.");
      return;
    }
    
    console.log("Laying out blocks...");
    this.unetSvg = svg;
    this.layoutBlocks(svg, input, middle, output);
    const flowLayer = svg.insert("g", ":first-child").attr("class", "flow-layer");
    console.log("Drawing flow arrows...");
    this.drawFlowArrows(flowLayer, input, middle, output);
    console.log("Drawing skip lines...");
    this.drawSkipLines(svg, input, output);
    if (typeof this.updateBlockHighlights === "function") this.updateBlockHighlights();

    const catMap = UNET_BLOCKS[imageManager.getModelName()]?.blockTypeCategories
      || UNET_BLOCKS["ComfyUI"]?.blockTypeCategories
      || UNET_BLOCKS["SD1.4"]?.blockTypeCategories;
    if (catMap && typeof window.updateLegendFromBlockTypeCategories === "function") {
      window.updateLegendFromBlockTypeCategories(catMap);
    }
    document.querySelector(".legend-container").style.visibility = "visible";
    console.log("UNet expansion complete");
  }

  /**
   * Prepare block data for visualization
   * @returns {Object} Categorized blocks
   */
  async prepareBlockData() {
    console.log("prepareBlockData", imageManager.getModelName());
    console.log("COMFY SESSION ID", window.comfySessionId);
    console.log("FETCH MODEL STRUCTURE", window.fetchModelStructure);
    
    // PRIORITY 1: Always try to fetch from ComfyUI session first - this is the primary source
    if (window.comfySessionId && window.fetchModelStructure) {
      try {
        console.log("Attempting to fetch model structure from ComfyUI...");
        const fetched = await window.fetchModelStructure();
        console.log("Fetch result:", fetched);
        
        // After fetching, look for the fetched data in UNET_BLOCKS
        // The fetch function stores it under multiple keys including "ComfyUI"
        const fetchedBlocks = UNET_BLOCKS["ComfyUI"] || UNET_BLOCKS[imageManager.getModelName()];
        console.log("Fetched blocks:", fetchedBlocks);

        if (fetchedBlocks && fetchedBlocks.blocks && fetchedBlocks.blocks.length > 0) {
          console.log(`✓ Using ${fetchedBlocks.blocks.length} blocks fetched from ComfyUI`);
          console.log("Sample blocks:", fetchedBlocks.blocks.slice(0, 5));
          console.log("Sample blockTypes:", fetchedBlocks.blockTypes.slice(0, 5));
          const grouped = Utils.groupBlocks(fetchedBlocks.blocks, fetchedBlocks.blockTypes, fetchedBlocks.typePaths);
          console.log("Grouped blocks:", Object.keys(grouped).slice(0, 10));
          const categorized = Utils.categorizeBlocks(grouped, { modelType: window.webBendDemoModelType });
          console.log("Categorized - input:", categorized.input.length, "middle:", categorized.middle.length, "output:", categorized.output.length);
          return categorized;
        } else {
          console.warn("Fetch succeeded but no blocks found in UNET_BLOCKS after fetch");
          console.log("Checking all UNET_BLOCKS keys for fetched data...");
          // Check all keys to see if data was stored elsewhere
          for (const key of Object.keys(UNET_BLOCKS)) {
            const blocks = UNET_BLOCKS[key];
            if (blocks && blocks.blocks && blocks.blocks.length > 0) {
              // Check if this looks like fetched data (recently added)
              console.log(`Found ${blocks.blocks.length} blocks in key: ${key}`);
              // If we have a session, prefer any non-predefined key
              if (key !== "SD1.4" && key !== "SDTurbo" && key !== "SD-1.4" && key !== "SD-turbo") {
                console.log(`✓ Using ${blocks.blocks.length} blocks from key: ${key} (likely from ComfyUI)`);
                console.log("Sample blocks:", blocks.blocks.slice(0, 5));
                const grouped = Utils.groupBlocks(blocks.blocks, blocks.blockTypes, blocks.typePaths);
                console.log("Grouped blocks:", Object.keys(grouped).slice(0, 10));
                const categorized = Utils.categorizeBlocks(grouped, { modelType: window.webBendDemoModelType });
                console.log("Categorized - input:", categorized.input.length, "middle:", categorized.middle.length, "output:", categorized.output.length);
                return categorized;
              }
            }
          }
        }
      } catch (e) {
        console.error("Failed to fetch model structure from ComfyUI:", e);
        // Continue to fallback only if fetch completely fails
      }
    } else {
      if (!window.comfySessionId) {
        console.warn("No ComfyUI session_id available - cannot fetch model structure");
      }
      if (!window.fetchModelStructure) {
        console.warn("fetchModelStructure function not available");
      }
    }
    
    // PRIORITY 2: Only use predefined structures as fallback if ComfyUI fetch failed or unavailable
    console.log("Falling back to predefined model structures (this should only happen if ComfyUI fetch failed)...");
    const modelName = imageManager.getModelName();
    console.log("Looking for predefined blocks with name:", modelName);
    console.log("Available UNET_BLOCKS keys:", Object.keys(UNET_BLOCKS));
    
    // Try to find predefined blocks
    let modelBlocks = null;
    const keysToTry = [modelName, "SD1.4", "SDTurbo"];
    
    for (const key of keysToTry) {
      const blocks = UNET_BLOCKS[key];
      if (blocks && blocks.blocks && blocks.blocks.length > 0) {
        modelBlocks = blocks;
        console.log(`⚠ Using predefined blocks from key: ${key} (fallback mode)`);
        break;
      }
    }
    
    if (!modelBlocks) {
      console.warn(`No predefined block data found, checking all keys...`);
      // Try to find any predefined model with blocks
      for (const key of Object.keys(UNET_BLOCKS)) {
        // Skip "ComfyUI" key as it should only be used if fetched
        if (key === "ComfyUI") continue;
        const blocks = UNET_BLOCKS[key];
        if (blocks && blocks.blocks && blocks.blocks.length > 0) {
          console.log(`⚠ Using predefined model blocks from key: ${key} (fallback mode)`);
          modelBlocks = blocks;
          break;
        }
      }
      
      if (!modelBlocks) {
        console.error("No valid block data found - neither from ComfyUI nor predefined structures");
        return { input: [], middle: [], output: [] };
      }
    }
    
    console.log(`⚠ Using ${modelBlocks.blocks.length} predefined blocks as fallback`);
    const grouped = Utils.groupBlocks(modelBlocks.blocks, modelBlocks.blockTypes, modelBlocks.typePaths);
    return Utils.categorizeBlocks(grouped, { modelType: window.webBendDemoModelType });
  }

  /**
   * Layout blocks in the visualization
   * @param {Object} svg - D3 SVG selection
   * @param {Array} input - Input blocks
   * @param {Array} middle - Middle blocks
   * @param {Array} output - Output blocks
   */
  layoutBlocks(svg, input, middle, output) {
    const legAngle = (CONFIG.LEG_ANGLE_DEG * Math.PI) / 180;
    const dxPerStep = CONFIG.DY / Math.tan(legAngle);

    // Record positions for skip lines
    const pos = new Map();
    const place = (block, x, y, segment) => {
      block._flowSegment = segment;
      pos.set(block.name, { x, y });
      this.drawBlock(svg, x, y, block);
    };
    var isComfyBigModel = !imageManager.getModelName().includes("-");
    // Left leg (down)
    input.forEach((block, i) => {
      place(
        block,
        (isComfyBigModel ? CONFIG.X_OFFSET : CONFIG.X_OFFSET * 10) +
          i * dxPerStep,
        CONFIG.Y_OFFSET + i * CONFIG.DY,
        "input"
      );
    });
    console.log(imageManager.getModelName().includes("-"));
    // Middle
    let midX =
      (isComfyBigModel ? CONFIG.X_OFFSET : CONFIG.X_OFFSET * 10) +
      input.length * dxPerStep;
    let midY = CONFIG.Y_OFFSET + input.length * CONFIG.DY;
    if (middle[0]) {
      place(
        middle[0],
        CONFIG.BLOCK_WIDTH / 2 +
          midX +
          CONFIG.GAP_MIDDLE / 2 -
          CONFIG.BLOCK_WIDTH / 2,
        midY,
        "middle"
      );
    }

    // Right leg (up)
    output.forEach((block, i) => {
      const x = midX + CONFIG.GAP_MIDDLE + i * dxPerStep;
      const y = midY - (i + 1) * CONFIG.DY;
      place(block, x, y, "output");
    });
  }
  /**
   * Draw a single block
   * @param {Object} svg - D3 SVG selection
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {Object} block - Block data
   */
  drawBlock(svg, x, y, block) {
    const g = svg
      .append("g")
      .attr("name", block.name)
      .attr("transform", `translate(${x},${y})`)
      .datum(block);
    const seg = block._flowSegment || "output";
    g.classed("flow-input", seg === "input").classed("flow-middle", seg === "middle").classed("flow-output", seg === "output");
    this.blockPositions[block.name] = { x, y };

    const hasSkip = Utils.hasSkipConnection(block);
    const HEADER_H = CONFIG.CONTAINER_HEADER_HEIGHT || 22;

    const containerGroups = Utils.buildContainerGroups(block.subs);
    block._containerGroups = containerGroups;

    function computeLayout() {
      let y = CONFIG.COLLAPSED_HEIGHT;
      const layout = [];
      const endGap = CONFIG.CONTAINER_CONTENT_END_GAP || 0;
      for (const grp of containerGroups) {
        const contentH = grp.collapsed ? 0 : grp.subs.length * CONFIG.DIST_BETWEEN_SUBS + endGap;
        layout.push({ y, headerH: HEADER_H, contentH, totalH: HEADER_H + contentH });
        y += HEADER_H + contentH;
      }
      const fullContentH = y - CONFIG.COLLAPSED_HEIGHT;
      const maxContentH = CONFIG.MAX_BLOCK_SUBS_VISIBLE * CONFIG.DIST_BETWEEN_SUBS;
      const cappedContentH = Math.min(fullContentH, maxContentH);
      const expandedHeight = CONFIG.COLLAPSED_HEIGHT + cappedContentH;
      const canScroll = fullContentH > maxContentH;
      block._layout = layout;
      block._fullContentH = fullContentH;
      block._expandedHeight = expandedHeight;
      block._canScroll = canScroll;
      return { layout, fullContentH, expandedHeight, canScroll };
    }
    computeLayout();

    const rect = g
      .append("rect")
      .attr("class", "block-main-rect")
      .attr("width", CONFIG.BLOCK_WIDTH)
      .attr("height", CONFIG.COLLAPSED_HEIGHT)
      .attr("rx", 4)
      .attr("fill", CONFIG.COLORS.BLOCK_FILL)
      .attr("stroke", "#333")
      .attr("stroke-width", 1.5);

    const labelText = Utils.blockDisplayName(block.name) + (hasSkip ? " ↔" : "");
    g.append("text")
      .attr("x", 55)
      .attr("y", 18)
      .attr("class", "label")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .text(labelText);

    const blockResetBtn = g
      .append("text")
      .attr("class", "block-reset-btn reset-btn-icon")
      .attr("x", CONFIG.BLOCK_WIDTH - 22)
      .attr("y", 18)
      .attr("cursor", "pointer")
      .attr("title", "Reset block")
      .style("user-select", "none")
      .style("font-size", "14px")
      .style("fill", "#666")
      .text(CONFIG.RESET_ICON || "↺");
    blockResetBtn.on("mouseenter", function () {
      d3.select(this).style("fill", "#333");
    });
    blockResetBtn.on("mouseleave", function () {
      d3.select(this).style("fill", "#666");
    });

    const clipId = "clip-block-" + block.name.replace(/[^a-z0-9_-]/gi, "-");
    g.append("defs")
      .append("clipPath")
      .attr("id", clipId)
      .append("rect")
      .attr("class", "block-clip-rect")
      .attr("width", CONFIG.BLOCK_WIDTH)
      .attr("height", block._expandedHeight)
      .attr("x", 0)
      .attr("y", 0);
    g.attr("clip-path", "url(#" + clipId + ")");

    const subsContainer = g.append("g").attr("class", "block-subs-container");
    const layer = subsContainer.append("g").attr("class", "container-groups-layer");

    let outerThis = this;
    const catMap = UNET_BLOCKS[imageManager.getModelName()]?.blockTypeCategories
      || UNET_BLOCKS["ComfyUI"]?.blockTypeCategories
      || UNET_BLOCKS["SD1.4"]?.blockTypeCategories
      || {};
    const sliderX = 8;
    const dropdownWidth = 42;
    const gapBetween = 8;
    const valueTextSpace = 10;
    const extraSliderSpace = 30;
    const sliderWidth = (3 * CONFIG.BLOCK_WIDTH) / 4 - dropdownWidth - 8 + extraSliderSpace - valueTextSpace;
    const dropdownX = sliderX + sliderWidth + gapBetween + valueTextSpace;
    const sliderY = 18;

    const allSubGroups = () => g.selectAll(".sub-group");

    const getSliderConfig = UNetVisualizer.getSliderConfig;
    const updateSliderTicks = UNetVisualizer.updateSliderTicks;
    const updateSliderValueDisplay = UNetVisualizer.updateSliderValueDisplay;

    containerGroups.forEach((grp, gi) => {
      const l = block._layout[gi];
      const groupG = layer
        .append("g")
        .attr("class", "container-group")
        .attr("transform", `translate(0, ${l.y})`)
        .datum(grp);

      const header = groupG
        .append("g")
        .attr("class", "container-group-header")
        .style("cursor", "pointer")
        .on("click", (event) => {
          event.stopPropagation();
          const opening = grp.collapsed;
          if (opening) {
            block._containerGroups.forEach((other) => {
              if (other !== grp) other.collapsed = true;
            });
          }
          grp.collapsed = !grp.collapsed;
          computeLayout();
          outerThis.updateContainerGroupLayout(g, block);
        });
      header
        .append("rect")
        .attr("class", "container-header-rect")
        .attr("width", CONFIG.BLOCK_WIDTH - 4)
        .attr("height", HEADER_H)
        .attr("x", 2)
        .attr("y", 0)
        .attr("rx", 2)
        .attr("fill", "#eee")
        .attr("stroke", "#ddd");
      header
        .append("text")
        .attr("class", "container-header-chevron")
        .attr("x", 8)
        .attr("y", HEADER_H / 2 + 1)
        .attr("dominant-baseline", "middle")
        .style("font-size", "10px")
        .style("font-weight", "bold")
        .text(() => (grp.collapsed ? "▶" : "▼"));
      header
        .append("text")
        .attr("class", "container-header-label")
        .attr("x", 24)
        .attr("y", HEADER_H / 2 + 1)
        .attr("dominant-baseline", "middle")
        .style("font-size", "11px")
        .style("font-weight", "600")
        .text(() => `${grp.container} (${grp.subs.length})`);

      const containerResetBtn = header
        .append("text")
        .attr("class", "container-reset-btn reset-btn-icon")
        .attr("x", CONFIG.BLOCK_WIDTH - 18)
        .attr("y", HEADER_H / 2 + 1)
        .attr("dominant-baseline", "middle")
        .attr("cursor", "pointer")
        .attr("title", "Reset container")
        .style("user-select", "none")
        .style("font-size", "12px")
        .style("fill", "#666")
        .text(CONFIG.RESET_ICON || "↺");
      containerResetBtn.on("click", (event) => {
        event.stopPropagation();
        grp.subs.forEach((sub) => imageManager.setBend(sub.full, 0));
        groupG.selectAll(".sub-group").each(function (d) {
          const sg = d3.select(this);
          d.bendType = "rotate";
          d.sliderIndex = 0;
          const sel = sg.select(".bend-type-select").node();
          if (sel) sel.value = "rotate";
          sg.select(".svg-slider circle").attr("cx", 0);
          UNetVisualizer.updateSliderTicks(sg.select(".svg-slider"), "rotate", sliderWidth);
          UNetVisualizer.updateSliderValueDisplay(sg.select(".slider-value-text"), "rotate", 0);
        });
        imageManager.updateBends();
        outerThis.updateRubberEffect(imageManager.getBendingIntensity());
        if (typeof outerThis.updateBlockHighlights === "function") outerThis.updateBlockHighlights();
        imageManager.commitBendsIfLive();
      });
      containerResetBtn.on("mouseenter", function () {
        d3.select(this).style("fill", "#333");
      });
      containerResetBtn.on("mouseleave", function () {
        d3.select(this).style("fill", "#666");
      });

      const endGap = CONFIG.CONTAINER_CONTENT_END_GAP || 0;
      const contentH = grp.subs.length * CONFIG.DIST_BETWEEN_SUBS + endGap;
      const contentW = CONFIG.BLOCK_WIDTH - 20;
      const content = groupG
        .append("g")
        .attr("class", "container-group-content")
        .attr("transform", `translate(10, ${HEADER_H + 10})`)
        .style("display", grp.collapsed ? "none" : "block");

      content
        .append("rect")
        .attr("class", "container-group-content-bounds")
        .attr("width", contentW)
        .attr("height", Math.max(1, contentH))
        .attr("fill", "none")
        .attr("pointer-events", "none");

      const subEls = content
        .selectAll(".sub-group")
        .data(grp.subs)
        .join("g")
        .attr("class", "sub-group")
        .attr("transform", (d, i) => `translate(0, ${i * CONFIG.DIST_BETWEEN_SUBS})`)
        .on("mouseleave", (event, d) => outerThis.hideTooltip(event))
        .on("click", (event) => event.stopPropagation());

      subEls
        .append("rect")
        .attr("class", "subgroup-rect")
        .attr("width", CONFIG.BLOCK_WIDTH - 20)
        .attr("height", 25)
        .attr("rx", 3)
        .attr("fill", "#f9f9f9")
        .attr("stroke", "#ddd");

      subEls.each(function (d, i) {
        const subGroup = d3.select(this);
        const existing = imageManager.getBend(d.full);
        if (existing) {
          d.bendType = existing.module_type;
          const t = CONFIG.BENDING_TYPES?.[existing.module_type];
          const key = t?.module_args_key;
          const val = key ? existing.module_args?.[key] : undefined;
          const levels = t?.levels || [0];
          const idx = levels.indexOf(val);
          d.sliderIndex = idx >= 0 ? idx : 0;
        } else {
          d.bendType = d.bendType || "rotate";
          d.sliderIndex = d.sliderIndex ?? 0;
        }

        subGroup
          .append("rect")
          .attr("class", "sub-accent")
          .attr("width", 4)
          .attr("height", 25)
          .attr("x", 0)
          .attr("y", 0)
          .attr("rx", 0)
          .attr("fill", () => catMap[d.category] || "#666");

        const name = (d.name || "");
        const useMarquee = name.length >= (CONFIG.MARQUEE_LABEL_MIN_LEN ?? 24);
        const labelX = 12;
        const labelY = 9;
        const clipW = CONFIG.MARQUEE_CLIP_W ?? 156;
        const offset = CONFIG.MARQUEE_OFFSET ?? 230;
        const dur = CONFIG.MARQUEE_DUR ?? "10s";
        const badgeColor = catMap[d.category] || "#666";

        if (!useMarquee) {
          const labelEl = subGroup
            .append("text")
            .attr("class", "sub-label")
            .attr("x", labelX)
            .attr("y", labelY)
            .style("user-select", "none")
            .style("-webkit-user-select", "none")
            .on("mouseenter", (event, d) => outerThis.showTooltip(event, d))
            .on("mouseleave", (event, d) => outerThis.hideTooltip(event));
          labelEl.append("tspan").text(() => `${d.name} `);
          labelEl
            .append("tspan")
            .attr("class", "sub-type-badge")
            .style("font-size", "9px")
            .style("font-weight", "600")
            .style("fill", () => badgeColor)
            .text(() => ` ${d.category || ""}`);
        } else {
          const clipId = "marquee-clip-" + [block.name, gi, i].join("-").replace(/[^a-z0-9_-]/gi, "-");
          const wrap = subGroup
            .append("g")
            .attr("class", "sub-label")
            .style("user-select", "none")
            .style("-webkit-user-select", "none")
            .on("mouseenter", (event, d) => outerThis.showTooltip(event, d))
            .on("mouseleave", (event, d) => outerThis.hideTooltip(event));
          wrap.append("defs")
            .append("clipPath")
            .attr("id", clipId)
            .append("rect")
            .attr("x", labelX)
            .attr("y", 0)
            .attr("width", clipW)
            .attr("height", 14);
          const marqueeG = wrap.append("g").attr("clip-path", `url(#${clipId})`);
          const txt = marqueeG.append("text").attr("x", 0).attr("y", 0);
          txt.append("animateTransform")
            .attr("attributeName", "transform")
            .attr("type", "translate")
            .attr("from", "0 0")
            .attr("to", `-${offset} 0`)
            .attr("dur", dur)
            .attr("repeatCount", "indefinite");
          txt.append("tspan").attr("x", labelX).attr("y", labelY).text(name);
          txt.append("tspan").attr("x", labelX + offset).attr("y", labelY).text(name);
          wrap
            .append("text")
            .attr("class", "sub-type-badge")
            .attr("x", labelX + clipW)
            .attr("y", labelY)
            .attr("dominant-baseline", "middle")
            .style("font-size", "9px")
            .style("font-weight", "600")
            .style("fill", badgeColor)
            .text(() => ` ${d.category || ""}`);
        }

        const sliderGroup = subGroup
          .append("g")
          .attr("class", "svg-slider")
          .attr("transform", `translate(${sliderX}, ${sliderY})`)
          .on("click", (event) => event.stopPropagation());

        sliderGroup
          .append("line")
          .attr("x1", 0)
          .attr("x2", sliderWidth)
          .attr("stroke", "#ccc")
          .attr("stroke-width", 4)
          .attr("stroke-linecap", "round");

        updateSliderTicks(sliderGroup, d.bendType || "rotate", sliderWidth);

        const cfg = getSliderConfig(d.bendType || "rotate");
        const numStepsInit = (d.bendType || "rotate") ? Math.max(1, cfg.levels.length - 1) : 1;
        const initX = ((d.sliderIndex ?? cfg.defaultIndex) / numStepsInit) * sliderWidth;
        const handle = sliderGroup
          .append("circle")
          .attr("r", 5)
          .attr("cx", initX)
          .attr("fill", "#555")
          .attr("stroke", "#fff")
          .attr("stroke-width", 1.5)
          .attr("cursor", "pointer");

        const startInteraction = (event) => {
          const domEvent = event.sourceEvent || event;
          domEvent?.stopPropagation();
          outerThis.handleSubLabelClick(domEvent, d, g, svg);
          if (d.decayTimer) d.decayTimer.stop();
          if (!window.comfySessionId) {
            allSubGroups().each(function (otherData) {
              if (d === otherData) return;
              if (otherData.decayTimer) otherData.decayTimer.stop();
              otherData.bendType = "rotate";
              otherData.sliderIndex = 0;
              const sg = d3.select(this);
              sg.select(".svg-slider circle").transition().duration(200).attr("cx", 0);
              UNetVisualizer.updateSliderTicks(sg.select(".svg-slider"), "rotate", sliderWidth);
              UNetVisualizer.updateSliderValueDisplay(sg.select(".slider-value-text"), "rotate", 0);
            });
          }
        };

        const updateValue = (x) => {
          const cfg = getSliderConfig(d.bendType);
          const levels = cfg.levels;
          if (!d.bendType || levels.length <= 1) return;
          const numSteps = Math.max(1, levels.length - 1);
          x = Math.max(0, Math.min(sliderWidth, x));
          const idx = Math.round((x / sliderWidth) * numSteps);
          const clampedIdx = Math.max(0, Math.min(idx, levels.length - 1));
          const snappedX = (clampedIdx / numSteps) * sliderWidth;
          handle.attr("cx", snappedX);
          if (d.sliderIndex !== clampedIdx) {
            d.sliderIndex = clampedIdx;
            const val = levels[clampedIdx];
            const key = CONFIG.BENDING_TYPES?.[d.bendType]?.module_args_key;
            if (window.comfySessionId && key) {
              imageManager.setBend(d.full, d.bendType, { [key]: val });
              imageManager.updateBends();
              outerThis.updateRubberEffect(imageManager.getBendingIntensity());
              if (typeof outerThis.updateBlockHighlights === "function") outerThis.updateBlockHighlights();
            } else {
              imageManager.setLayerName(d.full);
              imageManager.updateImage(val);
            }
          }
          updateSliderValueDisplay(valueText, d.bendType, levels[clampedIdx] ?? levels[0]);
        };

        const endInteraction = () => {
          if (d.decayTimer) d.decayTimer.stop();
          imageManager.commitBendsIfLive();
        };

        const drag = d3.drag().on("start", startInteraction).on("drag", (e) => updateValue(e.x)).on("end", endInteraction);
        handle.call(drag);
        sliderGroup.on("click", (event) => {
          event.stopPropagation();
          startInteraction(event);
          updateValue(d3.pointer(event)[0]);
          endInteraction();
        });

        const valueText = subGroup
          .append("text")
          .attr("class", "slider-value-text")
          .attr("x", sliderX + sliderWidth + 4)
          .attr("y", sliderY + 5)
          .attr("dominant-baseline", "middle")
          .style("font-size", "11px")
          .style("fill", "#555")
          .style("pointer-events", "none");
        updateSliderValueDisplay(valueText, d.bendType || "rotate", cfg.levels[d.sliderIndex ?? cfg.defaultIndex] ?? cfg.levels[0]);

        const fo = subGroup
          .append("foreignObject")
          .attr("x", dropdownX)
          .attr("y", 4)
          .attr("width", dropdownWidth)
          .attr("height", 18);
        const foBody = fo.append("xhtml:body")
          .style("margin", "0")
          .style("padding", "0")
          .style("background", "transparent");
        const sel = foBody.append("xhtml:select")
          .attr("class", "bend-type-select")
          .style("width", "100%")
          .style("height", "16px")
          .style("font-size", "11px")
          .style("cursor", "pointer");
        Object.entries(CONFIG.BENDING_TYPES || {}).forEach(([k, v]) => {
          sel.append("xhtml:option").attr("value", k).text(`${v.symbol} ${v.label}`);
        });
        sel.property("value", d.bendType || "rotate");
        sel.on("change", function () {
          const v = this.value || "rotate";
          d.bendType = v;
          const cfg = getSliderConfig(d.bendType);
          d.sliderIndex = cfg.defaultIndex;
          const val = cfg.levels[cfg.defaultIndex];
          const key = CONFIG.BENDING_TYPES?.[v]?.module_args_key;
          if (window.comfySessionId && key) {
            imageManager.setBend(d.full, v, { [key]: val });
          } else {
            imageManager.setLayerName(d.full);
            imageManager.updateImage(val);
          }
          imageManager.updateBends();
          updateSliderTicks(subGroup.select(".svg-slider"), d.bendType, sliderWidth);
          const numSteps = Math.max(1, cfg.levels.length - 1);
          const x = (d.sliderIndex / numSteps) * sliderWidth;
          subGroup.select(".svg-slider circle").attr("cx", x);
          updateSliderValueDisplay(subGroup.select(".slider-value-text"), d.bendType, val);
          outerThis.updateRubberEffect(imageManager.getBendingIntensity());
          if (typeof outerThis.updateBlockHighlights === "function") outerThis.updateBlockHighlights();
          imageManager.commitBendsIfLive();
        });
      });
    });

    layer.attr("display", "none");

    blockResetBtn.on("click", (event) => {
      event.stopPropagation();
      block.subs.forEach((sub) => imageManager.setBend(sub.full, 0));
      allSubGroups().each(function (d) {
        const sg = d3.select(this);
        if (d) d.bendType = "rotate", d.sliderIndex = 0;
        const sel = sg.select(".bend-type-select").node();
        if (sel) sel.value = "rotate";
        sg.select(".svg-slider circle").attr("cx", 0);
        UNetVisualizer.updateSliderTicks(sg.select(".svg-slider"), "rotate", sliderWidth);
        UNetVisualizer.updateSliderValueDisplay(sg.select(".slider-value-text"), "rotate", 0);
      });
      imageManager.updateBends();
      outerThis.updateRubberEffect(0);
      if (typeof outerThis.updateBlockHighlights === "function") outerThis.updateBlockHighlights();
      imageManager.commitBendsIfLive();
    });

    const scrollBarWidth = 8;
    const scrollTrack = g
      .append("rect")
      .attr("class", "block-scroll-track")
      .attr("x", CONFIG.BLOCK_WIDTH - scrollBarWidth - 2)
      .attr("y", CONFIG.COLLAPSED_HEIGHT)
      .attr("width", scrollBarWidth)
      .attr("height", Math.max(0, block._expandedHeight - CONFIG.COLLAPSED_HEIGHT))
      .attr("fill", "rgba(0,0,0,0.1)")
      .attr("rx", 4)
      .attr("display", "none");
    const scrollThumb = g
      .append("rect")
      .attr("class", "block-scroll-thumb")
      .attr("x", CONFIG.BLOCK_WIDTH - scrollBarWidth - 2)
      .attr("y", CONFIG.COLLAPSED_HEIGHT)
      .attr("width", scrollBarWidth)
      .attr("height", 20)
      .attr("fill", "rgba(0,0,0,0.35)")
      .attr("rx", 4)
      .attr("display", "none");

    block._scrollOffset = 0;
    function updateScrollUi() {
      const trackH = Math.max(0, block._expandedHeight - CONFIG.COLLAPSED_HEIGHT);
      const maxScroll = Math.max(0, block._fullContentH - trackH);
      scrollTrack.attr("height", trackH);
      const thumbH = Math.max(20, block._fullContentH <= 0 ? trackH : trackH * (trackH / block._fullContentH));
      scrollThumb.attr("height", thumbH);
      block._scrollOffset = Math.max(0, Math.min(block._scrollOffset, maxScroll));
      subsContainer.attr("transform", `translate(0, ${-block._scrollOffset})`);
      const t = maxScroll <= 0 ? 0 : block._scrollOffset / maxScroll;
      const thumbY = CONFIG.COLLAPSED_HEIGHT + t * (trackH - thumbH);
      scrollThumb.attr("y", thumbY);
    }

    g.on("wheel.block-scroll", function (event) {
      if (outerThis.openedBlockGroup?.node() !== g.node()) return;
      event.preventDefault();
      const maxScroll = Math.max(0, block._fullContentH - (block._expandedHeight - CONFIG.COLLAPSED_HEIGHT));
      block._scrollOffset = Math.max(0, Math.min(block._scrollOffset + event.deltaY, maxScroll));
      updateScrollUi();
    });
    const showScrollbar = () => {
      scrollTrack.attr("display", block._canScroll ? "block" : "none");
      scrollThumb.attr("display", block._canScroll ? "block" : "none");
    };
    const hideScrollbar = () => {
      scrollTrack.attr("display", "none");
      scrollThumb.attr("display", "none");
    };
    g.on("open.block-scroll", () => {
      showScrollbar();
      updateScrollUi();
    });
    g.on("close.block-scroll", hideScrollbar);

    block._scrollRefs = { updateScrollUi, showScrollbar, hideScrollbar };
    this.setupClickBehavior(g, rect, layer, () => block._expandedHeight, { subsContainer, updateScrollUi, showScrollbar, hideScrollbar });
  }

  /**
   * Update layout after container group expand/collapse: group positions, content visibility,
   * chevrons, block rect height, clip rect, scroll UI.
   */
  updateContainerGroupLayout(g, block) {
    const HEADER_H = CONFIG.CONTAINER_HEADER_HEIGHT || 22;
    const layout = block._layout || [];
    const containerGroups = block._containerGroups || [];
    const layer = g.select(".container-groups-layer");
    const groups = layer.selectAll(".container-group").data(containerGroups);

    groups.attr("transform", (grp, i) => {
      const l = layout[i];
      return l ? `translate(0, ${l.y})` : "translate(0,0)";
    });

    groups.select(".container-group-content").each(function (grp) {
      d3.select(this).style("display", grp.collapsed ? "none" : "block");
    });
    groups.select(".container-header-chevron").text((grp) => (grp.collapsed ? "▶" : "▼"));

    const blockRect = g.select(".block-main-rect");
    if (!blockRect.empty()) blockRect.attr("height", block._expandedHeight);

    const clipRect = g.select(".block-clip-rect");
    if (!clipRect.empty()) clipRect.attr("height", block._expandedHeight);

    const refs = block._scrollRefs;
    if (refs?.updateScrollUi) refs.updateScrollUi();
    if (refs?.showScrollbar) refs.showScrollbar();

    if (typeof this.updateBlockHighlights === "function") this.updateBlockHighlights();
  }

  /**
   * Draw a single block
   * @param {Object} svg - D3 SVG selection
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {Object} block - Block data
   */
  drawBlock2(svg, x, y, block) {
    const g = svg
      .append("g")
      .attr("name", block.name)
      .attr("transform", `translate(${x},${y})`);
    this.blockPositions[block.name] = { x, y };

    const hasSkip = Utils.hasSkipConnection(block);

    const expandedHeight =
      CONFIG.COLLAPSED_HEIGHT + block.subs.length * CONFIG.DIST_BETWEEN_SUBS;

    const rect = g
      .append("rect")
      .attr("width", CONFIG.BLOCK_WIDTH)
      .attr("height", CONFIG.COLLAPSED_HEIGHT)
      .attr("rx", 4)
      .attr("fill", CONFIG.COLORS.BLOCK_FILL)
      .attr("stroke", "#333")
      .attr("stroke-width", 1.5);

    g.append("text")
      .attr("x", 55)
      .attr("y", 18)
      .attr("class", "label")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .text(block.name + (hasSkip ? " ↔" : ""));

    const catMap = UNET_BLOCKS[imageManager.getModelName()]?.blockTypeCategories
      || UNET_BLOCKS["ComfyUI"]?.blockTypeCategories
      || UNET_BLOCKS["SD1.4"]?.blockTypeCategories
      || {};
    const subGroups = g
      .selectAll(".sub-group")
      .data(block.subs)
      .join("g")
      .attr("class", "sub-group")
      .attr(
        "transform",
        (d, i) =>
          `translate(10, ${
            CONFIG.COLLAPSED_HEIGHT + i * CONFIG.DIST_BETWEEN_SUBS
          })`
      )
      .attr("display", "none")
      .on("click", (event, d) => event.stopPropagation())
      .on("mouseleave", (event, d) => this.hideTooltip(event));

    subGroups
      .append("rect")
      .attr("class", "subgroup-rect")
      .attr("width", CONFIG.BLOCK_WIDTH - 20)
      .attr("height", 25)
      .attr("rx", 3)
      .attr("fill", "#f9f9f9")
      .attr("stroke", "#ddd");

    let outerThis = this;

    subGroups.each(function (d, i) {
      d.sliderValue = 0;
      const subGroup = d3.select(this);
      subGroup
        .append("rect")
        .attr("class", "sub-accent")
        .attr("width", 4)
        .attr("height", 25)
        .attr("x", 0)
        .attr("y", 0)
        .attr("rx", 0)
        .attr("fill", () => catMap[d.category] || "#666");

      const subtexts = subGroup
        .append("text")
        .attr("class", "sub-label")
        .attr("x", 12)
        .attr("y", 9)
        .text((d) => `• ${d.name}`)
        .style("user-select", "none")
        .style("-webkit-user-select", "none")
        .on("mouseenter", (event, d) => outerThis.showTooltip(event, d))
        .on("mouseleave", (event, d) => outerThis.hideTooltip(event));

      // Add sliders for each subgroup
      const sliders = d3
        .select(this)
        .append("foreignObject")
        .attr("x", 5)
        .attr("y", 12)
        .attr("width", (3 * CONFIG.BLOCK_WIDTH) / 4)
        .attr("height", 12)
        .append("xhtml:body")
        .style("margin", "0px")
        .style("padding", "0px")
        .style("background-color", "transparent")
        .append("xhtml:input")
        .attr("type", "range")
        .attr("min", 0)
        .attr("max", 3)
        .attr("step", 1)
        .attr("value", 0)
        .style("width", (3 * CONFIG.BLOCK_WIDTH) / 4 + "px")
        .style("height", "8px")
        .style("cursor", "pointer")

        .on("click", (event) => outerThis.handleSubLabelClick(event, d, g, svg))
        .on("input", function (event, d) {
          if (this.decayTimer) this.decayTimer.stop();
          const sliderThis = this;
          if (!window.comfySessionId) {
            subGroups.selectAll("input").each(function () {
              if (this !== sliderThis) {
                this.value = 0;
                outerThis.updateRubberEffect(0);
                this.decayTimer?.stop();
              }
            });
          }
          const value = +this.value;
          d.sliderValue = value;
          const angle = 90 * value;
          if (window.comfySessionId) {
            imageManager.setBend(d.full, angle);
            imageManager.updateBends();
            const maxAngle = imageManager.getBendingAngle();
            outerThis.updateRubberEffect(maxAngle / 270);
            if (typeof outerThis.updateBlockHighlights === "function") outerThis.updateBlockHighlights();
          } else {
            imageManager.setLayerName(d.full);
            imageManager.updateImage(angle);
          }
          console.log(`Slider for ${d.full} changed to: ${value}`);
        })
        .on("change", function (event, d) {
          event.stopPropagation();
          if (this.decayTimer) this.decayTimer.stop();
        });
    });

    //this.createTooltip();
    //this.setupHoldBehavior(subtexts, g);
    //this.setupDragBehavior(subtexts, g);

    this.setupClickBehavior(g, rect, subGroups, expandedHeight);
  }

  /**
   * Handle sub-label click events. Works with block→container→sub (drawBlock) and
   * block→sub (drawBlock2). Only touches block-level groups; never mutates
   * container headers or container/sub visibility. Keeps openedBlockGroup and
   * scroll/clip in sync.
   * @param {Event} event - Click event
   * @param {Object} d - Data object
   * @param {Object} g - D3 block group selection
   * @param {Object} svg - D3 SVG selection
   */
  handleSubLabelClick(event, d, g, svg) {
    event.stopPropagation();
    const self = this;
    const block = g.datum();
    const hasContainers = !g.select(".container-groups-layer").empty();

    // 1. Collapse all other blocks (only g[name]); never touch container structure. Skip current.
    const currentBlockNode = g.node();
    svg.selectAll("g[name]").each(function () {
      const bg = d3.select(this);
      if (bg.node() === currentBlockNode) return;

      const layer = bg.select(".container-groups-layer");
      const hasLayer = !layer.empty();

      if (hasLayer) {
        const mainRect = bg.select(".block-main-rect");
        if (!mainRect.empty()) {
          mainRect.transition().duration(200).attr("height", CONFIG.COLLAPSED_HEIGHT).style("fill", CONFIG.COLORS.BLOCK_FILL);
        }
        layer.transition().duration(200).style("display", "none");
      } else {
        const r = bg.select("rect");
        if (!r.empty()) r.transition().duration(200).attr("height", CONFIG.COLLAPSED_HEIGHT).style("fill", CONFIG.COLORS.BLOCK_FILL);
        bg.selectAll(".sub-group").transition().duration(200).style("display", "none");
      }

      if (self.openedBlockGroup && self.openedBlockGroup.node() === bg.node()) {
        const ob = bg.datum();
        if (ob?._scrollRefs?.hideScrollbar) ob._scrollRefs.hideScrollbar();
        const clip = bg.select(".block-clip-rect");
        if (!clip.empty()) clip.attr("height", CONFIG.COLLAPSED_HEIGHT);
        if (bg.node()) bg.node().dispatchEvent(new Event("close.block-scroll", { bubbles: true }));
        self.openedBlockGroup = null;
      }
    });

    svg.selectAll(".sub-group").classed("selected", false);

    const parentSubGroup = event.target.closest(".sub-group");
    if (!parentSubGroup) return;
    const label = parentSubGroup.querySelector(".sub-label");
    const accent = parentSubGroup.querySelector(".sub-accent");
    const rect = parentSubGroup.querySelector(".subgroup-rect");
    const slider = parentSubGroup.querySelector(".svg-slider > line");

    d3.select(parentSubGroup).classed("selected", true);
    this.selectedSubGroup = parentSubGroup;
    this.selectedGroup = g;

    // 2. Expand current block (g); use layer/subs per structure, no container mutation
    const expH = block?._expandedHeight ?? (CONFIG.COLLAPSED_HEIGHT + (block?.subs?.length || 0) * CONFIG.DIST_BETWEEN_SUBS);
    if (hasContainers) {
      const mainRect = g.select(".block-main-rect");
      if (!mainRect.empty()) mainRect.transition().duration(200).attr("height", expH).style("fill", CONFIG.COLORS.BLOCK_HIGHLIGHT);
      g.select(".container-groups-layer").transition().duration(200).style("display", "block");
      const clip = g.select(".block-clip-rect");
      if (!clip.empty()) clip.attr("height", expH);
    } else {
      g.select("rect").transition().duration(200).attr("height", expH).style("fill", CONFIG.COLORS.BLOCK_HIGHLIGHT);
      g.selectAll(".sub-group").transition().duration(200).style("display", "block");
    }
    g.raise();
    this.openedBlockGroup = g;
    if (g.node()) g.node().dispatchEvent(new Event("open.block-scroll", { bubbles: true }));

    imageManager.setLayerName(d.full);
    const bend = imageManager.getBend(d.full);
    const angleForUpdate = (bend?.module_type === "rotate") ? (bend.module_args?.angle_degrees ?? 0) : 0;
    imageManager.updateImage(angleForUpdate, true);

    if (typeof this.updateBlockHighlights === "function") this.updateBlockHighlights();

    this.updateSkipLines(svg, d, g);
  }

  /**
   * Apply rubber text effect to a text element
   * @param {Element} element - The text element to apply effect to
   */
  applyRubberEffectToElement(element, elementSelector, currentGroup) {
    // Remove effect from previously selected text
    const outerThis = this;
    currentGroup.selectAll(".sub-group").each(function (group) {
      //console.log(group, d3.select(this), d3.select(this).select(".sub-label"));
      outerThis.removeRubberEffectFromElement(d3.select(this).node());
      const n = d3.select(this).select(elementSelector);
      outerThis.removeRubberEffectFromElement(n.node());
    });

    // Apply effect to current text
    element.setAttribute("filter", "url(#fx-rubber)");
    this.activeEffects.set(element, true);

    // Animate the effect
    this.animateRubberEffect();
  }

  /**
   * Remove rubber text effect from a text element
   * @param {Element} element - The text element to remove effect from
   */
  removeRubberEffectFromElement(element) {
    if (!element) return;
    element.removeAttribute("filter");
    this.activeEffects.delete(element);
  }

  /**
   * Animate the rubber effect
   */
  animateRubberEffect() {
    // Reset scale to 0
    this.updateRubberEffect(0);

    // Animate to scale 1 and back
    let scale = 0;
    let direction = 1;
    const step = 0.05;

    const animate = () => {
      scale += step * direction;

      if (scale >= 1) {
        direction = -1;
        scale = 1;
      } else if (scale <= 0) {
        scale = 0;
        return; // Stop animation
      }

      this.updateRubberEffect(scale);
      requestAnimationFrame(animate);
    };

    animate();
  }

  /**
   * Create tooltip container
   */
  createTooltip() {
    if (!document.getElementById("tooltip")) {
      const tooltip = document.createElement("div");
      tooltip.id = "tooltip";
      tooltip.style.cssText = `
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-family: monospace;
                pointer-events: none;
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.2s ease;
                max-width: 300px;
                word-wrap: break-word;
            `;
      document.body.appendChild(tooltip);
    }
  }

  /**
   * Show tooltip with block type information
   * @param {Event} event - Mouse event
   * @param {Object} d - Data object
   */
  showTooltip(event, d) {
    const tooltip = document.getElementById("tooltip");
    if (!tooltip) return;
    let blockType = "—";
    try {
      if (d.type && d.type.indexOf("Conv2d") >= 0) blockType = Utils.extractBlockTypeName(d.type);
      else blockType = d.type || "—";
    } catch (_) {
      blockType = d.type || "—";
    }
    const container = d.container != null ? d.container : "—";
    const category = d.category != null ? d.category : "—";
    tooltip.innerHTML = `
            <strong>${d.name}</strong><br>
            <em>Container:</em> ${container}<br>
            <em>Category:</em> ${category}<br>
            <em>Type:</em> ${blockType}<br>
            <em>Full Path:</em> ${d.full}
        `;

    tooltip.style.left = event.pageX + 50 + "px";
    tooltip.style.top = event.pageY - 10 + "px";
    tooltip.style.opacity = "1";
  }

  /**
   * Hide tooltip
   * @param {Event} event - Mouse event
   */
  hideTooltip(event) {
    console.log("HIDE TOOLTIP");
    const tooltip = document.getElementById("tooltip");
    if (tooltip) {
      tooltip.style.opacity = "0";
    }
  }

  setupClickBehavior(g, rect, contentToToggle, expandedHeight, scrollRefs) {
    g.on("click", (event) => {
      const isAlreadyOpen =
        this.openedBlockGroup && this.openedBlockGroup.node() === g.node();
      const expH = typeof expandedHeight === "function" ? expandedHeight() : expandedHeight;

      const updateClip = (sel, h) => {
        const clip = sel.select(".block-clip-rect");
        if (!clip.empty()) clip.attr("height", h);
      };

      if (this.openedBlockGroup && !isAlreadyOpen) {
        const oldG = this.openedBlockGroup;
        const oldRect = oldG.select(".block-main-rect");
        const oldLayer = oldG.select(".container-groups-layer");

        oldRect
          .transition()
          .duration(200)
          .attr("height", CONFIG.COLLAPSED_HEIGHT)
          .style("fill", CONFIG.COLORS.BLOCK_FILL);

        oldLayer.transition().duration(200).style("display", "none");

        const oldBlock = oldG.datum();
        if (oldBlock?._scrollRefs?.hideScrollbar) oldBlock._scrollRefs.hideScrollbar();
        updateClip(oldG, CONFIG.COLLAPSED_HEIGHT);
        if (oldG.node()) oldG.node().dispatchEvent(new Event("close.block-scroll", { bubbles: true }));
      }

      if (!isAlreadyOpen) {
        rect
          .transition()
          .duration(200)
          .attr("height", expH)
          .style("fill", CONFIG.COLORS.BLOCK_HIGHLIGHT);

        contentToToggle.transition().duration(200).style("display", "block");

        updateClip(g, expH);
        g.raise();
        this.openedBlockGroup = g;
        if (g.node()) g.node().dispatchEvent(new Event("open.block-scroll", { bubbles: true }));
      } else {
        rect
          .transition()
          .duration(200)
          .attr("height", CONFIG.COLLAPSED_HEIGHT)
          .style("fill", CONFIG.COLORS.BLOCK_FILL);

        contentToToggle.transition().duration(200).style("display", "none");

        if (scrollRefs?.hideScrollbar) scrollRefs.hideScrollbar();
        updateClip(g, CONFIG.COLLAPSED_HEIGHT);
        this.openedBlockGroup = null;
        if (g.node()) g.node().dispatchEvent(new Event("close.block-scroll", { bubbles: true }));
      }
    });
  }

  setupHoldBehavior(subtexts, g) {
    let holdTimer = null;
    let holdInterval = null;
    let isHolding = false;
    let holdStartTime = 0;
    let currentAngle = 0;

    const startHold = (event, d) => {
      console.log("Hold started", event, d);
      this.dragStartY = event.y;
      this.dragStartAngle = imageManager.getBendingAngle();
      isHolding = false;
      holdStartTime = Date.now();

      const holdUpdate = (bypass = false) => {
        if (!isHolding && !bypass) return;

        const holdDuration = Date.now() - holdStartTime;
        var progress = Math.min(holdDuration / CONFIG.HOLD_DURATION, 1); // HOLD_DURATION in ms
        if (bypass) {
          progress = 0.2;
          isHolding = true;
        }

        // Calculate angle based on hold duration
        const raw = Utils.clamp(
          this.dragStartAngle + progress * 270, // Progress from 0 to 270 degrees
          0,
          270
        );

        currentAngle = Utils.snapAngle(raw);
        d3.select(this.selectedSubGroup).classed(
          "selected",
          currentAngle !== 0
        );
        imageManager.updateImage(currentAngle);

        // Update rubber effect based on angle
        const effectScale = currentAngle / 270;
        this.updateRubberEffect(effectScale);
      };

      // Start a timer to detect when holding begins
      holdTimer = setTimeout(() => {
        isHolding = true;
        console.log("Hold detected, starting angle progression");

        // Start continuous angle progression
        holdInterval = setInterval(() => {
          holdUpdate();
        }, CONFIG.HOLD_INTERVAL); // Update every HOLD_INTERVAL ms
      }, CONFIG.HOLD_DELAY); // Delay before hold is detected

      holdUpdate(true); // Initial update to set angle immediately
    };

    const stopHold = () => {
      if (holdTimer) {
        clearTimeout(holdTimer);
        holdTimer = null;
      }

      if (holdInterval) {
        clearInterval(holdInterval);
        holdInterval = null;
      }

      if (isHolding) {
        isHolding = false;

        // Spring back to 0
        d3.transition()
          .duration(CONFIG.HOLD_RELEASE_DURATION)
          .tween("unbend", () => {
            const interp = d3.interpolate(currentAngle, 0);
            return (t) => {
              const angle = Utils.snapAngle(interp(t));
              d3.select(this.selectedSubGroup).classed("selected", angle !== 0);
              imageManager.updateImage(angle);
              const effectScale = angle / 270;
              this.updateRubberEffect(effectScale);
            };
          })
          .on("end", () => {
            imageManager.commitBendsIfLive();
          });
      }
    };

    // Mouse events
    subtexts
      .on("mousedown", startHold)
      .on("mouseup", stopHold)
      .on("mouseleave", stopHold)
      .on("touchstart", (event) => {
        event.preventDefault();
        startHold(event, event.target.__data__);
      })
      .on("touchend", (event) => {
        event.preventDefault();
        stopHold();
      })
      .on("touchcancel", (event) => {
        event.preventDefault();
        stopHold();
      })
      .attr("cursor", "pointer"); // Changed from ns-resize to pointer

    // Store stopHold method for external access if needed
    this.stopHold = stopHold;
  }

  /**
   * Set up drag behavior for sub-labels
   * @param {Object} subtexts - D3 selection of sub-labels
   * @param {Object} g - D3 group selection
   */
  setupDragBehavior(subtexts, g) {
    const dragBehavior = d3
      .drag()
      .on("start", (event, d) => {
        this.dragStartY = event.y;
        this.dragStartAngle = imageManager.getBendingAngle();
        this.hasDragged = false;
      })
      .on("drag", (event, d) => {
        this.hasDragged = true;
        const dy = Math.abs(event.y - this.dragStartY);
        const raw = Utils.clamp(
          this.dragStartAngle + dy * CONFIG.DRAG_SENSITIVITY,
          0,
          270
        );
        const angle = Utils.snapAngle(raw);

        d3.select(this.selectedSubGroup).classed("selected", angle !== 0);

        imageManager.updateImage(angle);

        // Update rubber effect based on angle
        const effectScale = angle / 270; // Normalize to 0-1
        this.updateRubberEffect(effectScale);
      })
      .on("end", (event, d) => {
        if (!this.hasDragged) return;
        this.hasDragged = false;

        // Spring back to 0
        d3.transition()
          .duration(CONFIG.ANIMATION_DURATION)
          .tween("unbend", () => {
            const interp = d3.interpolate(imageManager.getBendingAngle(), 0);
            return (t) => {
              const angle = Utils.snapAngle(interp(t));
              d3.select(this.selectedSubGroup).classed("selected", angle !== 0);
              imageManager.updateImage(angle);
              const effectScale = angle / 270;
              this.updateRubberEffect(effectScale);
            };
          })
          .on("end", () => {
            imageManager.commitBendsIfLive();
          });
      });

    subtexts.attr("cursor", "ns-resize").call(dragBehavior);
  }

  /**
   * Set up hover behavior for blocks
   * @param {Object} g - D3 group selection
   * @param {Object} rect - D3 rect selection
   * @param {Object} subGroups - D3 subGroups selection
   * @param {number} expandedHeight - Expanded height
   */
  setupHoverBehavior(g, rect, subGroups, expandedHeight) {
    g.on("mouseenter", () => {
      rect
        .transition()
        .duration(200)
        .attr("height", expandedHeight)
        .style("fill", CONFIG.COLORS.BLOCK_HIGHLIGHT);
      subGroups.transition().duration(200).style("opacity", 1);
      g.raise();
    });

    g.on("mouseleave", (event) => {
      const related = event.relatedTarget;
      if (this.selectedSubGroup && g.node().contains(related)) return;
      if (!this.selectedSubGroup || !g.node().contains(this.selectedSubGroup)) {
        rect
          .transition()
          .duration(200)
          .attr("height", CONFIG.COLLAPSED_HEIGHT)
          .style("fill", CONFIG.COLORS.BLOCK_FILL)
          .on("end", () => {
            if (this.selectedGroup) {
              this.selectedGroup.raise();
            }
          });
        subGroups.transition().duration(200).style("opacity", 0);
      }
    });
  }

  /**
   * Update skip lines highlighting
   * @param {Object} svg - D3 SVG selection
   * @param {Object} d - Data object
   * @param {Object} currentGroup - Current group selection
   */
  updateSkipLines(svg, d, currentGroup) {
    if (d.name.includes("skip_connection")) {
      console.log(
        "Highlighting skip lines for block:",
        svg.selectAll("line.skip-line")
      );

      const base = currentGroup.attr("name");

      svg.selectAll("line.skip-line").classed("highlighted", function () {
        const f = d3.select(this).attr("data-from");
        const t = d3.select(this).attr("data-to");
        console.log(
          "Checking skip line:",
          f,
          t,
          "against base:",
          base,
          currentGroup
        );
        return f === base || t === base;
      });
    } else {
      svg.selectAll("line.skip-line").classed("highlighted", false);
    }
  }

  /**
   * Draw inter-block flow lines with arrowheads (execution order: input → middle → output).
   * @param {Object} svg - D3 SVG selection (zoom container)
   * @param {Array} input - Input blocks
   * @param {Array} middle - Middle blocks
   * @param {Array} output - Output blocks
   */
  drawFlowArrows(svg, input, middle, output) {
    const ordered = [...(input || []), ...(middle || []), ...(output || [])];
    if (ordered.length < 2) return;

    const cx = (pos) => pos.x + CONFIG.BLOCK_WIDTH / 2;
    const cy = (pos) => pos.y + CONFIG.COLLAPSED_HEIGHT / 2;

    for (let i = 0; i < ordered.length - 1; i++) {
      const a = ordered[i];
      const b = ordered[i + 1];
      const from = this.blockPositions[a.name];
      const to = this.blockPositions[b.name];
      if (!from || !to) continue;
      // Draw line reversed so arrow points backward along flow (to → from)
      svg
        .append("line")
        .attr("x1", cx(to))
        .attr("y1", cy(to))
        .attr("x2", cx(from))
        .attr("y2", cy(from))
        .attr("class", "flow-line")
        .attr("stroke", "#666")
        .attr("stroke-width", 3.5)
        .attr("stroke-dasharray", "6 4")
        .attr("stroke-opacity", 0.55);
    }
  }

  /**
   * Draw skip lines between blocks
   * @param {Object} svg - D3 SVG selection
   * @param {Array} input - Input blocks
   * @param {Array} output - Output blocks
   */
  drawSkipLines(svg, input, output) {
    // Flux does not use SD-style skip pairing; only draw for SD
    if (window.webBendDemoModelType === "flux") return;
    input.forEach((block, i) => {
      const inputHasSkip = Utils.hasSkipConnection(block);
      const outputBlock = output[11 - i];
      const outputHasSkip = outputBlock && Utils.hasSkipConnection(outputBlock);

      if (outputHasSkip) {
        const from = this.blockPositions[block.name];
        const to = this.blockPositions[outputBlock.name];
        if (from && to) {
          svg
            .append("line")
            .attr("x1", from.x + CONFIG.BLOCK_WIDTH)
            .attr("y1", from.y + CONFIG.COLLAPSED_HEIGHT / 2)
            .attr("x2", to.x)
            .attr("y2", to.y + CONFIG.COLLAPSED_HEIGHT / 2)
            .attr("class", "skip-line")
            .attr("data-from", block.name)
            .attr("data-to", outputBlock.name);
        }
      }
    });
  }
}

// Global instance
const unetVisualizer = new UNetVisualizer();
