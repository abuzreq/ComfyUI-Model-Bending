// Main Application
class DiffusionModelApp {
  constructor() {
    this.svgDiagram = null;
    this.initialized = false;
    this.infoPanelState = this.loadInfoPanelState();
    this.legendState = this.loadLegendState();
  }

  /**
   * Initialize the application
   */
  init() {
    if (this.initialized) return;

    this.loadSVGContent("#svg-diagram");
    this.svgDiagram = document.getElementById("svg-diagram");
    if (!this.svgDiagram) {
      console.error("SVG diagram element not found");
      return;
    }
    this.setupEventListeners();
    this.setupInfoPanelToggle();
    this.setupLegendToggle();
    this.setupMainResizer();
    this.setupDiagramResizer();

    this.initialized = true;

    // Configuration object
    const blockTypeCategories = {
      Conv2d: "#666666",
      "ResBlock.Sequential.Conv2d": "#1f77b4",
      "ResBlock.Conv2d": "#9467bd",
      "SpatialTransformer.Conv2d": "#ff7f0e",
      "Upsample.Conv2d": "#2ca02c",
      "Downsample.Conv2d": "#d62728",
    };

    // Function to update legend counts
    function updateLegendCounts(counts) {
      Object.entries(counts).forEach(([category, count]) => {
        const legendItem = document.querySelector(
          `[data-category="${category}"]`
        );
        if (legendItem) {
          const countElement = legendItem.querySelector(".legend-count");
          countElement.textContent = `(${count})`;
        }
      });
    }

    // Function to highlight legend items
    function highlightLegendItem(category, highlight = true) {
      const legendItem = document.querySelector(
        `[data-category="${category}"]`
      );
      if (legendItem) {
        if (highlight) {
          legendItem.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
          legendItem.style.fontWeight = "bold";
        } else {
          legendItem.style.backgroundColor = "";
          legendItem.style.fontWeight = "";
        }
      }
    }

    // Add data attributes to legend items for easier selection
    document.addEventListener("DOMContentLoaded", function () {
      const legendItems = document.querySelectorAll(".legend-item");
      const categories = Object.keys(blockTypeCategories);

      legendItems.forEach((item, index) => {
        if (index < categories.length) {
          item.setAttribute("data-category", categories[index]);
        }
      });
    });

    // Example usage:
    updateLegendCounts({
      Conv2d: 15,
      "ResBlock.Sequential.Conv2d": 8,
      "ResBlock.Conv2d": 12,
      "SpatialTransformer.Conv2d": 6,
      "Upsample.Conv2d": 4,
      "Downsample.Conv2d": 3,
    });

    // Example of highlighting a category:
    // highlightLegendItem("Conv2d", true);
  }

  /**
   * Set up the draggable resize handle between unet-expanded and right-column
   */
  setupMainResizer() {
    const resizer = document.getElementById("main-resizer");
    const outer = document.getElementById("outer-container");
    const rightCol = document.getElementById("right-column");
    if (!resizer || !outer || !rightCol) return;

    const MIN_RIGHT = 320;
    const MAX_RIGHT_PERCENT = 0.42;

    const clampRightWidth = (px) => {
      const maxPx = Math.floor(outer.clientWidth * MAX_RIGHT_PERCENT);
      return Math.max(MIN_RIGHT, Math.min(px, maxPx));
    };

    let startX = 0;
    let startWidth = 0;

    const onMouseMove = (e) => {
      const dx = startX - e.clientX;
      const newWidth = clampRightWidth(startWidth + dx);
      rightCol.style.flexBasis = `${newWidth}px`;
    };

    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };

    resizer.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startX = e.clientX;
      startWidth = rightCol.getBoundingClientRect().width;
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
    });
  }

  /**
   * Set up the draggable resize handle between info panel and diagram container.
   * Drag upward to give more space to the diagram (shrink info panel).
   */
  setupDiagramResizer() {
    const resizer = document.getElementById("diagram-resizer");
    const infoPanel = document.getElementById("info-parent-panel");
    const leftPanel = document.getElementById("left-panel");
    if (!resizer || !infoPanel || !leftPanel) return;

    const MIN_INFO_HEIGHT = 80;
    const getMaxInfoHeight = () => {
      const panelRect = leftPanel.getBoundingClientRect();
      return Math.max(MIN_INFO_HEIGHT, Math.floor(panelRect.height - 100));
    };

    const clampHeight = (px) => {
      return Math.max(MIN_INFO_HEIGHT, Math.min(px, getMaxInfoHeight()));
    };

    let startY = 0;
    let startHeight = 0;

    const onMouseMove = (e) => {
      const dy = e.clientY - startY;
      const newHeight = clampHeight(startHeight + dy);
      infoPanel.style.height = `${newHeight}px`;
    };

    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };

    resizer.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startY = e.clientY;
      startHeight = infoPanel.getBoundingClientRect().height;
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
      document.body.style.userSelect = "none";
      document.body.style.cursor = "ns-resize";
    });
  }

  /**
   * Set up event listeners for the application
   */
  setupEventListeners() {
    // Set up click handlers for clickable elements - click to show info panel content
    this.svgDiagram.querySelectorAll("g.clickable").forEach((g) => {
      g.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.handleElementClick(g);
        const id = g.id;
        const hasInfo = document.getElementById(`info-${id}`);
        unetVisualizer.updateInfo(hasInfo ? id : "unet");
      });
    });

    unetVisualizer.updateInfo("unet");
  }

  /**
   * Handle click events on diagram elements
   * @param {Element} element - Clicked element
   */
  handleElementClick(element) {
    if (element.id === "unet") {
    //  unetVisualizer.expandUnet();
    } else {
    //  this.toggleInfoPanel(true); 
    //  unetVisualizer.setupExpandedArea("unet-expanded", element.id);
    }

    const gEl = document.getElementById(element.id);
    const rectEl = gEl.querySelector("rect");
    const targetElement = rectEl ? rectEl : gEl.querySelector("path");
    const expandedSvg = document
      .getElementById("unet-expanded")
      .querySelector("svg");

    if (targetElement && expandedSvg && CONFIG.BENDABLE.includes(element.id)) {
     // svgConnector.linkSVGs(targetElement, expandedSvg);
    }
  }

  /**
   * Load SVG content into the diagram container
   */
  loadSVGContent(svg_selecter) {
    console.log("Creating placeholder SVG diagram");
    // Grab the svg (create if needed)
    const svg = d3
      .select(svg_selecter)
      .attr("preserveAspectRatio", "xMidYMid meet");

    // Wipe any previous content
    svg.selectAll("*").remove();

    // ---------- Defs: arrowheads ----------
    const defs = svg.append("defs");

    defs
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9) // tail position
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", "#000");

    const shimmerGradient = defs
      .append("linearGradient")
      .attr("id", "shimmer-gradient");

    shimmerGradient
      .append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#3a3a3a");

    shimmerGradient
      .append("stop")
      .attr("offset", "50%")
      .attr("stop-color", "#fafafa"); // Highlight color

    shimmerGradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#3a3a3a");

    // Animate the gradient to make it "sweep"
    shimmerGradient
      .append("animate")
      .attr("attributeName", "x1")
      .attr("from", "-100%")
      .attr("to", "200%")
      .attr("dur", "2s")
      .attr("repeatCount", "indefinite");

    shimmerGradient
      .append("animate")
      .attr("attributeName", "x2")
      .attr("from", "0%")
      .attr("to", "300%")
      .attr("dur", "2s")
      .attr("repeatCount", "indefinite");

    // --- 4. Create the Glow Filter ---
    const glowFilter = defs
      .append("filter")
      .attr("id", "glow-filter")
      .attr("x", "-50%") // Filter region attributes to prevent clipping
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");

    // Add the first glow color (cyan)
    glowFilter
      .append("feDropShadow")
      .attr("dx", 0)
      .attr("dy", 0)
      .attr("stdDeviation", 10) // This value will be animated by CSS
      .attr("flood-color", "#03e9f4")
      .attr("class", "glowing-shadow-component"); // Apply CSS class

    // Add the second glow color (magenta)
    glowFilter
      .append("feDropShadow")
      .attr("dx", 0)
      .attr("dy", 0)
      .attr("stdDeviation", 10) // This value will be animated by CSS
      .attr("flood-color", "#f403d1")
      .attr("class", "glowing-shadow-component"); // Apply CSS class

    // ---------- Node data ----------
    // Positions are the TOP-LEFT of each box (except the label-only "Prompt")
    const nodes = [
      { id: "prompt", type: "label", x: 20, y: 22, label: "Prompt" },

      {
        id: "clip",
        type: "rect",
        x: 230,
        y: 5,
        w: 90,
        h: 50,
        r: 12,
        label: "CLIP",
        cls: "clickable",
      },

      {
        id: "text_embeddings",
        type: "rect-dashed",
        x: 222,
        y: 85,
        w: 105,
        h: 40,
        r: 11,
        label: "Text Embeddings",
        cls: "clickable",
      },

      {
        id: "latent_seed_noise",
        type: "rect",
        x: 20,
        y: 170,
        w: 70,
        h: 50,
        r: 11,
        label: "Latent Seed\n(noise)",
        cls: "clickable",
      },
      {
        id: "latents",
        type: "rect",
        x: 125,
        y: 170,
        w: 70,
        h: 50,
        r: 11,
        label: "Latents",
        cls: "clickable",
      },

      {
        id: "unet",
        type: "rect",
        x: 220,
        y: 153,
        w: 110,
        h: 85,
        r: 11,
        label: "U-Net",
        cls: "clickable",
      },

      {
        id: "scheduler",
        type: "rect",
        x: 240,
        y: 250,
        w: 70,
        h: 45,
        r: 9,
        label: "Scheduler",
        cls: "clickable",
      },

      {
        id: "text_conditioned_latents",
        type: "rect",
        x: 360,
        y: 170,
        w: 70,
        h: 50,
        r: 11,
        label: "Text\nConditioned\nLatents",
        cls: "clickable",
      },

      //  VAE box
      {
        id: "variational_autoencoder_vae",
        type: "vae",
        x: 500,
        y: 195,
        w: 80,
        h: 85,
        scale: 2.2,
        label: "Variational\nAutoencoder\n(VAE)",
        cls: "clickable",
      },
    ];

    // Helper lookup
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

    // ---------- Link data ----------
    // Each endpoint uses {id, side} where side ∈ 'left'|'right'|'top'|'bottom'|'center'
    const links = [
      {
        from: { id: "prompt", side: "right" },
        to: { id: "clip", side: "left" },
        dashed: false,
      },

      {
        from: { id: "clip", side: "bottom" },
        to: { id: "text_embeddings", side: "top" },
        dashed: false,
      },

      {
        from: { id: "latent_seed_noise", side: "right" },
        to: { id: "latents", side: "left" },
        dashed: false,
      },

      {
        from: { id: "latents", side: "right" },
        to: { id: "unet", side: "left" },
        dashed: true,
      },

      // Conditioning paths as dashed (animated)
      {
        from: { id: "text_embeddings", side: "bottom" },
        to: { id: "unet", side: "top" },
        dashed: false,
      },
      {
        from: { id: "unet", side: "right" },
        to: { id: "text_conditioned_latents", side: "left" },
        dashed: true,
      },

      // Output
      {
        from: { id: "text_conditioned_latents", side: "right" },
        to: { id: "variational_autoencoder_vae", side: "left" },
        dashed: false,
      },
      {
        from: { id: "text_conditioned_latents", side: "bottom" },
        to: { id: "scheduler", side: "right" },
        dashed: true,
        elbow: { x: 395, y: 273 },
      },

      // Step control
      {
        from: { id: "scheduler", side: "left" },
        to: { id: "latents", side: "bottom" },
        dashed: true,
        elbow: { x: 160, y: 273 },
      },
    ];

    // ---------- Zoom container for pan/zoom ----------
    const zoomContainer = svg.append("g").attr("class", "diagram-zoom-container");

    // ---------- Node renderers ----------
    const gNodes = zoomContainer.append("g").attr("class", "diagram-nodes");

    function centerOf(n) {
      if (n.type === "label") {
        // approximate text box w/h for anchor math
        return { x: n.x + 40, y: n.y + 8 }; // small label
      }
      return { x: n.x + n.w / 2, y: n.y + n.h / 2 };
    }

    function anchorPoint(n, side) {
      const c = centerOf(n);
      switch (n.type) {
        case "rect":
        case "rect-dashed": {
          const { x, y, w, h } = n;
          return {
            left: { x: x, y: y + h / 2 },
            right: { x: x + w, y: y + h / 2 },
            top: { x: x + w / 2, y: y },
            bottom: { x: x + w / 2, y: y + h },
            center: c,
          }[side];
        }
        case "vae": {
          const c = { x: n.x, y: n.y };
          const s = n.scale || 1;
          const w2 = (n.w * s) / 2;
          const h2 = (n.h * s) / 2;
          return {
            left: { x: c.x - 40, y: c.y },
            right: { x: c.x + w2, y: c.y },
            top: { x: c.x, y: c.y - h2 },
            bottom: { x: c.x, y: c.y + h2 },
            center: c,
          }[side];
        }
        case "label": {
          const box = { x: n.x, y: n.y, w: 80, h: 16 };
          return {
            left: { x: box.x, y: box.y + box.h / 2 },
            right: { x: box.x + box.w, y: box.y + box.h / 2 },
            top: { x: box.x + box.w / 2, y: box.y },
            bottom: { x: box.x + box.w / 2, y: box.y + box.h },
            center: { x: box.x + box.w / 2, y: box.y + box.h / 2 },
          }[side];
        }
      }
    }

    // Draw nodes
    const node = gNodes
      .selectAll(".node")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", (d) => `node ${d.cls || ""}`)
      .attr("id", (d) => `${d.id || ""}`)
      .attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);

    // Rect types
    node
      .filter((d) => d.type === "rect")
      .each(function (d) {
        d3.select(this)
          .append("rect")
          .attr("class", "node-rect")
          .attr("id", d.id)
          .attr("width", d.w)
          .attr("height", d.h)
          .attr("rx", d.r || 11)
          .attr("ry", d.r || 11);
        if (d.id === "unet") {
          d3.select(this)
            .select("rect")
            .attr("fill", "url(#shimmer-gradient)")
            .attr("filter", "url(#glow-filter)");
        }
      });

    node
      .filter((d) => d.type === "rect-dashed")
      .append("rect")
      .attr("class", "node-rect dashed")
      .attr("width", (d) => d.w)
      .attr("height", (d) => d.h)
      .attr("rx", (d) => d.r || 11)
      .attr("ry", (d) => d.r || 11);

    node
      .filter((d) => d.type === "vae")
      .append("path")
      .attr("class", "node-rect")
      // .attr('d', "m 88.139678,-98.483372 33.757982,-17.641988 0.46837,59.757715 -34.538599,-18.890983 z");
      .attr(
        "d",
        "m -16.957053,-12.236869 L 16.800929,-29.878857 17.269299, 29.878858 -17.269300, 10.987875 z"
      )
      .attr("transform", (d) => (d.scale ? `scale(${d.scale})` : null));

    // Text labels (centered; supports \n)
    node
      .append("text")
      .attr("class", "node-label")
      .each(function (d) {
        const lines = String(d.label || "").split("\n");
        const text = d3.select(this);

        // Compute center inside the shape
        let cx = d.type === "label" ? 40 : d.w / 2;
        let cy = d.type === "label" ? 8 : d.h / 2;

        // For vae, still use its internal w/h center
        if (d.type === "vae") {
          cx = 0;
          cy = 0;
        }

        // First tspan
        const lineHeight = 14; // px
        const startY = cy - ((lines.length - 1) * lineHeight) / 2;

        lines.forEach((line, i) => {
          text
            .append("tspan")
            .attr("x", cx)
            .attr("y", startY + i * lineHeight)
            .text(line);
        });
      });

    // ---------- Links ----------
    const gLinks = zoomContainer.append("g").attr("class", "diagram-links");

    // Simple router: straight segment or an "L" if you prefer.
    function pathBetween(a, b, elbow = null) {
      if (!elbow) return `M${a.x},${a.y} L${b.x},${b.y}`;
      // elbow via mid-point
      const mid = { x: elbow.x, y: elbow.y };
      return `M${a.x},${a.y} L${mid.x},${mid.y} L${b.x},${b.y}`;
    }

    const linkSel = gLinks
      .selectAll(".link")
      .data(links)
      .enter()
      .append("path")
      .attr("class", (d) => `link ${d.dashed ? "dashed" : ""}`)
      .attr("marker-end", "url(#arrow)")
      .attr("d", (d) => {
        const fromN = byId[d.from.id];
        const toN = byId[d.to.id];
        const p1 = anchorPoint(fromN, d.from.side || "center");
        const p2 = anchorPoint(toN, d.to.side || "center");
        // Elbow for right-angle if x/y differ significantly
        const elbow = d.elbow;
        return pathBetween(p1, p2, elbow);
      });

    // ---------- Pan/zoom for diagram ----------
    const diagramZoom = d3.zoom()
      .scaleExtent([0.5, 2])
      .filter((event) => {
        const t = event.target;
        if (!t || !t.closest) return true;
        if (t.closest(".clickable")) return false; // don't pan when clicking diagram nodes
        return true;
      })
      .on("zoom", (event) => zoomContainer.attr("transform", event.transform));

      svg.call(diagramZoom);
      svg.call(diagramZoom.transform, d3.zoomIdentity.scale(0.8).translate(50, 0));

  }

  /**
   * Load info panel state from localStorage
   */
  loadInfoPanelState() {
    const saved = localStorage.getItem("infoPanelVisible");
    return saved !== null ? saved === "true" : true; // Default to visible
  }

  /**
   * Save info panel state to localStorage
   */
  saveInfoPanelState(visible) {
    localStorage.setItem("infoPanelVisible", visible.toString());
  }

  /**
   * Load legend state from localStorage
   */
  loadLegendState() {
    const saved = localStorage.getItem("legendVisible");
    return saved !== null ? saved === "true" : true; // Default to visible
  }

  /**
   * Save legend state to localStorage
   */
  saveLegendState(visible) {
    localStorage.setItem("legendVisible", visible.toString());
  }

  /**
   * Set up info panel toggle functionality
   */
  setupInfoPanelToggle() {
    const closeBtn = document.getElementById("info-panel-close");
    const toggleBtn = document.getElementById("info-panel-toggle");
    const infoPanel = document.getElementById("info-parent-panel");

    if (!closeBtn || !infoPanel) {
      console.warn("Info panel elements not found");
      return;
    }

    // Set initial state
    if (!this.infoPanelState) {
      infoPanel.classList.add("hidden");
      if (toggleBtn) toggleBtn.style.display = "block";
    }

    // Add click event listeners
    closeBtn.addEventListener("click", () => {
      this.toggleInfoPanel();
    });

    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        this.toggleInfoPanel();
      });
    }
  }

  /**
   * Toggle info panel visibility
   */
  toggleInfoPanel(forceShow = null) {
    const infoPanel = document.getElementById("info-parent-panel");
    const toggleBtn = document.getElementById("info-panel-toggle");

    if (!infoPanel) return;

    const isHidden = infoPanel.classList.contains("hidden");

    if (isHidden || forceShow === true) {
      infoPanel.classList.remove("hidden");
      if (toggleBtn) toggleBtn.style.display = "none";
      this.infoPanelState = true;
    } else {
      infoPanel.classList.add("hidden");
      if (toggleBtn) toggleBtn.style.display = "block";
      this.infoPanelState = false;
    }

    this.saveInfoPanelState(this.infoPanelState);
  }

  /**
   * Set up legend toggle functionality
   */
  setupLegendToggle() {
    const closeBtn = document.getElementById("legend-close");
    const toggleBtn = document.getElementById("legend-toggle");
    const legendContainer = document.querySelector(".legend-container");

    if (!closeBtn || !legendContainer) {
      console.warn("Legend elements not found");
      return;
    }

    // Set initial state
    if (!this.legendState) {
      legendContainer.classList.add("hidden");
      if (toggleBtn) toggleBtn.style.display = "block";
    }

    // Add click event listeners
    closeBtn.addEventListener("click", () => {
      this.toggleLegend();
    });

    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        this.toggleLegend();
      });
    }
  }

  /**
   * Toggle legend visibility
   */
  toggleLegend() {
    const legendContainer = document.querySelector(".legend-container");
    const toggleBtn = document.getElementById("legend-toggle");
    
    if (!legendContainer) return;

    const isHidden = legendContainer.classList.contains("hidden");
    console.log("Toggling legend", isHidden);

    if (isHidden) {
      legendContainer.classList.remove("hidden");
      if (toggleBtn) toggleBtn.style.display = "none";
      this.legendState = true;
    } else {
      legendContainer.classList.add("hidden");
      if (toggleBtn) toggleBtn.style.display = "block";
      this.legendState = false;
    }

    this.saveLegendState(this.legendState);
  }
}

function onDemoModeChange(mode) {
  console.log("Mode changed to:", mode);
  const pre = document.getElementById("precomputed_section");
  const live = document.getElementById("live_section");
  imageManager.onDemoModeChange(mode);
  if (pre && live) {
    if (mode === "precomputed") {
      pre.style.opacity = "1";
      pre.style.pointerEvents = "auto";
      live.style.opacity = "0.4";
      live.style.pointerEvents = "none";
    } else {
      pre.style.opacity = "0.4";
      pre.style.pointerEvents = "none";
      live.style.opacity = "1";
      live.style.pointerEvents = "auto";
    }
  }
}

function loadNewExperiment(modelName) {
  onPromptChange(modelName); // Update prompt input
  imageManager.updateImage(0, true);
}

// Global functions for backward compatibility
function onPromptChange(modelName) {
  const select = document.getElementById("prompt_text_select");
  if (!select) return;
  const experimentName = select.value;
  var experimentChanged = imageManager.setExperimentName(experimentName);
  console.log(`Experiment changed: ${experimentChanged}`);
  if (experimentChanged) {
    onModelChange(modelName);
  }

  // Update the prompt input with the selected experiment's prompt
  const promptInput = document.getElementById("prompt_input");
  if (promptInput && CONFIG.EXPERIMENTS?.[experimentName]) {
    promptInput.value = CONFIG.EXPERIMENTS[experimentName];
  }
}

function onLiveGenerationModeChange(mode) {
  imageManager.setLiveGenerationMode(mode);
}

function onModelChange(modelName) {
  imageManager.setModelName(modelName);
  unetVisualizer.setupExpandedArea("unet-expanded", "unet", true, false);
}

// Generate new image with current prompt and settings
async function generateNewImage(modelName) {
  onModelChange(modelName);
  const prompt = imageManager.getCurrentPrompt();
  if (!prompt || !prompt.trim()) {
    alert("Please enter a prompt first!");
    return;
  }
  await imageManager.updateImage(imageManager.getBendingAngle(), true);
}

/**
 * Update legend from blockTypeCategories (category -> hex color).
 * Uses last-1 type path as legend labels; Tableau10 colors.
 */
window.updateLegendFromBlockTypeCategories = function(blockTypeCategories) {
  const container = document.querySelector(".legend-container");
  if (!container) return;
  const items = container.querySelectorAll(".legend-item");
  items.forEach((el) => el.remove());
  const keys = Object.keys(blockTypeCategories || {}).sort();
  keys.forEach((label) => {
    const color = blockTypeCategories[label];
    const div = document.createElement("div");
    div.className = "legend-item";
    div.setAttribute("data-category", label);
    div.innerHTML = `<div class="legend-color" style="background-color: ${color}"></div><span class="legend-label">${label}</span>`;
    container.appendChild(div);
  });
};

// Apply model type to diagram and info panel (sd | flux | unknown)
window.applyModelTypeToUI = function (modelType) {
  const diagramContainer = document.getElementById("diagram-container");
  const genericMsg = document.getElementById("diagram-generic-message");
  const infoFlux = document.getElementById("info-flux");
  const infoUnknown = document.getElementById("info-unknown");
  const infoPanel = document.getElementById("info-panel");
  const sdSpans = infoPanel ? infoPanel.querySelectorAll("span:not(#info-flux):not(#info-unknown)") : [];
  if (modelType === "unknown") {
    if (diagramContainer) diagramContainer.style.display = "none";
    if (genericMsg) genericMsg.style.display = "block";
    if (infoFlux) infoFlux.style.display = "none";
    if (infoUnknown) infoUnknown.style.display = "block";
    sdSpans.forEach((s) => { s.style.display = "none"; });
    return;
  }
  if (diagramContainer) diagramContainer.style.display = "block";
  if (genericMsg) genericMsg.style.display = "none";
  if (infoUnknown) infoUnknown.style.display = "none";
  if (modelType === "flux") {
    if (infoFlux) infoFlux.style.display = "block";
    sdSpans.forEach((s) => { s.style.display = "none"; });
    window.loadFluxDiagram?.();
  } else {
    if (infoFlux) infoFlux.style.display = "none";
    sdSpans.forEach((s) => { s.style.display = ""; });
  }
};

// Minimal Flux architecture diagram (inputs -> single_blocks / double_blocks -> final_layer). Fits viewBox 0 0 450 300.
window.loadFluxDiagram = function () {
  const sel = d3.select("#svg-diagram");
  if (sel.empty()) return;
  sel.selectAll("*").remove();
  const svg = sel.attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("class", "diagram-zoom-container");
  const nodes = [
    { id: "txt_in", x: 15, y: 20, w: 52, h: 24, label: "txt_in" },
    { id: "img_in", x: 15, y: 55, w: 52, h: 24, label: "img_in" },
    { id: "time_in", x: 15, y: 90, w: 52, h: 24, label: "time_in" },
    { id: "vector_in", x: 15, y: 125, w: 52, h: 24, label: "vector_in" },
    { id: "single_blocks", x: 100, y: 55, w: 90, h: 60, label: "single_blocks" },
    { id: "double_blocks", x: 220, y: 55, w: 90, h: 60, label: "double_blocks" },
    { id: "final_layer", x: 350, y: 65, w: 65, h: 40, label: "final_layer" },
  ];
  g.selectAll("g.flux-node")
    .data(nodes)
    .join("g")
    .attr("class", "flux-node")
    .each(function (d) {
      const gr = d3.select(this);
      gr.append("rect").attr("x", d.x).attr("y", d.y).attr("width", d.w).attr("height", d.h).attr("rx", 6).attr("fill", "#e8e8e8").attr("stroke", "#333");
      gr.append("text").attr("x", d.x + d.w / 2).attr("y", d.y + d.h / 2).attr("text-anchor", "middle").attr("dy", "0.35em").attr("font-size", 11).text(d.label);
    });
  g.append("path").attr("d", "M67,32 L100,85").attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1);
  g.append("path").attr("d", "M67,67 L100,85").attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1);
  g.append("path").attr("d", "M67,102 L100,85").attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1);
  g.append("path").attr("d", "M67,137 L100,85").attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1);
  g.append("path").attr("d", "M190,85 L220,85").attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1);
  g.append("path").attr("d", "M310,85 L350,85").attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1);
};

// Function to fetch model structure from ComfyUI and convert to UNET_BLOCKS format
// partParam: optional part path to request (e.g. "diffusion_model"); if omitted, backend returns default.
window.fetchModelStructure = async function (refresh = false, partParam = null) {
  if (!window.comfySessionId) return false;
  window.clearComfyError?.();

  try {
    let url = `/web_bend_demo/layers?session_id=${encodeURIComponent(window.comfySessionId)}`;
    if (partParam) url += "&part=" + encodeURIComponent(partParam);
    const response = await fetch(url);

    if (!response.ok) {
      const msg = await window.parseErrorFromResponse(response);
      window.showComfyError?.("Model structure: " + msg);
      return false;
    }

    const data = await response.json();

    if (!data.ok || !data.tree) {
      window.showComfyError?.("Model structure: " + (data.error || "Unknown error"));
      return false;
    }

    // Diagram is chosen by entire model type (sd/flux/unknown), not by selected top-level part.
    window.webBendDemoModelType = data.model_type || "unknown";
    if (typeof window.applyModelTypeToUI === "function") {
      window.applyModelTypeToUI(window.webBendDemoModelType);
    }

    if (data.selected_part && typeof imageManager !== "undefined") {
      imageManager.selectedPart = data.selected_part;
    }

    if (data.ok && data.tree) {
      // Convert ComfyUI tree structure to UNET_BLOCKS format
      const blocks = [];
      const blockTypes = [];
      const typePaths = [];  // type path from root to leaf: grandparent.parent.LeafType

      function traverseTree(node, path = "", typePrefix = "") {
        const currentPath = node.path || (path ? path + "." + node.name : node.name);
        const typeName = node.type || "Module";
        const typePath = typePrefix ? typePrefix + "." + typeName : typeName;
        const hasChildren = node.children && node.children.length > 0;

        if (hasChildren) {
          node.children.forEach((child) => traverseTree(child, currentPath, typePath));
        } else {
          if (currentPath && currentPath !== "root" && currentPath !== "" && currentPath !== "root.") {
            blocks.push(currentPath);
            blockTypes.push(node.full_type || node.type || "Conv2d");
            typePaths.push(typePath);
          }
        }
      }

      traverseTree(data.tree);

      if (blocks.length === 0) {
        console.warn("First traversal got no blocks, trying simpler approach that includes all nodes...");
        function simpleTraverse(node, parentPath = "", typePrefix = "") {
          const currentPath = node.path || (parentPath ? parentPath + "." + node.name : node.name);
          const typeName = node.type || "Module";
          const typePath = typePrefix ? typePrefix + "." + typeName : typeName;
          if (currentPath && currentPath !== "root" && currentPath !== "" && currentPath !== "root.") {
            blocks.push(currentPath);
            blockTypes.push(node.full_type || node.type || "Conv2d");
            typePaths.push(typePath);
          }
          if (node.children) {
            node.children.forEach((child) => simpleTraverse(child, currentPath, typePath));
          }
        }
        blocks.length = 0;
        blockTypes.length = 0;
        typePaths.length = 0;
        simpleTraverse(data.tree);
        console.log(`Simple traversal found ${blocks.length} blocks`);
      }

      console.log(`Converted ${blocks.length} blocks from tree structure`);
      console.log("Sample blocks:", blocks.slice(0, 5));
      console.log("Sample blockTypes:", blockTypes.slice(0, 5));
      console.log("Sample typePaths:", typePaths.slice(0, 5));
      
      // Update UNET_BLOCKS with the fetched structure
      // Use a generic key that will always be found, or use the actual model name
      const modelName = imageManager.getModelName() || "SD1.4";
      // Also store under a generic "ComfyUI" key to ensure it's always found
      const storageKeys = [modelName, "ComfyUI", "SD1.4"];
      
      storageKeys.forEach(key => {
        if (!UNET_BLOCKS[key]) {
          UNET_BLOCKS[key] = {};
        }
        UNET_BLOCKS[key].blocks = blocks;
        UNET_BLOCKS[key].blockTypes = blockTypes;
        UNET_BLOCKS[key].typePaths = typePaths;
      });

      const stemLast1 = (p) => (p.split(".").pop() || p);
      const stemLast2 = (p) => {
        const parts = p.split(".");
        return parts.length >= 2 ? parts.slice(-2).join(".") : p;
      };
      const stemLast3 = (p) => {
        const parts = p.split(".");
        return parts.length >= 3 ? parts.slice(-3).join(".") : p;
      };

      // Build blockTypeCategories from last-1 type path, assign Tableau10 colors
      const uniqueCategories = [...new Set(typePaths.map(stemLast1))];
      const palette = CONFIG.TABLEAU10 || ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"];
      const blockTypeCategories = {};
      uniqueCategories.forEach((cat, idx) => {
        blockTypeCategories[cat] = palette[idx % palette.length];
      });
      storageKeys.forEach((key) => {
        UNET_BLOCKS[key].blockTypeCategories = blockTypeCategories;
      });
      if (typeof window.updateLegendFromBlockTypeCategories === "function") {
        window.updateLegendFromBlockTypeCategories(blockTypeCategories);
      }

      console.log(`Loaded ${blockTypes.length} blocks from ComfyUI model structure`);

      // Block categories: print all and unique counts before/after stemming
      const categories = blockTypes.slice();
      const uniqueBeforeStem = new Set(categories).size;
      const stemmed1 = categories.map(stemLast1);
      const stemmed2 = categories.map(stemLast2);
      const uniqueAfterStem1 = new Set(stemmed1).size;
      const uniqueAfterStem2 = new Set(stemmed2).size;
      console.log("[Block categories] All block categories (paths):", categories);
      console.log("[Block categories] Unique categories (before stemming):", uniqueBeforeStem);
      console.log("[Block categories] Unique after stemming to last 1 part:", uniqueAfterStem1, "→ sample:", [...new Set(stemmed1)].slice(0, 15));
      console.log("[Block categories] Unique after stemming to last 2 parts:", uniqueAfterStem2, "→ sample:", [...new Set(stemmed2)].slice(0, 15));

      // Type paths (grandparent.parent.leaf): print all and unique counts before/after stemming to 1, 2, 3 parts
      const uniqueTypePathsBeforeStem = new Set(typePaths).size;
      const typeStemmed1 = typePaths.map(stemLast1);
      const typeStemmed2 = typePaths.map(stemLast2);
      const typeStemmed3 = typePaths.map(stemLast3);
      const uniqueTypeStem1 = new Set(typeStemmed1).size;
      const uniqueTypeStem2 = new Set(typeStemmed2).size;
      const uniqueTypeStem3 = new Set(typeStemmed3).size;
      console.log("[Type paths] All type paths (grandparent.parent.leaf):", typePaths);
      console.log("[Type paths] Unique type paths (before stemming):", uniqueTypePathsBeforeStem);
      console.log("[Type paths] Unique after stemming to last 1 part:", uniqueTypeStem1, "→ sample:", [...new Set(typeStemmed1)].slice(0, 15));
      console.log("[Type paths] Unique after stemming to last 2 parts:", uniqueTypeStem2, "→ sample:", [...new Set(typeStemmed2)].slice(0, 15));
      console.log("[Type paths] Unique after stemming to last 3 parts:", uniqueTypeStem3, "→ sample:", [...new Set(typeStemmed3)].slice(0, 15));
      
      return true;
    }
  } catch (e) {
    console.error("Error fetching model structure from ComfyUI:", e);
    window.showComfyError?.("Model structure: " + (e.message || String(e)));
    return false;
  }
};

// Global function to refresh model structure
window.refreshModelStructure = async function() {
  if (!window.comfySessionId) {
    window.showComfyInfo?.("Not running in ComfyUI context. Refresh only works when opened from ComfyUI.");
    return;
  }
  
  const refreshBtn = document.getElementById("refresh-model-btn");
  if (refreshBtn) {
    refreshBtn.disabled = true;
    refreshBtn.textContent = "⟳";
  }
  
  try {
    const success = await window.fetchModelStructure(true);
    if (success) {
      window.showComfyInfo?.("Model structure refreshed successfully.");
    } else {
      window.showComfyError?.("Failed to refresh model structure. Make sure the node has been queued at least once.");
    }
  } catch (e) {
    console.error("Error refreshing model structure:", e);
    window.showComfyError?.("Error refreshing model structure: " + e.message);
  } finally {
    if (refreshBtn) {
      refreshBtn.disabled = false;
      refreshBtn.textContent = "↻";
    }
  }
};

// ComfyUI message banner helpers (available when running in ComfyUI context)
// showComfyError(msg) - error styling (red), showComfyInfo(msg) - info styling (blue)
window.showComfyError = function(msg, type = "error") {
  const b = document.getElementById("comfy-error-banner");
  const t = document.getElementById("comfy-error-text");
  if (b && t) {
    t.textContent = msg;
    b.style.display = "flex";
    b.classList.toggle("comfy-info-banner", type === "info");
  }
};
window.showComfyInfo = function(msg) {
  window.showComfyError?.(msg, "info");
};
window.clearComfyError = function() {
  const b = document.getElementById("comfy-error-banner");
  if (b) {
    b.style.display = "none";
    b.classList.remove("comfy-info-banner");
  }
};

/**
 * Parse error message from a failed fetch response.
 * Backend returns JSON { ok: false, error: "..." }; use that when possible.
 * @param {Response} response
 * @returns {Promise<string>}
 */
window.parseErrorFromResponse = async function(response) {
  const text = await response.text();
  try {
    const d = JSON.parse(text);
    if (d != null && typeof d.error === "string" && d.error) return d.error;
  } catch (_) {}
  return text || response.statusText || "Request failed";
};

/** Setup fast tooltip (150ms delay). Uses data-tooltip for plain text or data-tooltip-html for rich content (links, images). */
function setupFastTooltip(el, delayMs = 150) {
  const text = el?.dataset?.tooltip;
  const html = el?.dataset?.tooltipHtml;
  if (!el || (!text && !html)) return;
  let tt = document.getElementById("fast-tooltip");
  if (!tt) {
    tt = document.createElement("div");
    tt.id = "fast-tooltip";
    tt.className = "fast-tooltip";
    document.body.appendChild(tt);
  }
  let timeout;
  el.addEventListener("mouseenter", () => {
    timeout = setTimeout(() => {
      if (html) tt.innerHTML = html;
      else tt.textContent = text;
      const rect = el.getBoundingClientRect();
      tt.style.left = Math.min(rect.left, window.innerWidth - 290) + "px";
      tt.style.top = (rect.bottom + 6) + "px";
      tt.classList.add("visible");
    }, delayMs);
  });
  el.addEventListener("mouseleave", () => {
    clearTimeout(timeout);
    tt.classList.remove("visible");
  });
}

// Initialize the application when DOM is loaded
document.addEventListener("DOMContentLoaded", async () => {
  const dismissBtn = document.getElementById("comfy-error-dismiss");
  if (dismissBtn) dismissBtn.addEventListener("click", () => window.clearComfyError?.());

  const refreshBtn = document.getElementById("refresh-model-btn");
  if (refreshBtn) setupFastTooltip(refreshBtn, 150);
  const copyBendsBtn = document.getElementById("copy-bends-btn");
  if (copyBendsBtn) setupFastTooltip(copyBendsBtn, 150);
  const stepsResetBtn = document.getElementById("steps-reset-btn");
  if (stepsResetBtn) setupFastTooltip(stepsResetBtn, 150);

  // Get session_id from URL parameters if available
  const urlParams = new URLSearchParams(window.location.search);
  const sessionIdFromUrl = urlParams.get("session_id");
  if (sessionIdFromUrl && !window.comfySessionId) {
    window.comfySessionId = sessionIdFromUrl;
    console.log("Session ID loaded from URL:", window.comfySessionId);
  }
  
  const app = new DiffusionModelApp();
  app.init();
  console.log("App initialized");
  
  if (window.comfySessionId) {
    const unetHeader = document.getElementById("unet-expanded-header");
    if (unetHeader) unetHeader.style.display = "flex";
    const stepsControls = document.getElementById("unet-steps-controls");
    if (stepsControls) stepsControls.style.display = "flex";
    const controls = document.getElementById("comfy-queue-controls");
    const imgContainer = document.getElementById("image-container");
    if (controls) {
      controls.style.display = "flex";
      const live = controls.querySelector("input[value='live']");
      const onDemand = controls.querySelector("input[value='on_demand']");
      const queueBtn = document.getElementById("queue-prompt-btn");
      const clearBtn = document.getElementById("clear-all-bends-btn");
      if (live) live.addEventListener("change", () => imageManager.setQueueMode("live"));
      if (onDemand) onDemand.addEventListener("change", () => imageManager.setQueueMode("on_demand"));
      if (queueBtn) queueBtn.addEventListener("click", () => imageManager.sendSelectionToComfyUI());
      if (clearBtn) {
        clearBtn.addEventListener("click", async () => {
          await imageManager.clearAllComfyState();
          if (typeof unetVisualizer !== "undefined" && unetVisualizer.resetAllSliders) unetVisualizer.resetAllSliders();
          if (typeof unetVisualizer !== "undefined" && unetVisualizer.updateRubberEffect) unetVisualizer.updateRubberEffect(0);
        });
      }
    }
    if (imgContainer) imgContainer.classList.add("comfy-queue-visible");
    const historyStrip = document.getElementById("comfy-history-strip");
    if (historyStrip) {
      historyStrip.style.display = "flex";
      await imageManager._loadHistory();
    }
    console.log("Running in ComfyUI context, fetching model structure...");
    await window.fetchModelStructure();
    try {
      const selRes = await fetch(`/web_bend_demo/selection?session_id=${encodeURIComponent(window.comfySessionId)}`);
      if (selRes.ok) {
        const selData = await selRes.json();
        if (selData.ok && selData.bends) {
          imageManager.setBends(selData.bends);
          imageManager.setStepsFromSelection(selData);
          imageManager.updateBends();
          if (typeof unetVisualizer !== "undefined" && unetVisualizer.syncSlidersFromBends) {
            unetVisualizer.syncSlidersFromBends(selData.bends);
          }
        }
      }
    } catch (e) {
      console.warn("Could not load selection:", e);
    }
    imageManager.fetchDefaultImage();
  }

  await unetVisualizer.expandUnet();
  imageManager.updateImage(0, true);
});

// Export for potential module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = { DiffusionModelApp };
}
