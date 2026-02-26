// Image Manager Module

class ImageManager {
  constructor() {
    this.bendingAngle = 0;
    this.layerName = "";
    this.experimentName = "UNet_Rotation_Person";
    this.modelName = "SD1.4"; // Default model

    this.outputImage = document.getElementById("output_image");
    this.outputImageZoom = document.getElementById("output_image_zoom");
    this.defaultImage = document.getElementById("default_image");
    this.outputParams = document.getElementById("output_params");

    this.imageCache = new Map();
    this.demoMode = "live"; // or "live"
    this.liveGenerationMode = "local"; // "local" or "remote"
    /** @type {Map<string, {module_type: string, module_args: Object}>} path -> bend entry */
    this.bends = new Map();
    /** "live" | "on_demand" */
    this.queueMode = "live";
    /** @type {Array<{id: string, imageUrl: string, bends: Array<{path: string, module_type?: string, module_args?: Object, angle?: number}>, timestamp: number}>} */
    this.imageHistory = [];
    this.imageHistoryMax = 20;
    /** Stored URL for the unbent (default) image; shown in main preview until first bent image. */
    this.defaultImageUrl = "";
    this._fetchingDefault = false;
    /** True once a bent image has been generated; then default moves to corner overlay. */
    this._hasGeneratedBentImage = false;
    /** Bend timestep range: min/max denoising step (null = all steps). */
    this.stepsMin = null;
    this.stepsMax = null;
    this.maxDenoisingSteps = 200;
    /** Top-level part being bent (e.g. diffusion_model, single_blocks). */
    this.selectedPart = null;
    this.setupEventListeners();
    this._setupCopyBendsButton();
    this._setupStepsControls();
  }

  _getOutputParamsTextEl() {
    return document.getElementById("output_params_text") || this.outputParams;
  }

  _setupCopyBendsButton() {
    const btn = document.getElementById("copy-bends-btn");
    if (btn) btn.addEventListener("click", () => {
      const bends = this.getBends();
      const selection = {
        bends,
        steps_min: this.stepsMin,
        steps_max: this.stepsMax,
        max_denoising_steps: this.maxDenoisingSteps,
        selected_part: this.selectedPart,
      };
      const json = JSON.stringify(selection, null, 2);
      navigator.clipboard.writeText(json).then(() => {
        const orig = btn.textContent;
        btn.textContent = "✓";
        setTimeout(() => { btn.textContent = orig; }, 800);
      }).catch(() => {});
    });
  }

  _setupStepsControls() {
    const container = document.getElementById("unet-steps-controls");
    const minEl = document.getElementById("steps-min");
    const maxEl = document.getElementById("steps-max");
    const resetBtn = document.getElementById("steps-reset-btn");
    if (!container || !minEl || !maxEl) return;
    const apply = () => {
      const minVal = minEl.value.trim();
      const maxVal = maxEl.value.trim();
      this.stepsMin = minVal === "" ? null : parseInt(minVal, 10);
      this.stepsMax = maxVal === "" ? null : parseInt(maxVal, 10);
      if (Number.isNaN(this.stepsMin)) this.stepsMin = null;
      if (Number.isNaN(this.stepsMax)) this.stepsMax = null;
      if (window.comfySessionId && this.getBends().length > 0) this.sendSelectionToComfyUI();
    };
    minEl.addEventListener("change", apply);
    maxEl.addEventListener("change", apply);
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        this.stepsMin = null;
        this.stepsMax = null;
        minEl.value = "";
        maxEl.value = "";
        if (window.comfySessionId && this.getBends().length > 0) this.sendSelectionToComfyUI();
      });
    }
  }

  /** Sync step inputs and selected part from server (e.g. after GET selection). */
  setStepsFromSelection(data) {
    if (data && (data.steps_min !== undefined || data.steps_max !== undefined)) {
      this.stepsMin = data.steps_min ?? null;
      this.stepsMax = data.steps_max ?? null;
      if (data.max_denoising_steps != null) this.maxDenoisingSteps = data.max_denoising_steps;
      const minEl = document.getElementById("steps-min");
      const maxEl = document.getElementById("steps-max");
      if (minEl) minEl.value = this.stepsMin != null ? String(this.stepsMin) : "";
      if (maxEl) maxEl.value = this.stepsMax != null ? String(this.stepsMax) : "";
    }
    if (data && data.selected_part !== undefined) {
      this.selectedPart = data.selected_part || null;
      const selectEl = document.getElementById("model-part-select");
      if (selectEl && this.selectedPart) {
        selectEl.value = this.selectedPart;
      }
    }
  }

  setQueueMode(mode) {
    this.queueMode = mode === "on_demand" ? "on_demand" : "live";
  }

  getQueueMode() {
    return this.queueMode;
  }

  /**
   * Set up event listeners for the slider
   */
  setupEventListeners() {}

  _setOutputImageBg(url) {
    const val = url ? `url('${url}')` : "url()";
    if (this.outputImage) this.outputImage.style.backgroundImage = val;
    const zoom = this.outputImageZoom;
    if (zoom) {
      zoom.style.backgroundImage = val;
      const wrap = this.outputImage?.closest?.(".preview-with-hover-zoom");
      if (wrap) wrap.classList.toggle("has-preview-image", !!url);
    }
  }

  /** Update main preview and default overlay: no bent image = default in main; bent image = generated in main, default in corner. */
  _updatePreviewLayout() {
    if (!this.defaultImage) return;
    if (!this._hasGeneratedBentImage && this.defaultImageUrl) {
      this._setOutputImageBg(this.defaultImageUrl);
      this.defaultImage.style.display = "none";
    } else if (this._hasGeneratedBentImage) {
      this.defaultImage.style.backgroundImage = this.defaultImageUrl ? `url('${this.defaultImageUrl}')` : "url()";
      this.defaultImage.style.display = this.defaultImageUrl ? "block" : "none";
    } else {
      this._setOutputImageBg("");
      this.defaultImage.style.display = "none";
    }
  }

  /**
   * Update the experiment name
   * @param {string} experimentName - New experiment name
   */
  setExperimentName(experimentName) {
    var old = this.experimentName;
    this.experimentName = experimentName;
    return experimentName != old;
  }

  onDemoModeChange(newMode) {
    this.demoMode = newMode;
  }

  setLiveGenerationMode(newMode) {
    this.liveGenerationMode = newMode;
  }

  /**
   * Update the model name
   * @param {string} modelName - New model name
   */
  setModelName(modelName) {
    this.modelName = modelName;
  }

  /**
   * Set the current layer name
   * @param {string} layerName - Layer name
   */
  setLayerName(layerName) {
    this.layerName = layerName;
  }
  getModelName() {
    return this.modelName;
  }

  /** @returns {{module_type: string, module_args: Object}|null} */
  getBend(path) {
    return this.bends.get(path) ?? null;
  }

  /** @returns {Array<{path: string, module_type: string, module_args: Record<string, number}>} */
  getBends() {
    const out = [];
    this.bends.forEach((entry, path) => {
      if (!entry || !path) return;
      const def = CONFIG.BENDING_TYPES?.[entry.module_type];
      const key = def?.module_args_key;
      const val = key ? entry.module_args?.[key] : undefined;
      const defVal = def?.defaultValue;
      const isActive = val !== undefined && val !== defVal;
      if (isActive)
        out.push({ path, module_type: entry.module_type, module_args: { ...entry.module_args } });
    });
    return out;
  }

  /** @param {string} path - Layer path
   *  @param {string|number} module_type - "add_noise"|"multiply"|"rotate", or legacy number (angle) to clear/set rotate
   *  @param {Object} [module_args] - e.g. {noise_std: 1} or {angle_degrees: 90}
   */
  setBend(path, module_type, module_args) {
    if (!path) return;
    if (arguments.length === 2 && typeof module_type === "number") {
      if (module_type === 0) this.bends.delete(path);
      else this.bends.set(path, { module_type: "rotate", module_args: { angle_degrees: module_type } });
      return;
    }
    const def = CONFIG.BENDING_TYPES?.[module_type];
    const key = def?.module_args_key;
    const val = key ? module_args?.[key] : undefined;
    const defVal = def?.defaultValue;
    if (val !== undefined && val !== defVal)
      this.bends.set(path, { module_type, module_args: { ...module_args } });
    else
      this.bends.delete(path);
  }

  /** @param {Array<{path: string, module_type?: string, module_args?: Object, angle?: number}>} entries - supports legacy {path, angle} */
  setBends(entries) {
    this.bends.clear();
    (entries || []).forEach((b) => {
      if (!b?.path) return;
      if (b.module_type && b.module_args) {
        this.bends.set(b.path, { module_type: b.module_type, module_args: { ...b.module_args } });
      } else if (typeof b.angle === "number" && b.angle !== 0) {
        this.bends.set(b.path, { module_type: "rotate", module_args: { angle_degrees: b.angle } });
      }
    });
  }

  clearBends() {
    this.bends.clear();
  }

  _pushHistory(imageUrl, bends) {
    const id = `h-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.imageHistory.unshift({
      id,
      imageUrl,
      bends,
      timestamp: Date.now(),
      steps_min: this.stepsMin,
      steps_max: this.stepsMax,
    });
    if (this.imageHistory.length > this.imageHistoryMax) this.imageHistory.pop();
    this._renderHistoryStrip();
    this._persistHistory();
  }

  async _loadHistory() {
    if (!window.comfySessionId) return;
    try {
      const r = await fetch(
        `/web_bend_demo/history/load?session_id=${encodeURIComponent(window.comfySessionId)}`
      );
      const data = await r.json();
      if (data && data.ok && Array.isArray(data.history)) {
        this.imageHistory = data.history;
        this._renderHistoryStrip();
      }
    } catch (e) {
      console.warn("Failed to load history:", e);
    }
  }

  _persistHistory() {
    if (!window.comfySessionId) return;
    fetch("/web_bend_demo/history/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: window.comfySessionId,
        history: this.imageHistory,
      }),
    }).catch((e) => console.warn("Failed to persist history:", e));
  }

  _renderHistoryStrip() {
    const wrap = document.getElementById("comfy-history-strip");
    if (!wrap) return;
    let label = wrap.querySelector(".comfy-history-label");
    let thumbs = wrap.querySelector(".comfy-history-thumbs");
    if (!label) {
      label = document.createElement("span");
      label.className = "comfy-history-label";
      label.textContent = "History:";
      wrap.appendChild(label);
    }
    if (!thumbs) {
      thumbs = document.createElement("div");
      thumbs.className = "comfy-history-thumbs";
      wrap.appendChild(thumbs);
    }
    thumbs.innerHTML = "";
    this.imageHistory.forEach((entry) => {
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = "comfy-history-thumb";
      thumb.style.backgroundImage = `url('${entry.imageUrl}')`;
      thumb.title = "Restore these bends";
      thumb.dataset.id = entry.id;
      thumb.addEventListener("click", () => this._restoreFromHistory(entry));
      thumbs.appendChild(thumb);
    });
  }

  _restoreFromHistory(entry) {
    this.setBends(entry.bends);
    this.updateBends();
    if (entry.imageUrl) {
      this._hasGeneratedBentImage = true;
      this._setOutputImageBg(entry.imageUrl);
      this._updatePreviewLayout();
    }
    // Restore timestep params if present
    if (entry.steps_min !== undefined || entry.steps_max !== undefined) {
      this.setStepsFromSelection({
        steps_min: entry.steps_min,
        steps_max: entry.steps_max,
      });
    }
    // Sync backend state but don't trigger a new queue - we already have the image
    if (window.comfySessionId && this.getBends().length > 0) {
      this.sendSelectionToComfyUI(null, { skipQueue: true });
    }
    const uv = typeof unetVisualizer !== "undefined" ? unetVisualizer : null;
    if (uv) {
      if (uv.syncSlidersFromBends) uv.syncSlidersFromBends(entry.bends);
      const intensity = this.getBendingIntensity();
      if (uv.updateRubberEffect) uv.updateRubberEffect(intensity);
      if (uv.updateBlockHighlights) uv.updateBlockHighlights();
    }
  }

  /**
   * Clear bends, history, POST clear, stop polling, clear preview. Does not reset slider DOM; call resetAllSliders separately.
   */
  async clearAllComfyState() {
    if (!window.comfySessionId) return;
    this.clearBends();
    this.imageHistory = [];
    this._renderHistoryStrip();
    this._persistHistory();
    this.updateBends();
    try {
      await fetch("/web_bend_demo/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: window.comfySessionId }),
      });
    } catch (e) {
      console.error("Error clearing selection:", e);
    }
    this.stopImagePolling();
    this._hasGeneratedBentImage = false;
    this._updatePreviewLayout();
  }

  /**
   * Queue a run with no bends, poll for the result, and set it as the default (unbent) image.
   * Call once on ComfyUI init so default_image always shows the unbent result.
   */
  async fetchDefaultImage() {
    if (!window.comfySessionId || !this.defaultImage) return;
    try {
      await fetch("/web_bend_demo/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: window.comfySessionId }),
      });
      if (window.parent && window.parent !== window) {
        try {
          window.parent.postMessage({
            type: "web_bend_demo_selection_changed",
            session_id: window.comfySessionId,
            change_hash: "default",
          }, "*");
        } catch (e) {
          console.log("Could not notify parent for default fetch:", e);
        }
      }
      this._fetchingDefault = true;
      this.startImagePolling();
    } catch (e) {
      console.error("Error fetching default image:", e);
      this._fetchingDefault = false;
    }
  }

  /** Update params UI from current bends. Does not send to backend or queue. */
  updateBends() {
    if (!this.outputParams) return;
    const textEl = document.getElementById("output_params_text");
    const copyBtn = document.getElementById("copy-bends-btn");
    const arr = this.getBends();
    if (arr.length === 0) {
      if (textEl) textEl.innerHTML = "";
      if (copyBtn) copyBtn.style.display = "none";
      return;
    }
    const fmt = (b) => {
      const t = CONFIG.BENDING_TYPES?.[b.module_type];
      const k = t?.module_args_key;
      const v = k ? b.module_args?.[k] : undefined;
      if (b.module_type === "rotate") return `${v}°`;
      if (b.module_type === "add_noise") return `σ=${v}`;
      if (b.module_type === "multiply") return `×${v}`;
      return String(v);
    };
    const summary =
      arr.length === 1
        ? `Layer: ${arr[0].path}<br>${(CONFIG.BENDING_TYPES?.[arr[0].module_type]?.label || arr[0].module_type)}: ${fmt(arr[0])}`
        : `${arr.length} layers bent`;
    if (textEl) textEl.innerHTML = `Model: ${this.modelName}<br>${summary}`;
    if (copyBtn) copyBtn.style.display = "inline-flex";
  }

  /**
   * Update the displayed image based on angle and layer
   * @param {number} newAngle - Legacy: rotation angle (for non-ComfyUI/precomputed)
   * @param {boolean} bypass - Whether to bypass change detection
   */
  async updateImage(newAngle, bypass = false) {
    console.log(
      `Updating image for angle: ${newAngle}, layer: ${this.layerName}`
    );
    if (this.bendingAngle === newAngle && !bypass) return; // No change

    console.log(`New angle: ${newAngle}`);
    this.bendingAngle = newAngle;

    if (window.comfySessionId) {
      this.updateBends();
      return;
    }

    // Show loading state
    console.log("Generating mode ...", this.mode, this.liveGenerationMode);
    if (this.demoMode === "precomputed") {
      const defaultImageUrl = `${CONFIG.FILES_SERVER_URL}/${this.experimentName}/default/default.png`;

      if (this.bendingAngle === 0 || this.layerName === "") {
        this.defaultImage.style.backgroundImage = `url('${defaultImageUrl}')`;
        this._setOutputImageBg(defaultImageUrl);
        this._getOutputParamsTextEl().innerHTML = "";
      } else {
        this.defaultImage.style.backgroundImage = `url('${defaultImageUrl}')`;
        var url = `${CONFIG.FILES_SERVER_URL}/${this.experimentName}/${this.layerName}/angle_degrees_${this.bendingAngle}_00001_.png`;
        this._setOutputImageBg(url);
        this._getOutputParamsTextEl().innerHTML = `Model: ${this.modelName} <br> Layer: ${this.layerName} <br> Bending angle: ${this.bendingAngle}°`;
      }
    } else {
      //'live' mode
      this.showLoadingState();
      try {
        if (this.bendingAngle === 0 || this.layerName === "") {
          // For default state, use a default image or generate with no bending
          this.defaultImage.style.backgroundImage = `url()`;
          var imageUrl = await this.generateImage(
            "",
            0,
            this.liveGenerationMode
          );
          this.defaultImage.style.backgroundImage = `url('${imageUrl}')`;
          this._getOutputParamsTextEl().innerHTML = "";
        } else {
          // Generate image with bending parameters
          await this.generateImage(
            this.layerName,
            this.bendingAngle,
            this.liveGenerationMode
          );
        }
        this.hideLoadingState();
      } catch (error) {
        console.error("Error generating image:", error);
        this.showErrorState();
      }
    }
  }

  /**
   * Generate image by sending request to remote server
   * @param {string} layerName - Layer name for bending
   * @param {number} angle - Bending angle
   */
  async generateImageServer(layerName, angle) {
    const prompt = this.getCurrentPrompt();

    const requestBody = {
      input: {
        prompt: prompt,
        bending_layer: layerName,
        bending_angle: angle,
        negative_prompt: "",
        height: 512,
        width: 512,
        num_inference_steps: 1,
        guidance_scale: 0.0,
        seed: 42,
        num_images: 1,
      },
    };

    console.log("Sending request to server:", requestBody);

    const response = await fetch(`${CONFIG.GENERATION_SERVER_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    const result = await response.json();

    if (result.success && result.image_url) {
      return result.image_url;
    } else {
      throw new Error(result.error || "Failed to generate image");
    }
  }

  async generateImage(layerName, angle) {
    // Check if we're running in ComfyUI context
    if (window.comfySessionId) {
      // In ComfyUI mode, send selection to backend instead of generating directly
      await this.sendSelectionToComfyUI(layerName, angle);
      return null;
    }
    
    const prompt = this.getCurrentPrompt();
    const cacheKey = `${layerName}_${angle}_${prompt}`;
    // 🔹 Step 0: Check cache first
    if (this.imageCache.has(cacheKey)) {
      const cachedUrl = this.imageCache.get(cacheKey);
      console.log("Using cached image:", cachedUrl);

      this._setOutputImageBg(cachedUrl);
      this._getOutputParamsTextEl().innerHTML = `Model: ${this.modelName} <br> Layer: ${layerName} <br> Bending angle: ${angle}°`;

      return cachedUrl;
    }
    console.log("Generating new image with prompt:", prompt);

    var imageUrl = null;
    console.log(this.liveGenerationMode)
    if (this.liveGenerationMode === "remote") {
      imageUrl = await this.generateImageComfyView(prompt, layerName, angle);
    } else {
      imageUrl = await this.generateImageLocalComfy(prompt, layerName, angle);
    }

    this.imageCache.set(cacheKey, imageUrl);
    if (this.imageCache.size > 80) {
      const firstKey = this.imageCache.keys().next().value;
      this.imageCache.delete(firstKey);
      console.log(`🧹 Cache full — removed oldest entry (${firstKey})`);
    }
    //If user hasn't changed parameters during generation, update image
    if (this.bendingAngle == angle && this.layerName == layerName) {
      this._setOutputImageBg(imageUrl);
      this._getOutputParamsTextEl().innerHTML = `Model: ${this.modelName} <br> Layer: ${layerName} <br> Bending angle: ${angle}°`;

    }

    console.log("Image generated:", imageUrl);

    return imageUrl;
  }

  /**
   * If queue mode is live, commit current bends and queue. Call on slider/reset interaction end.
   */
  async commitBendsIfLive() {
    if (!window.comfySessionId || this.queueMode !== "live") return;
    await this.sendSelectionToComfyUI();
  }

  /**
   * Commit bends to ComfyUI and queue. Call on interaction end (live) or Generate click (on-demand).
   * @param {Array<{path: string, module_type: string, module_args: Record<string, number}>|undefined} bendsArg - Optional; uses getBends() if omitted.
   */
  /**
   * Look up experiment cache for current bends + steps. Returns image URL if hit, null otherwise.
   * @param {Array} bends - Bends array
   * @returns {Promise<string|null>} Image URL or null
   */
  async _lookupExperimentCache(bends) {
    try {
      const params = new URLSearchParams({
        session_id: window.comfySessionId,
        batch_id: window.comfySessionId,
        bends: JSON.stringify(bends),
        steps_min: this.stepsMin != null ? String(this.stepsMin) : "",
        steps_max: this.stepsMax != null ? String(this.stepsMax) : "",
      });
      const r = await fetch(`/web_bend_demo/experiments/lookup?${params}`, { method: "GET" });
      if (!r.ok) return null;
      const data = await r.json();
      if (data.ok && data.hit && data.image_url) return data.image_url;
      return null;
    } catch (e) {
      console.warn("Experiment cache lookup failed:", e);
      return null;
    }
  }

  /**
   * Commit bends to ComfyUI backend and optionally queue a new generation.
   * @param {Array|undefined} bendsArg - Optional bends array; uses getBends() if omitted.
   * @param {{skipQueue?: boolean}} options - If skipQueue is true, update backend but don't trigger ComfyUI queue.
   */
  async sendSelectionToComfyUI(bendsArg, options = {}) {
    if (!window.comfySessionId) return;
    const bends = bendsArg != null ? bendsArg : this.getBends();
    const skipQueue = options.skipQueue === true;
    try {
      if (bends.length > 0) {
        if (!skipQueue && CONFIG.USE_EXPERIMENT_CACHE) {
          const cached = await this._lookupExperimentCache(bends);
          if (cached) {
            this._hasGeneratedBentImage = true;
            this._setOutputImageBg(cached);
            this._updatePreviewLayout();
            this.updateBends();
            const el = this._getOutputParamsTextEl();
            if (el) el.innerHTML = (el.innerHTML || "") + "<br><em>✓ From experiment cache</em>";
            const bendsArr = this.getBends();
            this._pushHistory(cached, bendsArr);
            if (typeof unetVisualizer !== "undefined" && unetVisualizer.updateRubberEffect) {
              unetVisualizer.updateRubberEffect(this.getBendingIntensity());
            }
            return;
          }
        }
        const response = await fetch("/web_bend_demo/selection", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: window.comfySessionId,
            bends,
            steps_min: this.stepsMin,
            steps_max: this.stepsMax,
            max_denoising_steps: this.maxDenoisingSteps,
            selected_part: this.selectedPart,
          }),
        });
        console.log(response);
        if (response.ok) {
          const result = await response.json();
          console.log("Selection sent to ComfyUI:", { bends, change_hash: result.change_hash, skipQueue });
          window.clearComfyError?.();
          this.updateBends();
          if (skipQueue) {
            // Just synced backend state, don't queue or poll
            return;
          }
          const el = this._getOutputParamsTextEl();
          const base = el ? el.innerHTML : "";
          if (el) el.innerHTML = (base || "") + "<br><em>Workflow queued — image will update when ready</em>";
          if (window.parent && window.parent !== window) {
            try {
              window.parent.postMessage({
                type: "web_bend_demo_selection_changed",
                session_id: window.comfySessionId,
                change_hash: result.change_hash,
              }, "*");
            } catch (e) {
              console.log("Could not notify parent window:", e);
            }
          }
          this.startImagePolling();
        } else {
          const msg = await (window.parseErrorFromResponse?.(response) ?? response.text().then((t) => t || response.statusText || "Failed to send selection"));
          console.error("Failed to send selection:", msg);
          window.showComfyError?.("ComfyUI: " + msg);
        }
      } else {
        await fetch("/web_bend_demo/clear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: window.comfySessionId }),
        });
        this.clearBends();
        this.stopImagePolling();
        this._hasGeneratedBentImage = false;
        this._updatePreviewLayout();
        this.updateBends();
      }
    } catch (e) {
      console.error("Error sending selection to ComfyUI:", e);
      window.showComfyError?.("ComfyUI: " + (e.message || String(e)));
    }
  }

  _showProgressBar() {
    const el = document.getElementById("comfy-progress-bar");
    if (el) el.style.display = "block";
  }

  _hideProgressBar() {
    const el = document.getElementById("comfy-progress-bar");
    if (el) el.style.display = "none";
  }

  /**
   * Start polling for generated images from ComfyUI
   */
  startImagePolling() {
    this.stopImagePolling(); // Stop any existing polling
    if (!this._fetchingDefault) this._showProgressBar();

    const pollStartTime = Date.now();
    let pollCount = 0;
    const maxPolls = 60; // Poll for up to 2 minutes (2s intervals)
    
    this.imagePollInterval = setInterval(async () => {
      pollCount++;
      if (pollCount > maxPolls) {
        if (this._fetchingDefault) this._fetchingDefault = false;
        this.stopImagePolling();
        return;
      }
      
      try {
        // Poll ComfyUI's history endpoint directly
        const response = await fetch("/history");
        if (!response.ok) {
          const msg = await (window.parseErrorFromResponse?.(response) ?? response.text().then((t) => t || response.statusText || "Failed to fetch history"));
          window.showComfyError?.("ComfyUI: " + msg);
          this.stopImagePolling();
          return;
        }
        const history = await response.json();
        
        // Extract newest image since polling started
        const newestImage = this.extractNewestImageSince(history, pollStartTime);
        const newestError = this.extractNewestErrorSince(history, pollStartTime);

        if (newestImage && newestImage.filename) {
          const base = `/view?filename=${encodeURIComponent(newestImage.filename)}&subfolder=${encodeURIComponent(newestImage.subfolder || "")}&type=${encodeURIComponent(newestImage.type || "output")}`;
          const imageUrl = `${base}&t=${Date.now()}`;
          if (this._fetchingDefault) {
            this.defaultImageUrl = imageUrl;
            this._fetchingDefault = false;
            this._hideProgressBar();
            this.stopImagePolling();
            window.clearComfyError?.();
            this._updatePreviewLayout();
            return;
          }
          this._hasGeneratedBentImage = true;
          this._setOutputImageBg(imageUrl);
          this._updatePreviewLayout();
          this.updateBends();
          const el = this._getOutputParamsTextEl();
          if (el) el.innerHTML = (el.innerHTML || "") + "<br><em>✓ Image generated</em>";
          const bends = this.getBends();
          this._pushHistory(imageUrl, bends);
          this.stopImagePolling();
          window.clearComfyError?.();
          // Keep rubber effect at current bending intensity after image is generated
          if (typeof unetVisualizer !== "undefined" && unetVisualizer.updateRubberEffect) {
            unetVisualizer.updateRubberEffect(this.getBendingIntensity());
          }
        }

        if (newestError) {
          window.showComfyError?.("ComfyUI: " + newestError);
          this.stopImagePolling();
          return;
        }
      } catch (e) {
        console.warn("Error polling for image:", e);
        window.showComfyError?.("ComfyUI: " + (e.message || String(e)));
      }
    }, 2000); // Poll every 2 seconds
  }

  /**
   * Extract the newest execution error from history since a given timestamp (ms).
   * ComfyUI stores status { status_str: 'error', messages: [...] } on failed runs.
   * @param {Object} historyObj - History from /history
   * @param {number} sinceTs - Timestamp (ms) to filter by
   * @returns {string|null} Error message or null
   */
  extractNewestErrorSince(historyObj, sinceTs) {
    const toMs = (v) => {
      if (v == null || !Number.isFinite(v)) return 0;
      return v < 1e12 ? v * 1000 : v;
    };

    let best = null;
    let bestTs = 0;

    for (const pid of Object.keys(historyObj || {})) {
      const run = historyObj[pid];
      const st = run?.status;
      if (!st || st.status_str !== "error") continue;

      const extra = run?.prompt?.[3];
      const ts = toMs(
        extra?.create_time ?? run?.timestamp ?? run?.time ?? run?.created_at ?? 0
      );
      if (sinceTs && ts && ts < sinceTs) continue;
      if (ts <= bestTs) continue;

      const msgs = st.messages;
      let err = "";
      if (Array.isArray(msgs) && msgs.length) {
        const parts = msgs.map((m) => {
          if (typeof m === "string") return m;
          const d = Array.isArray(m) ? m[1] : m;
          if (d && typeof d === "object" && d.exception_message) return d.exception_message;
          if (d && typeof d === "object") return d.message || d.details || String(d);
          return String(m);
        });
        err = parts.filter(Boolean).join("\n") || "Execution failed";
      } else {
        err = "Execution failed";
      }
      best = err;
      bestTs = ts;
    }
    return best;
  }

  /**
   * Extract the newest image from history since a given timestamp (ms).
   * ComfyUI history stores the queue item in run.prompt; extra_data (create_time) is at index 3.
   * @param {Object} historyObj - History object from ComfyUI
   * @param {number} sinceTs - Timestamp (ms) to filter by
   * @returns {Object|null} Newest image or null
   */
  extractNewestImageSince(historyObj, sinceTs) {
    let best = null;

    const toMs = (v) => {
      if (v == null || !Number.isFinite(v)) return 0;
      return v < 1e12 ? v * 1000 : v;
    };

    for (const pid of Object.keys(historyObj || {})) {
      const run = historyObj[pid];
      const extra = run?.prompt?.[3];
      const ts = toMs(
        extra?.create_time ?? run?.timestamp ?? run?.time ?? run?.created_at ?? 0
      );

      if (sinceTs && ts && ts < sinceTs) continue;

      const outputs = run?.outputs;
      if (!outputs) continue;

      for (const nodeId of Object.keys(outputs)) {
        const out = outputs[nodeId];
        if (!out?.images?.length) continue;

        for (const im of out.images) {
          const candidate = { ...im, _ts: ts || 0 };
          if (!best) best = candidate;
          else if ((candidate._ts || 0) > (best._ts || 0)) best = candidate;
        }
      }
    }

    return best;
  }

  /**
   * Stop polling for generated images
   */
  stopImagePolling() {
    if (this.imagePollInterval) {
      clearInterval(this.imagePollInterval);
      this.imagePollInterval = null;
    }
    this._hideProgressBar();
  }

  async generateImageComfyView(prompt, layerName, angle) {
    var params = {};
    params["31-inputs-input_string"] = layerName;
    params["30-inputs-angle_degrees"] = angle;
    params["6-inputs-text"] = prompt;
    params["22-inputs-steps"] = 2;

    var currentTime = new Date();
    const result = await ViewComfyAPI.infer(
      CONFIG.VIEWCOMFY_API_URL,
      params,
      null,
      CONFIG.VIEWCOMFY_CLIENT_ID,
      CONFIG.VIEWCOMFY_CLIENT_SECRET
    );
    console.log("time:", new Date() - currentTime);
    var imageUrl = result.outputs[0].filepath;
    return imageUrl;
  }

  async generateImageLocalComfy(prompt, layerName, angle) {
    console.log("Generating image locally via ComfyUI...");
    const workflow = {
      5: {
        inputs: { width: 512, height: 512, batch_size: 1 },
        class_type: "EmptyLatentImage",
      },
      6: {
        inputs: { text: prompt, clip: ["20", 1] },
        class_type: "CLIPTextEncode",
      },
      7: {
        inputs: { text: "text, watermark", clip: ["20", 1] },
        class_type: "CLIPTextEncode",
      },
      8: {
        inputs: { samples: ["13", 0], vae: ["20", 2] },
        class_type: "VAEDecode",
      },
      13: {
        inputs: {
          add_noise: true,
          noise_seed: 42,
          cfg: 1,
          model: ["29", 0],
          positive: ["6", 0],
          negative: ["7", 0],
          sampler: ["14", 0],
          sigmas: ["22", 0],
          latent_image: ["5", 0],
        },
        class_type: "SamplerCustom",
      },
      14: {
        inputs: { sampler_name: "euler_ancestral" },
        class_type: "KSamplerSelect",
      },
      20: {
        inputs: { ckpt_name: "sd_xl_turbo_1.0_fp16.safetensors" },
        class_type: "CheckpointLoaderSimple",
      },
      22: {
        inputs: { steps: 1, denoise: 1, model: ["29", 0] },
        class_type: "SDTurboScheduler",
      },
      27: {
        inputs: { filename_prefix: "ComfyUI", images: ["8", 0] },
        class_type: "SaveImage",
      },
      29: {
        inputs: {
          path: ["31", 0],
          model: ["20", 0],
          bending_module: ["30", 0],
        },
        class_type: "Model Bending",
      },
      30: {
        inputs: {
          angle_degrees: angle,
        },
        class_type: "Rotate Module (Bending)",
        _meta: {
          title: "Rotate Module (Bending)",
        },
      },
      31: { inputs: { value: layerName }, class_type: "PrimitiveString" },
    };

    const body = { prompt: workflow };

    // Step 1: Send workflow to ComfyUI
    const response = await fetch(
      `${CONFIG.COMFY_GENERATION_SERVER_URL}/prompt`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok)
      throw new Error(`ComfyUI server error: ${response.status}`);
    const result = await response.json();
    const promptId = result.prompt_id;

    // Step 2: Poll for image
    let imageUrl = null;
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const history = await fetch(
        `${CONFIG.COMFY_GENERATION_SERVER_URL}/history/${promptId}`
      );
      if (history.ok) {
        const historyJson = await history.json();
        const entry = Object.values(historyJson)[0];
        if (entry?.outputs?.["27"]?.images?.length) {
          const img = entry.outputs["27"].images[0];
          imageUrl = `${CONFIG.COMFY_GENERATION_SERVER_URL}/view?filename=${img.filename}&subfolder=${img.subfolder}&type=${img.type}`;
          break;
        }
      }
    }

    if (!imageUrl) throw new Error("Image generation timed out or failed");

    return imageUrl;
  }

  /**
   * Get current prompt from UI
   * @returns {string} Current prompt text
   */
  getCurrentPrompt() {
    const promptInput = document.getElementById("prompt_input");
    return promptInput
      ? promptInput.value
      : (CONFIG.EXPERIMENTS?.[this.experimentName] || "A beautiful landscape");
  }

  /**
   * Show loading state while generating image
   */
  showLoadingState() {
    this.outputImage.style.opacity = "0.5";
    this.outputImage.style.filter = "blur(2px)";

    // Add loading indicator if it doesn't exist
    if (!document.getElementById("loading-indicator")) {
      const loadingDiv = document.createElement("div");
      loadingDiv.id = "loading-indicator";
      loadingDiv.innerHTML =
        '<div class="loading-spinner">Generating image...</div>';
      loadingDiv.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 8px;
                z-index: 1000;
            `;
      this.outputImage.parentElement.style.position = "relative";
      this.outputImage.parentElement.appendChild(loadingDiv);
    }
  }

  /**
   * Hide loading state
   */
  hideLoadingState() {
    this.outputImage.style.opacity = "1";
    this.outputImage.style.filter = "none";

    const loadingIndicator = document.getElementById("loading-indicator");
    if (loadingIndicator) {
      loadingIndicator.remove();
    }
  }

  /**
   * Show error state
   */
  showErrorState() {
    this.hideLoadingState();

    var errorImage =
      "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iI2NjYyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkVycm9yIGdlbmVyYXRpbmcgaW1hZ2U8L3RleHQ+PC9zdmc+";

    this._setOutputImageBg(errorImage);
    this._getOutputParamsTextEl().innerHTML = "";
    const copyBtn = document.getElementById("copy-bends-btn");
    if (copyBtn) copyBtn.style.display = "none";

  }

  /**
   * Get current bending angle (for rotate) or legacy. Used for rubber effect scale.
   * @returns {number} Max angle (0-270) for rubber effect
   */
  getBendingAngle() {
    let maxAngle = 0;
    this.bends.forEach((entry) => {
      if (entry?.module_type === "rotate") {
        const a = entry.module_args?.angle_degrees ?? 0;
        if (a > maxAngle) maxAngle = a;
      }
    });
    if (maxAngle > 0) return maxAngle;
    return this.bendingAngle;
  }

  /** Get 0-1 intensity for rubber effect based on distance from default value */
  getBendingIntensity() {
    let max = 0;
    this.bends.forEach((entry) => {
      if (!entry) return;
      const t = CONFIG.BENDING_TYPES?.[entry.module_type];
      const k = t?.module_args_key;
      const v = k ? entry.module_args?.[k] : undefined;
      const def = t?.defaultValue;
      const levels = t?.levels;
      if (v === undefined || v === def) return;
      if (!levels || levels.length <= 1) return;
      const defIdx = levels.indexOf(def);
      const valIdx = levels.indexOf(v);
      if (defIdx < 0 || valIdx < 0) return;
      const stepsFromDefault = Math.abs(valIdx - defIdx);
      const maxSteps = levels.length - 1;
      const intensity = maxSteps > 0 ? stepsFromDefault / maxSteps : 0;
      max = Math.max(max, intensity);
    });
    return Math.min(1, max);
  }

  /**
   * Get current layer name
   * @returns {string} Current layer name
   */
  getLayerName() {
    return this.layerName;
  }

  /**
   * Get current experiment name
   * @returns {string} Current experiment name
   */
  getExperimentName() {
    return this.experimentName;
  }
}

// Global instance
const imageManager = new ImageManager();
