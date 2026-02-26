// ComfyUI-Web-Bend-Demo/web/web_bend_demo.js
import { app } from "../../scripts/app.js";

console.log("[web_bend_demo] Extension file loaded");

/** ---------------- Helpers ---------------- **/

function getSelectedNode() {
  const sel = app.canvas?.selected_nodes;
  if (!sel) return null;
  const nodes = Object.values(sel);
  return nodes.length ? nodes[0] : null;
}

function getWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function normalizeSessionId(s) {
  return String(s || "").split("_hash_")[0].split("_ts_")[0];
}

function ensureSessionId(node) {
  const w = getWidget(node, "session_id");
  if (!w) return null;

  if (!w.value) {
    w.value = crypto?.randomUUID?.() ?? `${Date.now()}_${Math.random()}`;
  }

  // Keep widget stable & base-normalized
  w.value = normalizeSessionId(w.value);
  return w.value;
}

function ensureSelectionHashWidget(node) {
  // Optional: requires Python node to define optional input "selection_hash"
  const w = getWidget(node, "selection_hash");
  if (!w) return null;

  if (!w.value) w.value = "";
  return w;
}

function makeModal(title, instructions = "") {
  const modal = document.createElement("div");
  modal.style.position = "fixed";
  modal.style.top = "0";
  modal.style.left = "0";
  modal.style.right = "0";
  modal.style.bottom = "0";
  modal.style.background = "rgba(0,0,0,0.95)";
  modal.style.color = "white";
  modal.style.zIndex = "99999";
  modal.style.display = "flex";
  modal.style.flexDirection = "column";

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "center";
  header.style.padding = "10px 14px";
  header.style.borderBottom = "1px solid rgba(255,255,255,0.15)";
  header.style.gap = "12px";

  const h = document.createElement("div");
  h.textContent = title;
  h.style.fontSize = "16px";
  h.style.fontWeight = "600";
  h.style.flexShrink = "0";

  const instr = document.createElement("div");
  instr.textContent = instructions;
  instr.style.flex = "1";
  instr.style.textAlign = "center";
  instr.style.fontSize = "13px";
  instr.style.color = "rgba(255,255,255,0.85)";

  const close = document.createElement("button");
  close.textContent = "Close";
  close.style.padding = "6px 12px";
  close.style.cursor = "pointer";
  close.style.flexShrink = "0";

  header.appendChild(h);
  header.appendChild(instr);
  header.appendChild(close);
  modal.appendChild(header);

  const content = document.createElement("div");
  content.style.flex = "1";
  content.style.overflow = "hidden";
  content.style.background = "white";
  modal.appendChild(content);

  return { modal, content, close };
}

/** Build the iframe URL for your packaged web UI */
function getIframeUrl(session_id) {
  const baseUrl = window.location.origin;

  const candidates = [
    `${baseUrl}/web_bend_demo/index.html`,
    `${baseUrl}/extensions/ComfyUI-Web-Bend-Demo/index.html`,
  ];

  // Use first candidate; if it 404s you’ll see it in the iframe. (You can add prefetch probing later.)
  const u = new URL(candidates[0]);
  u.searchParams.set("session_id", session_id);
  return u.toString();
}

/** Open the layer picker UI for a specific node (used by context menu and by in-node button) */
async function openPickerForNode(node) {
  if (!node || node.comfyClass !== "InteractiveBendingWebUI") {
    alert("Select the 'Bend From Web UI (Demo)' node first.");
    return;
  }

  const session_id = ensureSessionId(node);
  if (!session_id) {
    alert("Could not find/create session_id widget on this node.");
    return;
  }

  const selectionHashWidget = ensureSelectionHashWidget(node);

  const { modal, content, close } = makeModal(
    "Interactive Diffusion Model Bending",
    "Click the U-Net in the diagram below to expand it, then bend layers using the sliders."
  );
  document.body.appendChild(modal);

  // Iframe
  const iframe = document.createElement("iframe");
  iframe.style.width = "100%";
  iframe.style.height = "100%";
  iframe.style.border = "none";
  iframe.src = getIframeUrl(session_id);
  content.appendChild(iframe);

  // Message handler: selection changed in iframe
  const messageHandler = (event) => {
    const data = event.data;
    if (!data || data.type !== "web_bend_demo_selection_changed") return;

    const msgSession = normalizeSessionId(data.session_id);
    if (msgSession !== session_id) return;

    if (selectionHashWidget) {
      selectionHashWidget.value = String(data.change_hash || Date.now());
    }

    app.graph?.setDirtyCanvas?.(true, true);
    app.queuePrompt(0);
  };

  window.addEventListener("message", messageHandler);

  const cleanup = () => {
    window.removeEventListener("message", messageHandler);
    modal.remove();
  };

  close.onclick = cleanup;

  iframe.onload = () => {
    try {
      iframe.contentWindow?.postMessage(
        { type: "web_bend_demo_init", session_id },
        window.location.origin
      );
    } catch {
      // ignore
    }
  };
}

/** ---------------- Extension ---------------- **/

app.registerExtension({
  name: "web_bend_demo.ui",

  async nodeCreated(node) {
    if (node?.comfyClass !== "InteractiveBendingWebUI") return;

    // Hide session_id and selection_hash widgets visually (still stored)
    const sid = getWidget(node, "session_id");
    if (sid) sid.hidden = true;

    const sh = getWidget(node, "selection_hash");
    if (sh) sh.hidden = true;

    // Add "Open Layer Picker" button inside the node (so user can open UI without context menu)
    try {
      if (typeof node.addWidget === "function") {
        node.addWidget("button", "Open Bending Interface", null, () => openPickerForNode(node));
      } else {
        // Fallback: inject a button into the node's widget area when it becomes available
        const tryInjectButton = () => {
          const container = node.widgets?.[0]?.parentEl ?? node.domElement ?? document.querySelector(`[data-node-id="${node.id}"]`);
          if (!container) return;
          if (container.querySelector(".web-bend-demo-open-btn")) return;
          const btn = document.createElement("button");
          btn.className = "web-bend-demo-open-btn";
          btn.textContent = "Open Layer Picker";
          btn.type = "button";
          btn.style.marginTop = "4px";
          btn.style.padding = "4px 8px";
          btn.style.cursor = "pointer";
          btn.style.width = "100%";
          btn.onclick = (e) => {
            e.stopPropagation();
            openPickerForNode(node);
          };
          container.appendChild(btn);
        };
        setTimeout(tryInjectButton, 0);
        setTimeout(tryInjectButton, 100);
      }
    } catch (e) {
      console.warn("[web_bend_demo] Could not add Open button to node:", e);
    }
  },

  getCanvasMenuItems(canvas) {
    return [
      null,
      {
        content: "Web Bend Demo: Open Layer Picker",
        callback: () => openPickerForNode(getSelectedNode()),
      },
    ];
  },

  getNodeMenuItems(node) {
    if (node?.comfyClass !== "InteractiveBendingWebUI") return [];
    return [
      null,
      {
        content: "Web Bend Demo: Open Layer Picker",
        callback: () => openPickerForNode(node),
      },
    ];
  },
});

