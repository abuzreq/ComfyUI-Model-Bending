import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "model_bending",
    async afterConfigureGraph() {
        function messageHandler(event) {
            // When the model is loaded we grab it's structure from the backend and visualize it
            for (var n of app.graph._nodes) {
                if (n.title === "Model Inspector" || n.title === "Model VAE Inspector") {
                    let treeData = convertJSONTree(
                        JSON.parse(event.detail.tree),
                        undefined,
                        ""
                    );

                    let hasChanged = !deepCompare(
                        treeData,
                        n.modelViz.treeData
                    );

                    if (hasChanged) {
                        n.modelViz.treeData = treeData;
                        n.modelViz.typeSet = undefined; // Reset the typeSet to force recomputation
                        n.modelViz.renderTree(); // Call the renderTree method to update the visualization
                    }
                }
                
            }
        }

        function updateNumLayers(event) {
            for (var n of app.graph._nodes) {
                if (n.title === "Model Bending (SD Layers)") {
                    console.log(event.detail.num_layers);
                    let numLayers = event.detail.num_layers;
                    const layerNumWidget = n.widgets.find(
                        (w) => w.name === "layer_num"
                    );

                    layerNumWidget.options.max = numLayers - 1;
                    if (layerNumWidget.value >= numLayers) {
                        layerNumWidget.value = numLayers - 1;
                    }
                }
            }
        }
        api.addEventListener("model_bending.inspect_model", messageHandler);
        api.addEventListener("model_bending.bend_sd_model", updateNumLayers);
    },

    nodeCreated(node, app) {
        if (node.title === "Model Inspector" || node.title === "Model VAE Inspector") {
            const placeholderWidget = node.widgets.find(
                (w) => w.name === "path_placeholder"
            );
            placeholderWidget.disabled = true;
            node.size = [450, 500];

            node.onResize = (newSize) => {
                if (Object.hasOwn(node, "modelViz")) {
                    node.modelViz.updateSize(newSize[0], newSize[1]);
                }
            };

            node.modelViz = new ModelViz(
                app,
                app.ctx,
                app.ctx.canvas,
                node,
                null
            );

            node.onDrawForeground = function (ctx, graphcanvas) {
                if (node.modelViz != undefined && node.modelViz != null) {
                    node.modelViz.ctx = ctx;
                    node.modelViz.canvas = ctx.canvas;
                    node.modelViz.renderTree();
                }
            };          
  
        } else if (node.title === "Latent Operation (Custom)") {
            // Locate the operation widget and other parameter widgets by name
            const operationWidget = node.widgets.find(
                (w) => w.name === "operation"
            );
            const float1Widget = node.widgets.find(
                (w) => w.name === "float_param"
            );
            const float2Widget = node.widgets.find(
                (w) => w.name === "float_param2"
            );
            const int1Widget = node.widgets.find((w) => w.name === "int_param");
            const boolWidget = node.widgets.find(
                (w) => w.name === "bool_param"
            );
            console.log("operationWidget", operationWidget);
            // Define a function to update widget visibility based on the operation value

            const int_operations = ["reflect", "dilation", "erosion"];
            const none_operations = [
                "absolute",
                "log",
                "gradient",
                "hadamard1",
            ];
            const dual_float_operations = ["clamp", "scale"];

            const bool_operations = ["sobel"];

            function updateParameterVisibility() {
                const op = operationWidget.value;
                console.log("operationWidget", operationWidget);

                float1Widget.name = "float_param";
                if (int_operations.includes(op)) {
                    float1Widget.disabled = true;
                    float2Widget.disabled = true;
                    int1Widget.disabled = false;
                    boolWidget.disabled = true;
                } else if (bool_operations.includes(op)) {
                    float1Widget.disabled = true;
                    float2Widget.disabled = true;
                    int1Widget.disabled = true;
                    boolWidget.disabled = false;
                    boolWidget.name = "normalize";
                } else if (none_operations.includes(op)) {
                    float1Widget.disabled = true;
                    float2Widget.disabled = true;
                    int1Widget.disabled = true;
                    boolWidget.disabled = true;
                } else if (dual_float_operations.includes(op)) {
                    float1Widget.disabled = false;
                    float1Widget.name = "min";
                    float2Widget.disabled = false;
                    float2Widget.name = "max";
                    int1Widget.disabled = true;
                    boolWidget.disabled = true;
                } else {
                    float1Widget.disabled = false;
                    float2Widget.disabled = true;
                    int1Widget.disabled = true;
                    boolWidget.disabled = true;
                }
            }
            updateParameterVisibility();

            // Attach an onChange handler to the operation widget to trigger updates
            operationWidget.callback = updateParameterVisibility;
        }
    },
});

class ModelViz {
    constructor(app, ctx, canvas, comfynode, treeData) {
        // Layout settings
        this.nodeRowHeight = 18; // Height for each node row
        this.rowIndent = 20; // Indentation per tree level
        this.rowPadding = 5; // Left padding for text and caret
        this.caretSize = 14; // Size of the caret icon
        this.offsetXInNode = 10;
        this.offsetYInNode = 85;
        // Array to hold the computed visible nodes with their positions
        this.visibleNodes = [];

        this.app = app;
        this.ctx = ctx;
        this.canvas = canvas;

        this.comfynode = comfynode;
        this.treeData = treeData;
        console.log(this.ctx, this.canvas, this.app);


        // Initial render of the tree view.
        this.renderTree();

        this.canvas.addEventListener("click", (event) => {
            this.app.canvas.adjustMouseEvent(event);

            const clickX =
                event.canvasX - this.comfynode._pos[0] - this.offsetXInNode;
            const clickY =
                event.canvasY - this.comfynode._pos[1] - this.offsetYInNode;

            for (const item of this.visibleNodes) {
                const { node, depth, y } = item;

                // Calculate caret coordinates as in the drawing code.
                const caretX = depth * this.rowIndent + this.rowPadding;
                const caretY = y + this.nodeRowHeight / 2;

                // Optionally, add a small margin for easier clicking.
                const margin = 5;
                const region = {
                    x: caretX - margin,
                    y: caretY - this.caretSize / 2 - margin,
                    width: this.comfynode._size[0] - caretX,
                    height: this.caretSize + 2 * margin,
                };
                if (
                    clickX >= region.x &&
                    clickX <= region.x + region.width &&
                    clickY >= region.y &&
                    clickY <= region.y + region.height
                ) {
                                console.log("click", event,this.canvas)
                    for (const item of this.visibleNodes) {
                        const { node, depth, y } = item;
                        node.selected = false;
                    }

                    // Toggle the collapsed state and re-render.
                    this.comfynode.setOutputData(0, node.path);
                    var pathPlaceholder = this.comfynode.widgets.find(
                        (w) => w.name == "path_placeholder"
                    );
                    pathPlaceholder.value = node.path;
                    // navigator.clipboard.writeText(node.path);
                    node.selected = true;

                    if (node.children) {
                        node.collapsed = !node.collapsed;
                    }

                    this.renderTree();
                    break;
                }
            }
        });
    }

    /**
     * Recursively computes visible nodes based on expansion state.
     * @param {Object} node - The current node.
     * @param {number} depth - The current tree depth.
     * @param {number} yPos - The current y-coordinate in the canvas.
     * @param {Array} visibleList - Array that accumulates visible nodes.
     * @returns {number} - Updated y-coordinate after processing this node.
     */
    computeVisibleNodes(node, depth = 0, yPos = 0, visibleList = []) {
        visibleList.push({ node, depth, y: yPos });
        let currentY = yPos + this.nodeRowHeight;
        if (node.children && !node.collapsed) {
            for (const child of node.children) {
                currentY = this.computeVisibleNodes(
                    child,
                    depth + 1,
                    currentY,
                    visibleList
                );
            }
        }
        return currentY;
    }
    updateSize(newNodeWidth, newNodeHeight) {
        // this.nodeWidth = newNodeWidth;
        // this.nodeHeight = newNodeHeight;
        this.renderTree();
    }

    processNodeTypes(node, typeSet = new Set()) {
      
        if (node && node.type) {
            typeSet.add(node.type);
            if (node.children) {
                for (const child of node.children) {
                    typeSet.add(child.type);
                    this.processNodeTypes(
                        child,
                        typeSet
                    );
                }
            }
        }
    }



    /**
     * Renders the tree on the canvas.
     */
    renderTree() {
        this.ctx.clearRect(
            0,
            0,
            this.comfynode._size[0],
            this.comfynode._size[1]
        );
        this.visibleNodes = [];
        if (!this.treeData) {
            return;
        }
        else 
        {
            if (this.typeSet == undefined) {
                this.typeSet = new Set();
                this.processNodeTypes(this.treeData, this.typeSet)
                console.log(this.typeSet);
        
                //https://sashamaps.net/docs/resources/20-colors/
                //var colorMap = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'];
                // https://gist.github.com/Myndex/997244b95d84788df96f4aab8b9edeb1
                var colorMap = [
                    '#fdfdfd', '#1d1d1d', '#ebce2b', '#702c8c',
                    '#db6917', '#96cde6', '#96cde6', '#ba1c30',
                    '#c0bd7f', '#7f7e80', '#7f7e80', '#5fa641',
                    '#d485b2', '#d485b2', '#4277b6', '#df8461',
                    '#df8461', '#df8461', '#df8461', '#463397',
                    '#463397', '#463397', '#e1a11a', '#e1a11a',
                    '#e1a11a', '#91218c', '#91218c', '#91218c',
                    '#e8e948', '#e8e948', '#e8e948', '#7e1510',
                    '#7e1510', '#92ae31', '#92ae31', '#92ae31',
                    '#6f340d', '#6f340d', '#6f340d', '#d32b1e',
                    '#d32b1e', '#d32b1e', '#2b3514', '#2b3514',
                    '#2b3514'
                  ]
                
                this.typesToColorMap = {};
                [...this.typeSet].forEach((type, index) => {
                    this.typesToColorMap[type] = colorMap[index % colorMap.length];
                 
                });
            }
        }

        this.computeVisibleNodes(this.treeData, 0, 0, this.visibleNodes);
       
        // Draw the tree nodes

        this.visibleNodes.forEach((item) => {
            const { node, depth, y } = item;
            const x = depth * this.rowIndent + this.offsetXInNode;
            var ny = y + this.offsetYInNode;
            // Optional: draw a white background for clarity.
            this.ctx.fillStyle = "#fff";

            // this.ctx.fillStyle = node.type in this.typesToColorMap && !node.children? this.typesToColorMap[node.type]: '#fff'; 

            this.ctx.fillRect(
                0,
                ny,
                this.comfynode._size[0],
                this.nodeRowHeight - 1
            );

            // Draw the node text (shift text right if there is a caret)
            this.ctx.fillStyle = node.selected ? "#00f" : "#000";
            this.ctx.strokeStyle = "#000";

            this.ctx.font = node.selected
                ? "bold 14px sans-serif"
                : "14px sans-serif";
            this.ctx.textBaseline = "middle";
            const textX =
                x + this.rowPadding + (node.children ? this.caretSize + 5 : 0);
            this.ctx.fillText(
                node.name + ": " + node.type,
                textX,
                ny + this.nodeRowHeight / 2
            );

            // If node has children, draw a caret icon.
            if (node.children) {
                const caretX = x + this.rowPadding;
                const caretY = ny + this.nodeRowHeight / 2;
                this.ctx.beginPath();
                if (node.collapsed) {
                    // Draw right-pointing caret (collapsed state)
                    this.ctx.moveTo(caretX, caretY - this.caretSize / 2);
                    this.ctx.lineTo(caretX, caretY + this.caretSize / 2);
                    this.ctx.lineTo(caretX + this.caretSize, caretY);
                } else {
                    // Draw down-pointing caret (expanded state)
                    this.ctx.moveTo(caretX, caretY - this.caretSize / 2);
                    this.ctx.lineTo(
                        caretX + this.caretSize,
                        caretY - this.caretSize / 2
                    );
                    this.ctx.lineTo(
                        caretX + this.caretSize / 2,
                        caretY + this.caretSize / 2
                    );
                }
                this.ctx.closePath();
                this.ctx.fillStyle = "#000";
                this.ctx.fill();
            }
        });
    }
}

/**
 * Converts a JSON object with keys "type" and "children" (as an object)
 * into an internal tree structure where each node has:
 * - name: a label (if a key is provided, it will be "key: type")
 * - collapsed: initially false
 * - children: an array of converted children (if any)
 *
 * @param {Object} json - The JSON node.
 * @param {string|null} key - An optional key for the node (if it's a child).
 * @returns {Object} - The converted node.
 */
function convertJSONTree(json, key = null, path) {
    const nodeName = key ? key : json.type;
    let newNode = {
        name: nodeName,
        path: path ? path + "." + nodeName : nodeName,
        type: json.type,
        collapsed: true,
        selected: false,
    };
    if (json.children) {
        newNode.children = [];
        // json.children is assumed to be an object with keys as child names.
        for (const [childKey, childValue] of Object.entries(json.children)) {
            newNode.children.push(
                convertJSONTree(childValue, childKey, newNode.path)
            );
        }
    }
    return newNode;
}

function deepCompare(obj1, obj2, excludedKeys = ["collapsed", "selected"]) {
    if (typeof obj1 !== typeof obj2) return false;

    if (typeof obj1 === "object" && obj1 !== null && obj2 !== null) {
        const keys1 = Object.keys(obj1).filter(
            (key) => !excludedKeys.includes(key)
        );
        const keys2 = Object.keys(obj2).filter(
            (key) => !excludedKeys.includes(key)
        );

        if (keys1.length !== keys2.length) return false;

        for (let key of keys1) {
            if (
                !keys2.includes(key) ||
                !deepCompare(obj1[key], obj2[key], excludedKeys)
            ) {
                return false;
            }
        }
        return true;
    }

    return obj1 === obj2;
}
