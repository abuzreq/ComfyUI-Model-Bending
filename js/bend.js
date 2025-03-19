import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: 'LatentOperation',
    async setup(){
        function messageHandler(event){
            alert(event.detail.message); 
     

        }
    },

    nodeCreated(node, app) {
        console.log("Node created", node);
        if (node.title == "LatentOperation") {

            // Locate the operation widget and other parameter widgets by name
            const operationWidget = node.widgets.find(w => w.name === 'operation');
            const float1Widget = node.widgets.find(w => w.name === 'float_param');
            const float2Widget = node.widgets.find(w => w.name === 'float_param2');
            const int1Widget = node.widgets.find(w => w.name === 'int_param');
            const boolWidget = node.widgets.find(w => w.name === 'bool_param');
            console.log("operationWidget", operationWidget);
            // Define a function to update widget visibility based on the operation value
            
            const int_operations = ["reflect", "dilation", "erosion"]
            const none_operations = ["absolute", "log", "gradient"]
            const dual_float_operations = ["clamp", "scale"]

            const bool_operations = ["sobel"]

            function updateParameterVisibility() {
                return;
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
                }  else if (dual_float_operations.includes(op)) {
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
    }
})
