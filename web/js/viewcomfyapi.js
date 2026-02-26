// viewcomfy-browser.js
// Requires socket.io-client available as window.io (CDN) or replace with an import from a bundler.

class ViewComfyAPI {
  static API_URL = "https://api.viewcomfy.com";

  static uuidv4() {
    const a = crypto.getRandomValues(new Uint8Array(16));
    a[6] = (a[6] & 0x0f) | 0x40;
    a[8] = (a[8] & 0x3f) | 0x80;
    const hex = [...a].map((b) => b.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
      .slice(6, 8)
      .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
  }

  static buildFormData(
    params,
    overrideWorkflowApi,
    prompt_id,
    viewComfyApiUrl,
    sid
  ) {
    const formData = new FormData();
    const params_str = {};
    for (const key in params) {
      const value = params[key];
      if (value instanceof File) formData.set(key, value);
      else params_str[key] = value;
    }

    formData.set("params", JSON.stringify(params_str));
    formData.set("prompt_id", prompt_id);
    formData.set("view_comfy_api_url", viewComfyApiUrl);
    formData.set("sid", sid);
    if (overrideWorkflowApi) {
      formData.set("workflow_api", JSON.stringify(overrideWorkflowApi));
    }
    console.log("formData", formData.getAll("view_comfy_api_url"));
    return formData;
  }

  static async fetchInferenceByIds(clientId, clientSecret, promptIds) {
    try {
   
      const meta = await ViewComfyAPI.fetchInferenceByIds({
        clientId,
        clientSecret,
        promptIds: promptIds, // or multiple ids
      });
      console.log("Follow-up fetch (by prompt_ids):", meta);
    } catch (e) {
      console.log("Follow-up fetch failed:", e);
    }
  }

  static infer(
    viewComfyApiUrl,
    params,
    overrideWorkflowApi,
    clientId,
    clientSecret
  ) {
    
    if (!viewComfyApiUrl) throw new Error("viewComfyApiUrl is not set");
    if (!clientId) throw new Error("clientId is not set");
    if (!clientSecret) throw new Error("clientSecret is not set");
    const auth = { client_id: clientId, client_secret: clientSecret };

    return new Promise((resolve, reject) => {
      const prompt_id = ViewComfyAPI.uuidv4();
      let isWorkflowExecuted = false;
      let socket;
      try {
        socket = io(ViewComfyAPI.API_URL, { auth, transports: ["websocket"] });
      } catch (err) {
        return reject(err);
      }

      let loadingInterval;
      const cleanup = (result) => {
        if (loadingInterval) clearInterval(loadingInterval);
        try {
          socket.disconnect();
        } catch {}
        resolve(result);
      };
      
      socket.on("connect", async () => {
        const formData = ViewComfyAPI.buildFormData(
          params,
          overrideWorkflowApi,
          prompt_id,
          viewComfyApiUrl,
          socket.id
        );
       
        try {
          const resp = await fetch(
            `${ViewComfyAPI.API_URL}/api/workflow/infer`,
            {
              method: "POST",
              body: formData,
              redirect: "follow",
              headers: auth,
            }
          );

          if (!resp.ok) {
            const msg = `Failed to fetch viewComfy: ${resp.status} ${
              resp.statusText
            }, ${await resp.text()}`;
            try {
              socket.disconnect();
            } catch {}
            return reject(new Error(msg));
          }
          await resp.json(); // server ack; events will come via socket
          loadingInterval = setInterval(() => {}, 1000); // optional spinner hook
        } catch (err) {
          try {
            socket.disconnect();
          } catch {}
          reject(err);
        }
      });

      socket.on("connect_error", (err) => {
        if (loadingInterval) clearInterval(loadingInterval);
        try {
          socket.disconnect();
        } catch {}
        reject(err);
      });

      socket.on("disconnect", (reason) => {
        if (reason !== "io client disconnect") {
          if (loadingInterval) clearInterval(loadingInterval);
          reject(new Error(`Socket disconnected unexpectedly: ${reason}`));
        }
      });

      const E = {
        LogMessage: "infer_log_message",
        ErrorMessage: "infer_error_message",
        ExecutedMessage: "infer_executed_message",
        ResultMessage: "infer_result_message",
        CanceledInference: "infer_canceled_message",
      };

      socket.on(E.LogMessage, () => {});
      socket.on(E.ErrorMessage, (data) => {
        if (loadingInterval) clearInterval(loadingInterval);
        try {
          socket.disconnect();
        } catch {}
        if (!isWorkflowExecuted) reject(new Error(JSON.stringify(data)));
      });
      socket.on(E.ExecutedMessage, () => {});
      socket.on(E.ResultMessage, (data) => {
        isWorkflowExecuted = true;
        if (data) cleanup(new PromptResult(data));
        else cleanup(null);
      });
      socket.on(E.CanceledInference, () => {
        isWorkflowExecuted = true;
        cleanup(null);
      });

      // attach last prompt id to function (for convenience)
      ViewComfyAPI.infer._lastPromptId = prompt_id;
    });
  }

  static async inferCancel(clientId, clientSecret, promptId, viewComfyApiUrl) {
    if (!viewComfyApiUrl) throw new Error("viewComfyApiUrl is not set");
    if (!clientId) throw new Error("clientId is not set");
    if (!clientSecret) throw new Error("clientSecret is not set");
    if (!promptId) throw new Error("promptId is not set");

    const headers = {
      client_id: clientId,
      client_secret: clientSecret,
      "content-type": "application/json",
    };

    const resp = await fetch(
      `${ViewComfyAPI.API_URL}/api/workflow/infer/cancel`,
      {
        method: "POST",
        body: JSON.stringify({ promptId, viewComfyApiUrl }),
        headers,
      }
    );
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    return resp.json();
  }
}
class PromptResult {
  constructor(data) {
    this.prompt_id = data.prompt_id;
    this.status = data.status;
    this.completed = data.completed;
    this.execution_time_seconds = data.execution_time_seconds;
    this.prompt = data.prompt;
    this.outputs = data.outputs || [];
  }
}
