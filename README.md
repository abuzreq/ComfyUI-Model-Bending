# Diffusion Model Bending
This project implements [model bending](https://github.com/terrybroad/network-bending) techniques for diffusion models in ComfyUI. Model bending allows you to apply transformations to the inner workings of your model in order to push it towards new and diverse aesthetics. You can see it as providing granular controls for introducing diversity and randomization beyond your simple randomization seeds. Transformations can include addition, multiplication, noise, rotation, erosion, dilation, to name a few, ... or even your own custom ones.

This project provides multiple ways to achieve bending:
### 1. Model Bending: [[Workflow](workflows/basic_unet_bending.json)]
Inject bending modules (torch.nn) into your models (specifically, your UNet but Comfy calls them MODEL). Plug in your model, pick or write a bending module, and choose the layer in your model where you would like to inject the bending module. The Model Inspector node is available to help you experiment with different layers. 

An example of using the Rotate module to the middle block (middle_block.2.out_layers) of a UNet of the realisticvisionv51_v51vae model, doing a full rotation (0-360deg).
![image](docs/imgs/bending_add_analog_portrait.gif)

Another example of injecting an Add Scalar module to the middle block (middle_block.2.out_layers) of a UNet of the sd_xl_turbo model. Amounts added range from -10 to 30, while freezing everything else (prompt, seed, ... etc.).
![image](docs/imgs/bending_rotate_analog_portrait.gif)

### 2. VAE Bending [[Workflow](workflows/vae_bending.json)]
Inject bending modules into your VAE models (Not as strong results as above but I haven't experimented enough to confirm). 
### 3. Conditionings x Operations  [[Workflow](workflows/conditioning_bending.json)]
Apply transforming operations to conditionings (what comes out of encoding with CLIP), this helps you move them around in the semantic latent space (i.e. that of CLIP's encodings)
### 3. CFG Step-wise Operations [[Workflow](workflows/denoising_step_bending.json)]
Apply transforming operations to the intermediate latents in the sampling/denoising process. In particular, you are asking your KSampler to apply a transformation right before a chosen cdf scaling step. Comfy currently provides an experimental support for applying transformation during all denoising steps, e.g., see LatentApplyOperationCFG and LATENT_OPERATION, where as the provided node (LatentApplyOperationCFGToStep) picks one step.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
4. Restart ComfyUI.
5. All the nodes can be found under model_bending/

## Notes
This is an ongoing project. I am excited to share it with you and happy to respond to [issues](https://github.com/abuzreq/ComfyUI-Model-Bending/issues) and features requests as time permits. 

## Available Nodes
| **    Node Name   **                        |       **    Category   **       |
|---------------------------------------------|:-------------------------------:|
|     Model   Bending                         |             UNet (MODEL)        |
|     Model VAE   Bending                     |                VAE              |
|     Model   Inspector                       |              MODEL              |
|     Add Noise   Module (Bending)            |             MODEL/VAE           |
|     Add   Scalar Module (Bending)           |            MODEL/VAE            |
|     Multiply   Scalar Module (Bending)      |            MODEL/VAE            |
|     Threshold   Module (Bending)            |            MODEL/VAE            |
|     Rotate   Module (Bending)               |            MODEL/VAE            |
|     Scale Module   (Bending)                |            MODEL/VAE            |
|     Erosion   Module (Bending)              |            MODEL/VAE            |
|     Gradient   Module (Bending)             |            MODEL/VAE            |
|     Dilation   Module (Bending)             |            MODEL/VAE            |
|     Sobel   Module (Bending)                |            MODEL/VAE            |
|     LatentApplyOperationCFGToStep           |     Sampling pre-CFG (LATENT)   |
|     ConditioningApplyOperation              |        CLIP (CONDITIONING)      |
|     Latent   Operation (Multiply Scalar)    |       LATENT/CONDITIONING       |
|     Latent   Operation (Add Scalar)         |       LATENT/CONDITIONING       |
|     Latent   Operation (Threshold)          |       LATENT/CONDITIONING       |
|     Latent   Operation (Rotate)             |       LATENT/CONDITIONING       |
|     Latent   Operation (Add Noise)          |       LATENT/CONDITIONING       |
|     Latent   Operation (Custom)             |       LATENT/CONDITIONING       |
