# Diffusion Bending
This description is a work in progress, but here is a draft.

These nodes implement techniqeus for [model bending](https://arxiv.org/abs/2005.12420), which allow you to apply transformations to the internal workings of your model in order to push it towards new and diverse aesthetics. Typically, model bending is applied to chosen layers in a model, e.g. in [GANs](https://github.com/terrybroad/network-bending) but in this implementation for diffusion model I implement it such that you can apply transformations at chosen denoising steps. 

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

