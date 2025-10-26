# ComfyUI_DyPE
[DyPE](https://github.com/guyyariv/DyPE):  Dynamic Position Extrapolation for Ultra High Resolution Diffusion ,you can use a wrapper node it in comfyUI

Upadte
-----
* coming soon, 24G Vram infer 4096*4096

  
1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_DyPE

```

2.requirements  
----

```
pip install -r requirements.txt
```

3.checkpoints 
----

* 3.1 [Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/tree/main)  or [fp8 ](https://huggingface.co/boricuapab/flux1-krea-dev-fp8/tree/main)   don't support scaled model / 不支持scaled 模型   
* 3.2 ae/T5/clip_l / normal flux dev
* 3.3 turbo lora  [alimama-creative](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/tree/main)
  
```
├── ComfyUI/models/diffusion_models
|     ├── flux1-krea-dev-fp8.safetensors   #  or  flux1-krea-dev.safetensors
├── ComfyUI/models/vae
|        ├──ae.safetensors
├── ComfyUI/models/clip
|        ├──clip_l.safetensors
|        ├──t5xxl_fp8_e4m3fn.safetensors
├── ComfyUI/models/lora
|        ├──flux_turbo.safetensors
```
  
4.Example
-----
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/example.png)

5.Citation
-----
```
@misc{issachar2025dype,
      title={DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion}, 
      author={Noam Issachar and Guy Yariv and Sagie Benaim and Yossi Adi and Dani Lischinski and Raanan Fattal},
      year={2025},
      eprint={2510.20766},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.20766}, 
}

``
