# ComfyUI_DyPE
[DyPE](https://github.com/guyyariv/DyPE):  Dynamic Position Extrapolation for Ultra High Resolution Diffusion ,you can use a wrapper node it in comfyUI

Upadte
-----
* add z image,qwen-image,flux2-klein9/4B support，klein need diffuser>0.37
* 新增z-image，千问，克莱因9/4B的dype支持, klein需要最新版的diffuser


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
* 3.4 ip adapter [xflux](https://huggingface.co/XLabs-AI/flux-ip-adapter)
* 3.5 clip_vision [ open](https://huggingface.co/openai/clip-vit-large-patch14)
* qwen/zimage/Klein use comfyUI split models，only klein vae need diffuser version
```
├── ComfyUI/models/diffusion_models
|     ├── flux1-krea-dev-fp8.safetensors   #  or  flux1-krea-dev.safetensors # flux1
|     ├── FLUX.2-klein-4B.safetensors  #  or 9B or gguf   #flux2
|     ├──Qwen-Image-2512-bf16.safetensors  or gguf  # qwen
|     ├──z_image_turbo_bf16.safetensors or gguf  #zimage
├── ComfyUI/models/vae
|        ├──ae.safetensors or UltraFlux.safetensors  ##flux1 and z image
|        ├──flux2_4b.safetensors  # flux2 klein need diffuser version,don't  support  comfyUI version 需要diffuser的模型
|        ├──qwen_image_vae.safetensors #qwen
├── ComfyUI/models/clip
|        ├──clip_l.safetensors  #flux1
|        ├──t5xxl_fp8_e4m3fn.safetensors #flux1
|        ├──qwen_3_4b.safetensors # z image flux2 klein4b
|        ├──qwen_3_8b_fp8mixed.safetensors # flux2 klein9b
|        ├──qwen_2.5_vl_7b.safetensors #qwen
├── ComfyUI/models/lora
|        ├──flux_turbo.safetensors # flux1
|        ├──flux_real.safetensors #flux1
|        ├──Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors  #qwen
├── ComfyUI/models/photomaker  #ip only flux1
|        ├──ip_adapter.safetensors 
├── ComfyUI/models/clip_vision #ip only flux1
|        ├──clip-vit-large-patch14.safetensors
```
  
4.Example
-----
* z-image
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/zimage.png)
* 9B Klein
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/9B.png)
* 4B Klein
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/4B.png)
* qwen
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/dype_qwen.png)
* t2i flux1
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/example111.png)
* i2i flux2
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/example_ip.png)
* example flux1
![](https://github.com/smthemex/ComfyUI_DyPE/blob/main/example_workflows/ComfyUI_00008_.png)

5.License and Commercial Use
-----
This work is patent pending. For commercial use or licensing inquiries, please contact the [authors](mailto:noam.issachar@mail.huji.ac.il).

6.Citation
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
```
