 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  clear_comfyui_cache,apply_base_model,load_conditioning_model,infer_dype,phi2narry
import folder_paths
from.DyPE.flux.pipeline_flux import FluxPipeline
from .qwen.pipeline_qwenimage import QwenImagePipeline
from diffusers.hooks import apply_group_offloading
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")
MAX_SEED = np.iinfo(np.int32).max

node_cr_path = os.path.dirname(os.path.abspath(__file__))
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

class DyPE_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "diffusion_models": (["none"] + folder_paths.get_filename_list("diffusion_models"),),
                "gguf": (["none"] + folder_paths.get_filename_list("gguf"),),
                },
            }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "DyPE"
    
    def main(self,diffusion_models,gguf,):
        model=apply_base_model(diffusion_models,gguf)
        return (model,)


class DyPE_Condition:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "ip_adpter": (["none"] +folder_paths.get_filename_list("photomaker"),),
                    "lora1": (["none"] +folder_paths.get_filename_list("loras"),),
                    "lora2": (["none"] +folder_paths.get_filename_list("loras"),),
                    "scale1": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.001,}),
                    "scale2": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "round": 0.001,}),
                    }
                }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "DyPE"
    
    def main(self,model,ip_adpter,lora1,lora2,scale1,scale2):
        ip_adpter_path=folder_paths.get_full_path("photomaker", ip_adpter) if ip_adpter!="none" else None      
        model=load_conditioning_model(model,ip_adpter_path,lora1,lora2,[scale1,scale2])
        return (model,)
    
class DyPE_Encoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "latent": ("LATENT",),
                    "vae": (["none"] +folder_paths.get_filename_list("vae"),),
                    }
                }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "DyPE"
    
    def main(self,latent,vae):
        clear_comfyui_cache()
        latents=latent["samples"]
        from diffusers import AutoencoderKLFlux2
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
        from safetensors.torch import load_file as load_safetensors
        image_processor = Flux2ImageProcessor(vae_scale_factor=16)
        vae_dict=load_safetensors(folder_paths.get_full_path("vae", vae))
        vae=AutoencoderKLFlux2.from_config(AutoencoderKLFlux2.load_config(os.path.join(node_cr_path, "flux2_klein/4B/vae/config.json")))
        vae.load_state_dict(vae_dict,strict=False,assign=True)
        vae.eval().to(device, torch.bfloat16) 
        del vae_dict
        vae.use_tiling  =True
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        # torch.save(latents_bn_mean, "latents_bn_mean.pt")
        # torch.save(latents_bn_std, "latents_bn_std.pt")
        latents = latents * latents_bn_std + latents_bn_mean
        def unpatchify_latents(latents):
            batch_size, num_channels_latents, height, width = latents.shape
            latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
            latents = latents.permute(0, 1, 4, 2, 5, 3)
            latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
            return latents
        latents = unpatchify_latents(latents)
        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")[0]
        return (phi2narry(image),)     

class DyPE_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,}),
                "positive": ("CONDITIONING", ),
                "offload":("BOOLEAN", {"default": True}),
                "num_blocks_per_group": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display": "number"}),
            },
            "optional": {
                "negative": ("CONDITIONING", ),
            },
                
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DyPE"


    def sample(self, model,width,height, seed, steps, cfg, positive,offload,num_blocks_per_group, **kwargs):
        clear_comfyui_cache()
        if offload:
            apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=num_blocks_per_group,)
        else:
            model.enable_model_cpu_offload()
        ip_adapter_image_embeds=positive[0][1].get("unclip_conditioning",None)
        ip_adapter_image_embeds=[ip_adapter_image_embeds[0]["clip_vision_output"]["image_embeds"]] if ip_adapter_image_embeds is not None else None
        negative=kwargs.get("negative", None)
        if isinstance(model, FluxPipeline):
            samples=infer_dype(
                model,
                ip_adapter_image_embeds=ip_adapter_image_embeds, #image_embeds.
                prompt_embeds=positive[0][0],
                pooled_prompt_embeds=positive[0][1].get("pooled_output"),
                negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                seed=seed,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                )
        elif isinstance(model, QwenImagePipeline):          
            with torch.inference_mode():
                samples = model(
                    true_cfg_scale=cfg,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    prompt_embeds=positive[0][0],
                    negative_prompt_embeds=negative[0][0] if negative is not None else torch.zeros_like(positive[0][0]),
                    seed=seed,
                    ).images
        else :
            with torch.inference_mode():
                samples = model(                    
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    prompt_embeds=positive[0][0].to(device),
                    negative_prompt_embeds=negative[0][0].to(device) if negative is not None else torch.zeros_like(positive[0][0]).to(device),
                    ).images
                print(samples.shape)

        out = {}
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DyPE_Model": DyPE_Model,
    "DyPE_Condition": DyPE_Condition,
    "DyPE_Encoder": DyPE_Encoder,
    "DyPE_KSampler":DyPE_KSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DyPE_Model": "DyPE_Model",
    "DyPE_Condition": "DyPE_Condition",
    "DyPE_Encoder": "DyPE_Encoder",
    "DyPE_KSampler":"DyPE_KSampler",
}
