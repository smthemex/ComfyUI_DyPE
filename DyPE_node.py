 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  nomarl_upscale,apply_base_model,load_conditioning_model,nomarl_upscale,infer_dype,tensor_upscale,tensor2pillist_upscale
import folder_paths
import comfy
import node_helpers
from diffusers.hooks import apply_group_offloading

MAX_SEED = np.iinfo(np.int32).max

node_cr_path = os.path.dirname(os.path.abspath(__file__))
weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

class DyPE_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "diffusion_models": (["none"] + folder_paths.get_filename_list("diffusion_models"),),
            "gguf": (["none"] + folder_paths.get_filename_list("gguf"),),
            "use_dype":("BOOLEAN", {"default": True}),
            "method":(["yarn", "ntk",  "base"],),
            },
            }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "DyPE"
    
    def main(self,diffusion_models,gguf,use_dype,method,):
        model=apply_base_model(diffusion_models,gguf,use_dype,method,)
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
    

class DyPE_Encode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "pos_text": ("STRING", {"multiline": True,"default": "A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic buildings in the distance."}),
            },       
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive", )
    FUNCTION = "encode"
    CATEGORY = "DyPE"


    def encode(self, clip,width,height,pos_text,**kwargs):

      
        tokens_p = clip.tokenize(pos_text)
        postive = clip.encode_from_tokens_scheduled(tokens_p) 

        add_dict={"size":(width,height),}
        postive=node_helpers.conditioning_set_values(postive, {"add_dict": add_dict}) 
        return (postive,)
        

class DyPE_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,}),
                "positive": ("CONDITIONING", ),
                "offload":("BOOLEAN", {"default": True}),
                "num_blocks_per_group": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display": "number"}),
            },
                
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "DyPE"


    def sample(self, model, seed, steps, cfg, positive,offload,num_blocks_per_group, ):
        if offload:
            apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=num_blocks_per_group,)
        else:
            model.enable_model_cpu_offload()
        condition=positive[0][1].get("add_dict")
        width,height=condition.get("size",(1024,1024))
        ip_adapter_image_embeds=positive[0][1].get("unclip_conditioning",None)
        ip_adapter_image_embeds=[ip_adapter_image_embeds[0]["clip_vision_output"]["image_embeds"]] if ip_adapter_image_embeds is not None else None
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
     
        out = {}
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DyPE_Model": DyPE_Model,
    "DyPE_Condition": DyPE_Condition,
    "DyPE_Encode": DyPE_Encode,
    "DyPE_KSampler":DyPE_KSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DyPE_Model": "DyPE_Model",
    "DyPE_Condition": "DyPE_Condition",
    "DyPE_Encode": "DyPE_Encode",
    "DyPE_KSampler":"DyPE_KSampler",
}
