# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import cv2
from omegaconf import OmegaConf

from safetensors.torch import load_file
from comfy.utils import common_upscale
import folder_paths
import node_helpers
import sys 
from.DyPE.flux.transformer_flux import FluxTransformer2DModel as   DyPEFluxTransformer2DModel 

try:
    diffusers_module = sys.modules.get('diffusers')
    if diffusers_module:
        setattr(diffusers_module, 'FluxTransformer2DModel', DyPEFluxTransformer2DModel)
except Exception as e:
    print(f"Warning: Could not register DyPEFluxTransformer2DModel with diffusers module: {e}")


cur_path = os.path.dirname(os.path.abspath(__file__))


def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img



def cv2tensor(img,bgr2rgb=True):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).permute(1, 2, 0).unsqueeze(0)  # 255也可以改为256



def load_lora_for_unet_only(pipeline, lora_path, adapter_name="default", lora_scale=1.0):

    try:
      
        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
        pipeline.set_adapters(adapter_name, adapter_weights=lora_scale)
        print(f"成功加载LoRA权重: {adapter_name} (scale: {lora_scale})")
    except Exception as e:
        print(f"加载LoRA权重失败: {e}")
    
    return pipeline


def load_flux_tansformer(gguf_path,unet_path,use_dype,method,):

    if gguf_path : 
        print("use gguf quantization")
        from diffusers import  GGUFQuantizationConfig
        transformer = DyPEFluxTransformer2DModel.from_single_file(
            gguf_path,
            config=os.path.join(cur_path, "Flux/FLUX.1-Krea-dev/transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

    elif unet_path :
        print("use single unet")

        try:
            transformer =DyPEFluxTransformer2DModel.from_single_file(unet_path,config=os.path.join(cur_path, "Flux/FLUX.1-Krea-dev/transformer"),torch_dtype=torch.bfloat16,)
        except:
            t_state_dict=load_file(unet_path,device="cpu")
            quantization_config = DyPEFluxTransformer2DModel.load_config(os.path.join(cur_path,"Flux/FLUX.1-Krea-dev/transformer/config.json"))
            quantization_config["dype"]=use_dype
            quantization_config["method"]=method
            transformer = DyPEFluxTransformer2DModel.from_config(quantization_config,torch_dtype=torch.bfloat16)
            transformer.load_state_dict(t_state_dict, strict=False)
            del t_state_dict
            gc_cleanup()
    else:
        raise "you must choice a unet or gguf "

    return transformer

def load_conditioning_model(model,lora1,lora2,lora_scales=[1.0,1.0]):

    lora1_path=folder_paths.get_full_path("loras", lora1) if lora1!="none" else None
    lora2_path=folder_paths.get_full_path("loras", lora2) if lora2!="none" else None
    lora_list=[i for i in [lora1_path,lora2_path] if i is not None]
    vae=OmegaConf.load(os.path.join(cur_path,"Flux/FLUX.1-Krea-dev/vae/config.json"))
   
    from.DyPE.flux.pipeline_flux import FluxPipeline
    pipeline = FluxPipeline.from_pretrained(os.path.join(cur_path,"Flux/FLUX.1-Krea-dev"),VAE=vae,vae=None,transformer=model,text_encoder=None,text_encoder_2=None, torch_dtype=torch.bfloat16)
    lora_list=lora_list if lora_list else None

    if lora_list is None:
        return pipeline
    try:
        if len(lora_list)!=len(lora_scales): #sacles  
            lora_scales = lora_scales[:1]
        for idx, (lora_path, scale) in enumerate(zip(lora_list, lora_scales)):
            if lora_path is not None:
                try:
                    pipeline = load_lora_for_unet_only(
                        pipeline, 
                        lora_path, 
                        adapter_name=os.path.basename(lora_path).split('.')[0], 
                        lora_scale=scale
                    )
                except Exception as e:
                    print(f"Failed to apply LoRA {idx+1} ({lora_path}): {str(e)}")
    
        return pipeline
        
    except Exception as e:
        print(f"Failed to apply LoRA {str(e)}")
        return pipeline


def apply_base_model(diffusion_models,gguf,use_dype,method,):

    gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
    unet_path=folder_paths.get_full_path("diffusion_models", diffusion_models) if diffusion_models != "none" else None

    transformer=load_flux_tansformer(gguf_path, unet_path,use_dype,method,) 
    return transformer


def infer_dype(pipeline, latent, prompt_embeds,pooled_prompt_embeds,negative_prompt_embeds,negative_pooled_prompt_embeds,seed, 
                          guidance_scale,num_inference_steps,width,height):

    inputs = {
        "prompt": None,
        "generator": torch.manual_seed(seed),
        "guidance_scale": guidance_scale,
        "negative_prompt": None,
        "num_inference_steps": num_inference_steps,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds":pooled_prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_pooled_prompt_embeds":negative_pooled_prompt_embeds,
        "latents":latent,
        "height": height,
        "width": width,  
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images

    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
    return output_image