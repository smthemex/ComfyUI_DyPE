# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import cv2
from omegaconf import OmegaConf
from contextlib import contextmanager
from safetensors.torch import load_file
from comfy.utils import common_upscale
import folder_paths
from diffusers import  GGUFQuantizationConfig
import sys 
from.DyPE.flux.transformer_flux import FluxTransformer2DModel
from .zimage.transformer_z_image import ZImageTransformer2DModel
from.qwen.transformer_qwenimage import QwenImageTransformer2DModel  
from.flux2.transformer_flux2 import Flux2Transformer2DModel
import comfy.model_management as mm
cur_path = os.path.dirname(os.path.abspath(__file__))


def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")


@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

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



def load_flux_tansformer(gguf_path,dit_path):
    use_dype=True
    method="yarn" #ntk
    if gguf_path : 
        print("use gguf quantization")
        if "flux" in gguf_path.lower():
            if "klein" in gguf_path.lower():
                from.flux2.transformer_flux2 import Flux2Transformer2DModel
                with temp_patch_module_attr("diffusers", "Flux2Transformer2DModel", Flux2Transformer2DModel):
                    repo="4B" if "4b" in gguf_path.lower() else "9B"
                    transformer = Flux2Transformer2DModel.from_single_file(
                        gguf_path,
                        config=os.path.join(cur_path, f"flux2_klein/{repo}/transformer"),
                        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                        torch_dtype=torch.bfloat16,
                    )
            else:
                from.DyPE.flux.transformer_flux import FluxTransformer2DModel 
                with temp_patch_module_attr("diffusers", "FluxTransformer2DModel", FluxTransformer2DModel):
                    transformer = FluxTransformer2DModel.from_single_file(
                        gguf_path,
                        config=os.path.join(cur_path, "Flux/FLUX.1-Krea-dev/transformer"),
                        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                        torch_dtype=torch.bfloat16,
                    )
        elif "qwen" in gguf_path.lower():
            from.qwen.transformer_qwenimage import QwenImageTransformer2DModel  
            with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
                transformer = QwenImageTransformer2DModel.from_single_file(
                    gguf_path,
                    config=os.path.join(cur_path, "Qwen-Image/transformer"),
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                )
        else:
            
            if "turbo" in gguf_path.lower():
                repo="Z-Image-turbo"
                from .zimage.transformer_z_image import ZImageTransformer2DModel
            else:
                repo="Z-Image"
                from .zimage.transformer_z_image_ import ZImageTransformer2DModel
            with temp_patch_module_attr("diffusers", "ZImageTransformer2DModel", ZImageTransformer2DModel):
                transformer = ZImageTransformer2DModel.from_single_file(
                    gguf_path,
                    config=os.path.join(cur_path, f"{repo}/transformer"),
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),)
            transformer.repo=repo

    elif dit_path :
        print("use single dit")
        if "flux" in dit_path.lower():
            if "klein" in dit_path.lower():
                from.flux2.transformer_flux2 import Flux2Transformer2DModel
                repo="4B" if "4b" in dit_path.lower() else "9B"
                with temp_patch_module_attr("diffusers", "Flux2Transformer2DModel", Flux2Transformer2DModel):
                    
                    try: 
                        transformer =Flux2Transformer2DModel.from_single_file(dit_path,config=os.path.join(cur_path, f"flux2_klein/{repo}/transformer"),torch_dtype=torch.bfloat16,)
                    except Exception as e:
                        print(e)
                        from accelerate import init_empty_weights
                        t_state_dict=load_file(dit_path,device="cpu") 
                        config_ = Flux2Transformer2DModel.load_config(os.path.join(cur_path,f"flux2_klein/{repo}/transformer/config.json") )
                        config_["dype"]=use_dype
                        config_["method"]=method
                        with init_empty_weights():
                            transformer = Flux2Transformer2DModel.from_config(config_,torch_dtype=torch.bfloat16)
                        transformer.load_state_dict(t_state_dict, strict=False,assign=True)
                        del t_state_dict
                        gc_cleanup()   
                transformer.repo=repo                 
            else:
                from.DyPE.flux.transformer_flux import FluxTransformer2DModel 
                with temp_patch_module_attr("diffusers", "FluxTransformer2DModel", FluxTransformer2DModel):
                    try:          
                        transformer =FluxTransformer2DModel.from_single_file(dit_path,config=os.path.join(cur_path, "Flux/FLUX.1-Krea-dev/transformer"),torch_dtype=torch.bfloat16,)
                    except Exception as e:
                        print(e)
                        from accelerate import init_empty_weights
                        t_state_dict=load_file(dit_path,device="cpu") 
                        config_ = FluxTransformer2DModel.load_config(os.path.join(cur_path,"Flux/FLUX.1-Krea-dev/transformer/config.json") )
                        config_["dype"]=use_dype
                        config_["method"]=method
                        with init_empty_weights():
                            transformer = FluxTransformer2DModel.from_config(config_,torch_dtype=torch.bfloat16)
                        transformer.load_state_dict(t_state_dict, strict=False,assign=True)
                        del t_state_dict
                        gc_cleanup()
            
        elif "qwen" in dit_path.lower():
            from.qwen.transformer_qwenimage import QwenImageTransformer2DModel  
            with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
                try:              
                    transformer =QwenImageTransformer2DModel.from_single_file(dit_path,config=os.path.join(cur_path, "Qwen-Image/transformer"),torch_dtype=torch.bfloat16,)
                except Exception as e:
                    print(e)
                    from accelerate import init_empty_weights
                    
                    t_state_dict=load_file(dit_path,device="cpu") 
                    config_ = QwenImageTransformer2DModel.load_config(os.path.join(cur_path, "Qwen-Image/transformer/config.json"),)
                    config_["dype"]=use_dype
                    config_["method"]=method
                    with init_empty_weights():
                        transformer = QwenImageTransformer2DModel.from_config(config_,torch_dtype=torch.bfloat16)
                    transformer.load_state_dict(t_state_dict, strict=False,assign=True)
                    del t_state_dict
                    gc_cleanup()
        else:
            if "turbo" in dit_path.lower():
                repo="Z-Image-turbo"
                from .zimage.transformer_z_image import ZImageTransformer2DModel
            else:
                repo="Z-Image"
                from .zimage.transformer_z_image_ import ZImageTransformer2DModel
            with temp_patch_module_attr("diffusers", "ZImageTransformer2DModel", ZImageTransformer2DModel):
                try:
                    transformer = ZImageTransformer2DModel.from_single_file(dit_path,config=os.path.join(cur_path, f"{repo}/transformer"),torch_dtype=torch.bfloat16,)
                except Exception as e:
                    print(e)
                    from accelerate import init_empty_weights
                    t_state_dict=load_file(dit_path,device="cpu") 
                    config_ = ZImageTransformer2DModel.load_config(os.path.join(cur_path, f"{repo}/transformer/config.json"),)
                    config_["dype"]=use_dype
                    config_["method"]=method
                    with init_empty_weights():
                        transformer = ZImageTransformer2DModel.from_config(config_,torch_dtype=torch.bfloat16)
                    transformer.load_state_dict(t_state_dict, strict=False,assign=True)
                    del t_state_dict
                    gc_cleanup()
                transformer.repo=repo
    else:
        raise "you must choice a unet or gguf "

    return transformer

def load_conditioning_model(model,ip_adpter_path,lora1,lora2,lora_scales=[1.0,1.0]):

    lora1_path=folder_paths.get_full_path("loras", lora1) if lora1!="none" else None
    lora2_path=folder_paths.get_full_path("loras", lora2) if lora2!="none" else None
    lora_list=[i for i in [lora1_path,lora2_path] if i is not None]
    if isinstance(model,FluxTransformer2DModel):
        vae=OmegaConf.load(os.path.join(cur_path,"Flux/FLUX.1-Krea-dev/vae/config.json")  )
    elif isinstance(model,QwenImageTransformer2DModel):
        vae=OmegaConf.load(os.path.join(cur_path,"Qwen-Image/vae/config.json"))
    elif isinstance(model,ZImageTransformer2DModel):
        vae=OmegaConf.load(os.path.join(cur_path,"Z-Image-turbo/vae/config.json"))
    elif isinstance(model,Flux2Transformer2DModel):
        if hasattr(model,"repo"):
            repo= model.repo
        else:
            repo="4B"
        vae=OmegaConf.load(os.path.join(cur_path,f"flux2_klein/{repo}/vae/config.json"))
    else:
        vae=OmegaConf.load(os.path.join(cur_path,"Z-Image/vae/config.json"))
    if isinstance(model,FluxTransformer2DModel):
        from.DyPE.flux.pipeline_flux import FluxPipeline
        pipeline = FluxPipeline.from_pretrained(os.path.join(cur_path,"Flux/FLUX.1-Krea-dev"),VAE=vae,vae=None,transformer=model,text_encoder=None,text_encoder_2=None, torch_dtype=torch.bfloat16,dype=True)
    elif isinstance(model,QwenImageTransformer2DModel):
        from.qwen.pipeline_qwenimage import QwenImagePipeline
        pipeline = QwenImagePipeline.from_pretrained(os.path.join(cur_path,"Qwen-Image"),VAE=vae,transformer=model,torch_dtype=torch.bfloat16,dype=True) 
    elif isinstance(model,ZImageTransformer2DModel):
        from.zimage.pipeline_z_image import ZImagePipeline
        if hasattr(model,"repo"):
            repo= model.repo
        else:
            repo="Z-Image-turbo"
        pipeline = ZImagePipeline.from_pretrained(os.path.join(cur_path,repo),VAE=vae,transformer=model,torch_dtype=torch.bfloat16,dype=True)
    elif isinstance(model,Flux2Transformer2DModel):
        from .flux2.pipeline_flux2_klein import Flux2KleinPipeline
        if hasattr(model,"repo"):
            repo= model.repo
        else:
            repo="4B"
        
        pipeline = Flux2KleinPipeline.from_pretrained(os.path.join(cur_path,f"flux2_klein/{repo}"),VAE=vae,transformer=model,torch_dtype=torch.bfloat16,dype=True)
    else:
        from.zimage.pipeline_z_image_ import ZImagePipeline
        pipeline = ZImagePipeline.from_pretrained(os.path.join(cur_path,"Z-Image"),VAE=vae,transformer=model,torch_dtype=torch.bfloat16,dype=True)
    lora_list=lora_list if lora_list else None
    if ip_adpter_path is not None and isinstance(pipeline,FluxPipeline):
        from safetensors import safe_open
        if os.path.basename(ip_adpter_path).endswith(".safetensors"):
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ip_adpter_path, framework="pt", device="cpu") as f:
                image_proj_keys = ["ip_adapter_proj_model.", "image_proj."]
                ip_adapter_keys = ["double_blocks.", "ip_adapter."]
                for key in f.keys():
                    if any(key.startswith(prefix) for prefix in image_proj_keys):
                        diffusers_name = ".".join(key.split(".")[1:])
                        state_dict["image_proj"][diffusers_name] = f.get_tensor(key)
                    elif any(key.startswith(prefix) for prefix in ip_adapter_keys):
                        diffusers_name = (
                            ".".join(key.split(".")[1:])
                            .replace("ip_adapter_double_stream_k_proj", "to_k_ip")
                            .replace("ip_adapter_double_stream_v_proj", "to_v_ip")
                            .replace("processor.", "")
                        )
                        state_dict["ip_adapter"][diffusers_name] = f.get_tensor(key)
        else:
            from diffusers.models.modeling_utils import load_state_dict
            state_dict = load_state_dict(ip_adpter_path)
        pipeline.load_ip_adapter(state_dict,os.path.basename(ip_adpter_path))
        pipeline.set_ip_adapter_scale(1.0)
     
    if lora_list is None:
        return pipeline
    try:    
        if len(lora_list)!=len(lora_scales): #sacles  
            lora_scales = lora_scales[:1]
        all_adapters = pipeline.get_list_adapters()
        dit_list=[]
        if all_adapters:
            dit_list= all_adapters['transformer']
        adapter_name_list=[]
        for path in lora_list:
            if path is not None:
                name=os.path.basename(path).split('.')[0]
                adapter_name_list.append(name)
                if name in dit_list:
                    continue
                pipeline.load_lora_weights(path, adapter_name=name)
        print(f"成功加载LoRA权重: {adapter_name_list} (scale: {lora_scales})")        
        pipeline.set_adapters(adapter_name_list, adapter_weights=lora_scales)
        try:
            active_adapters = pipeline.get_active_adapters()
            all_adapters = pipeline.get_list_adapters()
            print(f"当前激活的适配器: {active_adapters}")
            print(f"所有可用适配器: {all_adapters}") 
        except:
            pass
        return pipeline
        
    except Exception as e:
        print(f"Failed to apply LoRA {str(e)}")
        return pipeline


def apply_base_model(diffusion_models,gguf,):

    gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
    unet_path=folder_paths.get_full_path("diffusion_models", diffusion_models) if diffusion_models != "none" else None

    transformer=load_flux_tansformer(gguf_path, unet_path) 
    return transformer


def infer_dype(pipeline, ip_adapter_image_embeds, prompt_embeds,pooled_prompt_embeds,negative_prompt_embeds,negative_pooled_prompt_embeds,seed, 
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
        "ip_adapter_image_embeds":ip_adapter_image_embeds,
        "height": height,
        "width": width,  
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images

    # max_gpu_memory = torch.cuda.max_memory_allocated()
    # print(f"Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")
    return output_image
