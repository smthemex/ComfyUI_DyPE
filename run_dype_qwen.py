"""
Official implementation for ultra-high resolution image generation with Qwen
DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion
"""

import os
import argparse
import torch
from diffusers import DiffusionPipeline
from .qwen.transformer_qwenimage import QwenImageTransformer2DModel

def main():
    parser = argparse.ArgumentParser(
        description='DyPE: Generate ultra-high resolution images with Qwen'
    )

    # 1. Specific Prompt Default
    parser.add_argument(
        '--prompt',
        type=str,
        default="A woman kneels on the forest floor, smiling as she offers grapes to a large brown bear beside her, surrounded by tall birch trees.",
        help='Text prompt for image generation'
    )

    # 2. Resolution Default (3072x3072)
    parser.add_argument('--height', type=int, default=3072, help='Image height in pixels')
    parser.add_argument('--width', type=int, default=3072, help='Image width in pixels')

    # 3. Steps Default (40)
    parser.add_argument('--steps', type=int, default=40, help='Number of inference steps')

    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    # 4. Method logic (dype vs base)
    parser.add_argument(
        '--method',
        type=str,
        choices=['dype', 'base'],
        default='dype',
        help='Position encoding method (dype or base)'
    )

    args = parser.parse_args()

    # Output setup
    os.makedirs("outputs", exist_ok=True)
    model_name = "Qwen/Qwen-Image"

    # Device setup
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"Loading {model_name} on {device}...")

    # Logic: if method is dype, use_dype is True. If base, False.
    use_dype = (args.method == 'dype')

    # Load local transformer with Qwen-specific class
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        dype=use_dype,
    )

    # Initialize pipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)

    # Qwen "Magic" suffix
    positive_magic = ", Ultra HD, 4K, cinematic composition."
    full_prompt = args.prompt + positive_magic
    negative_prompt = " "  # Default

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating {args.height}x{args.width} image with {args.steps} steps using Qwen (Method: {args.method})...")

    # Generate image using Qwen specific parameters
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        true_cfg_scale=4.0,
        generator=generator
    ).images[0]

    # Save image
    filename = f"outputs/seed_{args.seed}_method_{args.method}_res_{args.height}x{args.width}.png"
    image.save(filename)
    print(f"✓ Image saved to: {filename}")


if __name__ == "__main__":
    main()