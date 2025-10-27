"""
Official implementation for ultra-high resolution image generation as presented in:
DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion
"""

import torch
import argparse
import os
from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel
# from diffusers import FluxTransformer2DModel


def main():
    parser = argparse.ArgumentParser(
        description='DyPE: Generate ultra-high resolution images with FLUX'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="A mysterious woman stands confidently in elaborate, dark armor adorned with intricate designs, holding a staff, against a backdrop of smoke and an ominous red sky, with shadowy, gothic buildings in the distance.",
        help='Text prompt for image generation'
    )
    parser.add_argument('--height', type=int, default=4096, help='Image height in pixels')
    parser.add_argument('--width', type=int, default=4096, help='Image width in pixels')
    parser.add_argument('--steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument(
        '--method',
        type=str,
        choices=['yarn', 'ntk', 'base'],
        default='yarn',
        help='Position encoding method (yarn, ntk, or base)'
    )
    parser.add_argument(
        '--no_dype',
        action='store_true',
        help='Disable DyPE (dynamic position encoding)'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Set random seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Determine DyPE configuration
    use_dype = not args.no_dype

    # Load transformer with DyPE configuration
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        dype=use_dype,
        method=args.method,
    )

    # Initialize pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    # Generate image
    print(f"Generating {args.height}x{args.width} image with {args.steps} steps...")
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=4.5,
        generator=generator,
        num_inference_steps=args.steps,
    ).images[0]

    # Construct method name for filename
    method_name = args.method
    if use_dype:
        method_name = f"dy_{method_name}"

    # Save image with descriptive filename
    filename = f"outputs/seed_{args.seed}_method_{method_name}_res_{args.height}x{args.width}.png"
    image.save(filename)
    print(f"✓ Image saved to: {filename}")


if __name__ == "__main__":
    main()