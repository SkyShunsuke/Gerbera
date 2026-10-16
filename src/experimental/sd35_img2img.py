"""Image-to-image translation of an MVTecAD image with Stable Diffusion 3.5.

Example:
    python src/experimental/sd35_img2img.py \
        --image data/mvtec_ad/hazelnut/test/good/000.png \
        --prompt "a hazelnut with a deep crack on its shell, macro photo" \
        --strength 0.6 --out vis/hazelnut_crack.png
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline

MODEL_ROOT = "/groups/gch51615/shared/image"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an MVTecAD image")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--model", default="stable-diffusion-3.5-large",
                        choices=["stable-diffusion-3.5-large", "stable-diffusion-3.5-large-turbo", "stable-diffusion-3.5-medium"])
    parser.add_argument("--strength", type=float, default=0.6, help="0: keep input, 1: ignore input")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="vis/sd35_img2img.png")
    args = parser.parse_args()

    if args.model.endswith("turbo"):  # distilled: few steps, no CFG
        args.steps, args.guidance = 4, 0.0

    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        f"{MODEL_ROOT}/{args.model}", torch_dtype=torch.bfloat16
    ).to("cuda")

    src = Image.open(args.image).convert("RGB").resize((args.res, args.res), Image.BICUBIC)
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        image=src,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]

    # Save input | output side by side
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (args.res * 2, args.res))
    canvas.paste(src, (0, 0))
    canvas.paste(out, (args.res, 0))
    canvas.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
