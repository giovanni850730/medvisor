"""
image_gen.py
============
Diffusion-based image generation module for MedVisor.
Supports SDXL (high quality), SDXL-Lightning (fast), and SDXL-Turbo (fastest).
"""

import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image


class MedicalImageGenerator:
    """Generates educational medical illustrations using Stable Diffusion."""

    def __init__(self, model_type: str = "sdxl", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[ImageGen] Loading {model_type} on {self.device}...")

        if model_type == "sdxl":
            # Full SDXL base — best quality
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True,
            )
            self.num_steps = 30
            self.guidance_scale = 7.5
            self.resolution = 1024

        elif model_type == "sdxl-lightning":
            # SDXL-Lightning — 8-step distilled, good quality + fast
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            ckpt = "sdxl_lightning_8step_unet.safetensors"
            unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
                self.device, self.dtype
            )
            unet.load_state_dict(
                load_file(hf_hub_download("ByteDance/SDXL-Lightning", ckpt), device=self.device)
            )
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, torch_dtype=self.dtype, variant="fp16"
            )
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )
            self.num_steps = 8
            self.guidance_scale = 0.0
            self.resolution = 1024

        else:  # sdxl-turbo (fastest, lowest quality)
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None,
            )
            self.num_steps = 4
            self.guidance_scale = 0.0
            self.resolution = 512

        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":
            self.pipe.enable_vae_tiling()
            self.pipe.enable_attention_slicing()

        print(f"[ImageGen] Ready ({model_type}, {self.num_steps} steps, {self.resolution}px).")

    # ------------------------------------------------------------------
    def generate(self, prompt: str, negative_prompt: str = None, seed: int = 42) -> Image.Image:
        """Generate a single high-quality illustration from a text prompt."""
        # Richer, more specific style prompt for cleaner medical illustrations
        full_prompt = (
            f"{prompt}, professional medical illustration, anatomical diagram, "
            "clean vector art style, clear labels, bright even lighting, "
            "high detail, sharp focus, textbook quality, white background, 4k"
        )

        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed anatomy, ugly, "
                "messy, cluttered, dark, grainy, noisy, watermark, signature, "
                "text artifacts, photorealistic gore, disturbing, jpeg artifacts"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = dict(
            prompt=full_prompt,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            height=self.resolution,
            width=self.resolution,
        )
        # Negative prompts only work when guidance_scale > 0
        if self.guidance_scale > 0:
            kwargs["negative_prompt"] = negative_prompt

        image = self.pipe(**kwargs).images[0]
        return image


if __name__ == "__main__":
    gen = MedicalImageGenerator(model_type="sdxl")
    img = gen.generate(
        "clean medical illustration of the human brain showing a blocked middle cerebral artery"
    )
    img.save("test_output.png")
    print("Saved test_output.png")