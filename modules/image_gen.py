"""
image_gen.py
============
Diffusion-based image generation module for MedVisor.

Uses Stable Diffusion (via HuggingFace diffusers) to generate educational
medical illustrations from text prompts produced by the LLM/RAG module.

By default it uses **SDXL-Turbo**, which generates images in just 1-4
inference steps — fast enough to run interactively on a Colab T4 GPU.
A fallback to SD 1.5 is provided for lower-memory environments.
"""

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from PIL import Image


class MedicalImageGenerator:
    """Generates educational medical illustrations using Stable Diffusion."""

    def __init__(self, model_type: str = "sdxl-turbo", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[ImageGen] Loading {model_type} on {self.device}...")

        if model_type == "sdxl-turbo":
            # SDXL-Turbo: 1-4 step generation, very fast
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None,
            )
            self.num_steps = 4
            self.guidance_scale = 0.0  # Turbo uses no classifier-free guidance
        else:
            # Fallback: SD 1.5
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                torch_dtype=self.dtype,
                safety_checker=None,
            )
            self.num_steps = 25
            self.guidance_scale = 7.5

        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

        print(f"[ImageGen] Ready ({model_type}, {self.num_steps} steps).")

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, text, watermark, gore, graphic",
        seed: int = 42,
    ) -> Image.Image:
        """Generate a single illustration from a text prompt."""
        # Enhance the prompt with a consistent educational style
        full_prompt = (
            f"{prompt}, clean medical illustration, educational diagram, "
            "soft colors, professional, high quality, detailed"
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = dict(
            prompt=full_prompt,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        # SD 1.5 supports negative prompts; Turbo with guidance=0 ignores them
        if self.model_type != "sdxl-turbo":
            kwargs["negative_prompt"] = negative_prompt

        image = self.pipe(**kwargs).images[0]
        return image


if __name__ == "__main__":
    gen = MedicalImageGenerator(model_type="sdxl-turbo")
    img = gen.generate(
        "clean medical textbook illustration of the human brain showing a blocked artery"
    )
    img.save("test_output.png")
    print("Saved test_output.png")
