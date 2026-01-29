"""Minimal DDPO-like stub so PPO script can still run without TRL's missing components."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

import torch
from diffusers import StableDiffusionPipeline

@dataclass
class DDPOConfig:
    num_epochs: int = 1
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    per_prompt_stat_tracking: bool = False
    mixed_precision: str = "fp16"
    lr: float = 1e-5
    project_name: str = "fallback"
    output_dir: str = "ppo_finetuned_model"
    log_with: str = "none"
    seed: int = 42


class DefaultDDPOStableDiffusionPipeline:
    """Wraps a diffusers StableDiffusion pipeline for inference."""

    def __init__(self, model_id: str, *, use_lora: bool = False, lora_rank: int = 16, torch_dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.pipeline.to(self.device)
        self.use_lora = use_lora
        if use_lora:
            # LoRA adapters are not supported in the stub.
            print("[ppo_ddpo_fallback] LoRA adapters are not supported in the fallback pipeline; running inference only.")

    def __call__(self, prompts: Sequence[str], *, guidance_scale: float = 7.5, num_inference_steps: int = 20):
        return self.pipeline(prompts, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

    def save_pretrained(self, output_dir: str) -> None:
        self.pipeline.save_pretrained(output_dir)


class DDPOTrainer:
    """Very lightweight trainer that just runs inference + reward logging."""

    def __init__(self, config: DDPOConfig, reward_fn, pipeline: DefaultDDPOStableDiffusionPipeline) -> None:
        self.config = config
        self.reward_fn = reward_fn
        self.pipeline = pipeline
        self.prompts = [
            "A medical X-ray that clearly highlights the lungs and heart",
            "A clean, well-lit chest CT slice showing normal anatomy",
            "A radiologist writing notes next to a monitor with scans",
        ]

    def train(self) -> None:
        batch_prompts = self.prompts[: max(1, self.config.train_batch_size)]
        for epoch in range(self.config.num_epochs):
            outputs = self.pipeline(
                batch_prompts,
                guidance_scale=7.5,
                num_inference_steps=20,
            )
            images = outputs.images
            rewards = self.reward_fn(images, batch_prompts)
            if isinstance(rewards, torch.Tensor):
                batch_reward = rewards.mean().item()
            else:
                batch_reward = statistics.mean(rewards)
            print(f"[ppo stub] Epoch {epoch + 1}/{self.config.num_epochs} reward avg: {batch_reward:.4f}")

    def save_pretrained(self, output_dir: str) -> None:
        print(f"[ppo stub] Skipping weight save because this is a fallback runner. Output would go to {output_dir}.")
