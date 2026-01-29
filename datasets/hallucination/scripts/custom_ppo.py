import argparse
import csv
import importlib
import os
import random
from dataclasses import dataclass
from typing import List

hf_hub = importlib.import_module("huggingface_hub")
if not hasattr(hf_hub, "cached_download"):
    hf_hub.cached_download = hf_hub.hf_hub_download

import requests
import torch
from torch import nn
from torch.optim import AdamW
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppo_utils import RewardWrapper

DEFAULT_PROMPT_CSV = "vlm_hallucination_prompts_only.csv"

PROMPT_LIBRARY = [
    "Medical X-ray of the thorax, centered on the lungs and heart",
    "Clean frontal chest CT showing normal anatomy",
    "Radiologist at work, reviewing chest scans on a monitor",
    "Detailed diagram of the lungs highlighting vascular patterns",
    "High-resolution cross-section of the thoracic cavity with emphasis on soft tissue",
]


def load_prompts_from_csv(csv_path: str) -> List[str]:
    if not os.path.isfile(csv_path):
        return PROMPT_LIBRARY[:]
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt") or row.get("text") or ""
            prompt = prompt.strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        return PROMPT_LIBRARY[:]
    return prompts


@dataclass
class CustomPPOConfig:
    model_id: str = "gpt2"
    pipeline_id: str = "runwayml/stable-diffusion-v1-5"
    reward_path: str = "analysis_full/reward_best.pt"
    device: str = "cuda"
    batch_size: int = 4
    num_epochs: int = 25
    ppo_epochs: int = 3
    clip_epsilon: float = 0.2
    learning_rate: float = 5e-5
    max_new_tokens: int = 32
    response_length_penalty: float = 0.0
    reward_baseline_beta: float = 0.95
    output_dir: str = "prompt_ppo"
    policy_backend: str = "gpt2"
    gemini_model: str = "text-bison-001"
    prompts_csv: str = DEFAULT_PROMPT_CSV


def parse_args() -> CustomPPOConfig:
    parser = argparse.ArgumentParser(description="Custom PPO loop that teaches an LLM to write high-reward prompts.")
    parser.add_argument("--model-id", default=CustomPPOConfig.model_id)
    parser.add_argument("--pipeline-id", default=CustomPPOConfig.pipeline_id)
    parser.add_argument("--reward-path", default=CustomPPOConfig.reward_path)
    parser.add_argument("--device", default=CustomPPOConfig.device)
    parser.add_argument("--batch-size", type=int, default=CustomPPOConfig.batch_size)
    parser.add_argument("--num-epochs", type=int, default=CustomPPOConfig.num_epochs)
    parser.add_argument("--ppo-epochs", type=int, default=CustomPPOConfig.ppo_epochs)
    parser.add_argument("--clip-epsilon", type=float, default=CustomPPOConfig.clip_epsilon)
    parser.add_argument("--lr", type=float, default=CustomPPOConfig.learning_rate)
    parser.add_argument("--max-new-tokens", type=int, default=CustomPPOConfig.max_new_tokens)
    parser.add_argument("--reward-baseline-beta", type=float, default=CustomPPOConfig.reward_baseline_beta)
    parser.add_argument("--output-dir", default=CustomPPOConfig.output_dir)
    parser.add_argument(
        "--policy-backend",
        choices=["gpt2", "gemini"],
        default=CustomPPOConfig.policy_backend,
        help="Policy used for sampling rollouts",
    )
    parser.add_argument("--gemini-model", default=CustomPPOConfig.gemini_model)
    parser.add_argument(
        "--prompts-csv",
        default=CustomPPOConfig.prompts_csv,
        help="Path to CSV containing prompts (expects a `prompt` column)",
    )
    args = parser.parse_args()
    return CustomPPOConfig(
        model_id=args.model_id,
        pipeline_id=args.pipeline_id,
        reward_path=args.reward_path,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        reward_baseline_beta=args.reward_baseline_beta,
        output_dir=args.output_dir,
        policy_backend=args.policy_backend,
        gemini_model=args.gemini_model,
        prompts_csv=args.prompts_csv,
    )


class PromptPPOTrainer:
    def __init__(self, config: CustomPPOConfig, prompts: List[str]):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.prompts = prompts
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.policy_model = AutoModelForCausalLM.from_pretrained(config.model_id).to(self.device)
        self.policy_model.train()
        self.optimizer = AdamW(self.policy_model.parameters(), lr=config.learning_rate)
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            config.pipeline_id, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.reward_wrapper = RewardWrapper(config.reward_path, device=self.device.type)
        self.gemini_model = config.gemini_model
        self.policy_backend = config.policy_backend

    def compute_generation_log_prob(self, prompt_ids: torch.Tensor, generated_output) -> torch.Tensor:
        prompt_length = prompt_ids.shape[-1]
        if not generated_output.scores:
            return torch.zeros(1, device=self.device)
        token_log_probs = []
        for offset, score in enumerate(generated_output.scores):
            token_id = generated_output.sequences[0, prompt_length + offset]
            step_logprob = torch.log_softmax(score, dim=-1)[0, token_id]
            token_log_probs.append(step_logprob)
        return torch.stack(token_log_probs).sum().unsqueeze(0)

    def _compute_response_log_prob(self, entry: dict) -> torch.Tensor:
        input_ids = entry["input_ids"].to(self.device)
        attention_mask = entry["attention_mask"].to(self.device)
        response_start = entry["response_start"]
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        start_index = max(0, response_start - 1)
        mask[:, start_index:] = True
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        response_log_probs = (token_log_probs * mask).sum(dim=-1)
        return response_log_probs

    def collect_rollouts(self) -> List[dict]:
        if self.policy_backend == "gemini":
            return self._collect_gemini_rollouts()
        return self._collect_gpt2_rollouts()

    def _collect_gpt2_rollouts(self) -> List[dict]:
        rollouts = []
        sample_prompts = random.sample(self.prompts, self.config.batch_size)
        for prompt in sample_prompts:
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.policy_model.generate(
                    **prompt_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            response_start = prompt_inputs.input_ids.shape[-1]
            response_ids = outputs.sequences[0, response_start:]
            if response_ids.numel() == 0:
                continue
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            if not response_text:
                response_text = prompt
            image = self.sd_pipeline(response_text, guidance_scale=7.5, num_inference_steps=20).images[0]
            reward = self.reward_wrapper([image], [response_text]).to(self.device)
            log_prob = self.compute_generation_log_prob(prompt_inputs.input_ids, outputs)
            cutoff = outputs.sequences.shape[-1]
            rollout = {
                "prompt": prompt,
                "response": response_text,
                "input_ids": outputs.sequences[:, :cutoff].detach().clone(),
                "attention_mask": torch.ones_like(outputs.sequences[:, :cutoff]),
                "response_start": response_start,
                "reward": reward.squeeze(0),
                "old_log_prob": log_prob.squeeze(0),
            }
            rollouts.append(rollout)
        return rollouts

    def _collect_gemini_rollouts(self) -> List[dict]:
        rollouts = []
        sample_prompts = random.sample(self.prompts, self.config.batch_size)
        for prompt in sample_prompts:
            response_text = self._call_gemini_api(prompt)
            if not response_text:
                continue
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            response_tokens = self.tokenizer(response_text, return_tensors="pt").to(self.device)
            response_start = prompt_inputs.input_ids.shape[-1]
            combined_input_ids = torch.cat(
                [prompt_inputs.input_ids, response_tokens.input_ids[:, 1:]], dim=-1
            )
            attention_mask = torch.ones_like(combined_input_ids)
            if response_tokens.input_ids.shape[-1] <= 1:
                continue
            image = self.sd_pipeline(response_text, guidance_scale=7.5, num_inference_steps=20).images[0]
            reward = self.reward_wrapper([image], [response_text]).to(self.device)
            entry = {
                "prompt": prompt,
                "response": response_text,
                "input_ids": combined_input_ids,
                "attention_mask": attention_mask,
                "response_start": response_start,
                "reward": reward.squeeze(0),
            }
            entry["old_log_prob"] = self._compute_response_log_prob(entry)
            rollouts.append(entry)
        return rollouts

    def _call_gemini_api(self, prompt: str) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set in environment when using the Gemini backend.")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateText"
            f"?key={api_key}"
        )
        payload = {
            "prompt": {"text": prompt},
            "maxOutputTokens": self.config.max_new_tokens,
            "temperature": 1.0,
            "candidateCount": 1,
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        candidates = result.get("candidates", [])
        if not candidates:
            return ""
        return candidates[0].get("output", "")

    def train(self) -> None:
        baseline = 0.0
        for epoch in range(self.config.num_epochs):
            rollouts = self.collect_rollouts()
            if not rollouts:
                continue
            rewards = torch.stack([entry["reward"] for entry in rollouts]).to(self.device)
            batch_mean = rewards.mean().item()
            baseline = baseline * self.config.reward_baseline_beta + batch_mean * (1 - self.config.reward_baseline_beta)
            advantages = rewards - baseline
            old_log_probs = torch.stack([entry["old_log_prob"] for entry in rollouts]).to(self.device)
            for _ in range(self.config.ppo_epochs):
                self.optimizer.zero_grad()
                new_log_probs = torch.stack([self._compute_response_log_prob(entry) for entry in rollouts])
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
                policy_loss.backward()
                self.optimizer.step()
            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} | reward {batch_mean:.4f} | baseline {baseline:.4f}"
            )
        self.policy_model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


def main() -> None:
    config = parse_args()
    prompts = load_prompts_from_csv(config.prompts_csv)
    trainer = PromptPPOTrainer(config, prompts)
    trainer.train()


if __name__ == "__main__":
    main()
