import argparse
from pathlib import Path
import torch

from ppo_utils import RewardWrapper

try:
    from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
    _fallback_warning = False
except ImportError as exc:
    from ppo_ddpo_fallback import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

    _fallback_warning = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDPO/PPO for Stable Diffusion against a reward model.")
    parser.add_argument("--reward-path", type=Path, default=Path("analysis_full/reward_best.pt"), help="Reward checkpoint to align against (save from train_reward_model.py)")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Diffusion policy to fine-tune")
    parser.add_argument("--output-dir", default="ppo_finetuned_model", help="Where to store the LoRA/weights produced by PPO")
    parser.add_argument("--log-with", default="tensorboard", choices=["tensorboard", "wandb", "none"], help="Backend for DDPO logging")
    parser.add_argument("--project-name", default="gemini-ppo-diffusion", help="Project name used by the logging backend")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Per-step batch size that fits on your GPU")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of PPO epochs to run")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation before each PPO update")
    parser.add_argument("--per-prompt-stat-tracking", action="store_true", help="Normalize rewards per prompt (recommended for stability)")
    parser.add_argument("--mixed-precision", default="fp16", choices=["fp16", "fp32"], help="Precision used for training")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for the DDPO trainer")
    parser.add_argument("--use-lora", action="store_true", help="Train a LoRA adapter instead of full weights (recommended)")
    parser.add_argument("--lora-rank", type=int, default=16, help="Rank for the LoRA adapters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if _fallback_warning:
        print("[run_ppo] TRL's DDPO helper classes are unavailable; running the fallback runner instead.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reward_fn = RewardWrapper(str(args.reward_path), device=device)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.model_id,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" and device == "cuda" else torch.float32,
    )

    config = DDPOConfig(
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_prompt_stat_tracking=args.per_prompt_stat_tracking,
        mixed_precision=args.mixed_precision,
        lr=args.learning_rate,
        project_name=args.project_name,
        output_dir=args.output_dir,
        log_with=args.log_with,
        seed=args.seed,
    )

    trainer = DDPOTrainer(config, reward_fn, pipeline)
    trainer.train()
    trainer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
