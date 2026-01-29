import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

CALIBRATION_PATH = Path("analysis_output/safety/calibration_set.json")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM on the 50-shot calibration set with LoRA")
    parser.add_argument("--model-id", default="microsoft/Phi-3.5-mini-instruct", help="Base model to fine-tune")
    parser.add_argument("--calibration", default=str(CALIBRATION_PATH), help="Path to calibration JSON file")
    parser.add_argument("--output-dir", default="./sft_baseline_adapter", help="Trainer output directory")
    parser.add_argument("--final-dir", default="./sft_final", help="Directory to save final adapter")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit for memory savings")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load base model in 8-bit for memory savings")
    parser.add_argument("--device-map", default="auto", help="Device map passed to from_pretrained")
    return parser


def load_calibration(path: Path) -> Dataset:
    records = json.loads(path.read_text())
    data = [
        {
            "text": f"User: {row['prompt']}\nAssistant: {row['response']}"
        }
        for row in records
    ]
    return Dataset.from_list(data)


def main() -> None:
    args = build_argparser().parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset = load_calibration(Path(args.calibration))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model_kwargs = {"device_map": args.device_map}
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=384,
        args=TrainingArguments(
            output_dir=str(Path(args.output_dir)),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=10,
            fp16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
        ),
        peft_config=peft_config,
        model_init_kwargs=model_kwargs,
    )

    trainer.train()
    final_dir = Path(args.final_dir)
    trainer.save_model(str(final_dir))
    print(f"SFT complete for {args.model_id}. Adapter saved to {final_dir}.")


if __name__ == "__main__":
    main()
