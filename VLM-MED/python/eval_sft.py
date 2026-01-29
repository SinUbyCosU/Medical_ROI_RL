import argparse
import json
from pathlib import Path
from typing import Iterator, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a base model + LoRA adapter over the 364-probe set")
    parser.add_argument("--base-model", default="microsoft/Phi-3.5-mini-instruct", help="Base model identifier")
    parser.add_argument("--adapter", default="./sft_final", help="Directory containing the LoRA adapter")
    parser.add_argument("--input", default="data/probes_364.json", help="Prompt JSON/JSONL path")
    parser.add_argument("--output", default="outputs_sft_baseline.jsonl", help="Where to write generations")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_prompts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation set at {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            data = [json.loads(line) for line in fh if line.strip()]
    prompts: List[str] = []
    for item in data:
        if isinstance(item, dict):
            prompts.append(item.get("prompt", ""))
        else:
            prompts.append(str(item))
    return prompts

def format_prompt(prompt: str) -> str:
    return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

def chunked(items: List[str], size: int) -> Iterator[List[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> None:
    args = build_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    prompts = load_prompts(input_path)
    print(f"Loaded {len(prompts)} prompts from {input_path}.")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, Path(args.adapter))
    model.eval()

    results = []
    device = next(model.parameters()).device
    for batch in tqdm(list(chunked(prompts, args.batch_size)), desc="Generating"):
        formatted_batch = [format_prompt(p) for p in batch]
        inputs = tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
            )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt, decoded in zip(batch, texts):
            if "<|assistant|>" in decoded:
                response = decoded.split("<|assistant|>")[-1].strip()
            else:
                response = decoded.strip()
            results.append({
                "prompt": prompt,
                "response": response,
                "model": "SFT_Baseline_N50",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} generations to {output_path}.")


if __name__ == "__main__":
    main()
