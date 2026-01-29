#!/usr/bin/env python3
"""
Sequentially run the CheX Bias prompts through a set of locally hosted models.
Requires: transformers>=4.42, accelerate, bitsandbytes (for 4-bit loads),
          plus any model-specific packages (e.g., llava for VLM).
"""

from pathlib import Path
import csv
import json
import time
from typing import Dict, List, Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    pipeline,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROMPTS_CSV = Path("PromptPersona_Full_600.csv")
OUTPUT_DIR = Path("outputs_local")
BATCH_SAVE = 10  # flush every N prompts

MODEL_REGISTRY: Dict[str, Dict] = {
    # 1. Mistral-7B Instruct
    "mistral_7b": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 2. Qwen 2.5 7B Instruct
    "qwen25_7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 3. Phi-3 Medium (4B)
    "phi3_medium": {
        "repo": "microsoft/Phi-3-medium-4k-instruct",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 4. Yi-1.5 6B Base
    "yi15_6b": {
        "repo": "01-ai/Yi-1.5-6B",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 5. Zephyr-7B: Instruction-tuned, fully open
    "zephyr_7b": {
        "repo": "HuggingFaceH4/zephyr-7b-beta",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 6. Qwen 2 7B Instruct: Fully open, Apache 2.0
    "qwen2_7b": {
        "repo": "Qwen/Qwen2-7B-Instruct",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 7. OpenHermes 2.5 Mistral: 7B, fully open
    "openhermes_mistral_7b": {
        "repo": "teknium/OpenHermes-2.5-Mistral-7B",
        "type": "causal-lm",
        "task": "text-generation",
    },
    # 8. Nous Hermes 2 Mistral DPO: 7B, fully open
    "nous_hermes_mistral_7b": {
        "repo": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "type": "causal-lm",
        "task": "text-generation",
    },
}

GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.95,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_prompts(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def build_text_pipeline(repo: str) -> Callable[[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=DTYPE,
    )

    def generate(prompt: str) -> str:
        outputs = generator(prompt, **GENERATION_KWARGS)
        completion = outputs[0]["generated_text"]
        return completion[len(prompt):].strip()

    return generate

def build_vision_language_pipeline(repo: str, image_root: Path) -> Callable[[Dict[str, str]], str]:
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()

    def generate(row: Dict[str, str]) -> str:
        image_path = image_root / row.get("image_filename", "")
        if not image_path.exists():
            return f"ERROR: missing image {image_path}"

        inputs = processor(
            images=image_path.open("rb"),
            text=row["prompt_text"],
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_KWARGS["max_new_tokens"],
            temperature=GENERATION_KWARGS["temperature"],
            do_sample=GENERATION_KWARGS["do_sample"],
            top_p=GENERATION_KWARGS["top_p"],
        )
        return processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return generate

def save_batch(writer, buffer: List[Dict[str, str]], handle) -> None:
    if not buffer:
        return
    writer.writerows(buffer)
    handle.flush()
    buffer.clear()

# --------------------------------------------------------------------------- #
# Main Loop
# --------------------------------------------------------------------------- #

def run_model(model_key: str, prompts: List[Dict[str, str]]) -> None:
    spec = MODEL_REGISTRY[model_key]
    out_path = OUTPUT_DIR / f"{PROMPTS_CSV.stem}_{model_key}.jsonl"
    print(f"[INFO] Running {model_key} ({spec['repo']}) -> {out_path}")

    if spec["type"] == "vision-language":
        generator = build_vision_language_pipeline(spec["repo"], spec["image_root"])
        adapter = lambda row: generator(row)
    else:
        generator = build_text_pipeline(spec["repo"])
        adapter = lambda row: generator(row["prompt_text"])

    with out_path.open("w", encoding="utf-8") as handle:
        buffer = []
        for idx, row in enumerate(prompts, start=1):
            start = time.time()
            try:
                response = adapter(row)
            except Exception as err:
                response = f"ERROR: {err}"
            duration = time.time() - start
            buffer.append({
                **row,
                "model_key": model_key,
                "model_repo": spec["repo"],
                "response": response,
                "latency_sec": round(duration, 3),
            })
            if len(buffer) >= BATCH_SAVE:
                for item in buffer:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                buffer.clear()
            if idx % 20 == 0:
                print(f"  processed {idx}/{len(prompts)} prompts for {model_key}")

        for item in buffer:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] {model_key} written to {out_path}")

def main() -> None:
    prompts = load_prompts(PROMPTS_CSV)
    ensure_output_dir(OUTPUT_DIR)

    # Skip models that are already completed
    completed = {
        "mistral_7b",
        "phi3_medium", 
        "qwen25_7b",
        "yi15_6b",
    }
    
    for key in MODEL_REGISTRY:
        if key in completed:
            print(f"[SKIP] {key} already completed")
            continue
        run_model(key, prompts)

if __name__ == "__main__":
    main()