#!/usr/bin/env python3
"""
Generate persona responses with English-first strategy for improved Hinglish quality.
Uses updated prompt that generates English thinking first, then translates to Hinglish.
"""

import argparse
import csv
import sys
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------- SETTINGS ------------------- #

DEFAULT_INPUT = "PromptPersona_Full_600.csv"
DEFAULT_OUTPUT_DIR = "outputs_local"
DEFAULT_API_KEY = ""
DEFAULT_OPENAI_API_KEY = ""

GEMINI_HARDCODED_KEY = ""  # Set via environment variable or CLI flag
OPENAI_HARDCODED_KEY = ""  # Set via environment variable or CLI flag

# UPDATED SYSTEM PROMPT: English-first strategy with Hinglish translation
GEMINI_SYSTEM_PROMPT = (
    "You are an expert assistant. Mirror the persona and language cues provided. "
    "For Hinglish queries: First think through the solution clearly in English with detailed steps. "
    "Then translate to natural Hinglish (mixing Hindi and English) while preserving all technical details. "
    "Offer practical, comprehensive, safety-first guidance."
)

OPENAI_SYSTEM_PROMPT = (
    "You are an expert assistant. Mirror the persona and language cues provided. "
    "For Hinglish queries: First think through the solution clearly in English with detailed steps. "
    "Then translate to natural Hinglish while preserving all technical details. "
    "Offer practical, comprehensive, safety-first guidance."
)

LOCAL_MODEL_SYSTEM_PROMPT = (
    "You are an expert technical assistant. "
    "When responding to Hinglish: Think in English first with numbered steps, then translate to Hinglish. "
    "Be clear, practical, and comprehensive in your response."
)

@dataclass
class ModelConfig:
    key: str
    kind: str  # "gemini" | "gemma" | "openai" | "ollama"
    identifier: str

MODEL_SPECS: Dict[str, ModelConfig] = {
    "gemini_pro": ModelConfig("gemini_pro", "gemini", "gemini-2.5-pro"),
    "gemini_flash": ModelConfig("gemini_flash", "gemini", "gemini-2.5-flash"),
}

SAFETY_SETTINGS = [
    {
        "category": cat,
        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
    }
    for cat in (
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    )
]

# ------------------- IO HELPERS ------------------- #

def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f"Warning: No rows to save to {path}")
        return
    
    fieldnames = rows[0].keys()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ------------------- MODEL FUNCTIONS ------------------- #

def call_gemini(prompt: str, system_prompt: str, model_id: str) -> Optional[str]:
    """Call Gemini API with system prompt."""
    try:
        client = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_prompt,
            safety_settings=SAFETY_SETTINGS
        )
        response = client.generate_content(prompt, stream=False)
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"  Error calling Gemini: {e}", file=sys.stderr)
        return None

def call_openai(prompt: str, system_prompt: str, model_id: str) -> Optional[str]:
    """Call OpenAI API with system prompt."""
    if not OpenAI:
        print("OpenAI library not installed", file=sys.stderr)
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_HARDCODED_KEY)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        print(f"  Error calling OpenAI: {e}", file=sys.stderr)
        return None

def generate_with_model(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: str
) -> Optional[str]:
    """Generate response with specified model."""
    
    if model_config.kind == "gemini":
        genai.configure(api_key=GEMINI_HARDCODED_KEY)
        return call_gemini(prompt, system_prompt, model_config.identifier)
    
    elif model_config.kind == "openai":
        return call_openai(prompt, system_prompt, model_config.identifier)
    
    else:
        print(f"Unknown model kind: {model_config.kind}", file=sys.stderr)
        return None

# ------------------- MAIN ------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate persona responses with models")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()), help="Models to use")
    args = parser.parse_args()
    
    # Load input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    prompts = load_csv(args.input)
    print(f"Loaded {len(prompts)} prompts from {args.input}")
    
    # Generate for each model
    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            print(f"Warning: Unknown model {model_key}, skipping")
            continue
        
        config = MODEL_SPECS[model_key]
        output_path = args.output_dir / f"{args.input.stem}_{model_key}.csv"
        
        print(f"\nGenerating with {model_key}...")
        print(f"  Model: {config.identifier}")
        print(f"  Output: {output_path}")
        
        rows = []
        success_count = 0
        fail_count = 0
        
        for idx, prompt_row in enumerate(prompts, 1):
            prompt_text = prompt_row.get("prompt", "")
            
            if not prompt_text.strip():
                print(f"  [{idx}/{len(prompts)}] Skipping empty prompt", file=sys.stderr)
                fail_count += 1
                continue
            
            print(f"  [{idx}/{len(prompts)}]", end=" ", flush=True)
            
            start_time = time.time()
            response = generate_with_model(
                prompt_text,
                config,
                GEMINI_SYSTEM_PROMPT if config.kind == "gemini" else OPENAI_SYSTEM_PROMPT
            )
            elapsed = time.time() - start_time
            
            if response:
                rows.append({
                    "id": prompt_row.get("id", str(idx)),
                    "prompt": prompt_text,
                    "response": response,
                    "time_seconds": f"{elapsed:.2f}"
                })
                success_count += 1
                print(f"✅ ({elapsed:.1f}s)")
            else:
                fail_count += 1
                print(f"❌ Failed")
            
            # Rate limiting
            time.sleep(1)
        
        # Save results
        save_csv(output_path, rows)
        print(f"\n✅ Generated {success_count}/{len(prompts)} responses")
        print(f"   Failures: {fail_count}")
        print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    main()
