#!/usr/bin/env python3
"""
Step 1: Regenerate Worst-Rated Prompts Through Their Respective Models
Uses English-first thinking strategy
"""

import json
import pandas as pd
import numpy as np
import csv
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("outputs_local/regenerated")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_JSONL = OUTPUT_DIR / "regenerated_worst_responses.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "regenerated_worst_responses.csv"

GEMINI_KEY = ""  # Set via environment variable
OPENAI_KEY = ""  # Set via environment variable

# Model mapping: model_name -> (model_type, identifier, api_key)
MODEL_REGISTRY = {
    'gemini_flash': ('gemini', 'gemini-2.5-flash', GEMINI_KEY),
    'gemini_pro': ('gemini', 'gemini-2.5-pro', GEMINI_KEY),
    'qwen25_7b': ('local', 'Qwen/Qwen2.5-7B-Instruct', None),
    'nous_hermes_mistral_7b': ('local', 'NousResearch/Hermes-3-Llama-3.1-8B', None),
    'openhermes_mistral_7b': ('local', 'teknium/OpenHermes-2.5-Mistral-7B', None),
    'llama31_8b': ('skip', 'meta-llama/Llama-3.1-8B-Instruct', None),  # gated; skip
    'zephyr_7b': ('local', 'HuggingFaceH4/zephyr-7b-beta', None),
    'yi15_6b': ('local', '01-ai/Yi-1.5-6B-Chat', None),
    'phi35_mini': ('local', 'microsoft/phi-3.5-mini-instruct', None),
    'qwen2_7b': ('local', 'Qwen/Qwen2-7B-Instruct', None),
    'mistral_7b': ('local', 'mistralai/Mistral-7B-Instruct-v0.1', None),
}


def parse_model_name(model_field: str) -> str:
    """Extract short model key from stored path/name"""
    if not model_field:
        return "unknown"
    name = model_field
    if name.startswith("outputs_local/PromptPersona_Full_600_"):
        name = name.replace("outputs_local/PromptPersona_Full_600_", "")
    name = name.replace(".jsonl", "").replace(".csv", "")
    return name


def load_outputs_lookup(model_key: str) -> Dict[str, Dict[str, str]]:
    """Build id -> {prompt, response} map from original model outputs.

    Falls back across jsonl/csv files in outputs_local/ and outputs/ so we can
    recover prompts for models whose generations were stored in CSV (e.g.
    gemini_flash) instead of JSONL.
    """

    lookup: Dict[str, Dict[str, str]] = {}

    # Priority order: local jsonl -> global jsonl -> local csv -> global csv
    candidate_paths = [
        Path(f"outputs_local/PromptPersona_Full_600_{model_key}.jsonl"),
        Path(f"outputs/PromptPersona_Full_600_{model_key}.jsonl"),
        Path(f"outputs_local/PromptPersona_Full_600_{model_key}.csv"),
        Path(f"outputs/PromptPersona_Full_600_{model_key}.csv"),
    ]

    path = next((p for p in candidate_paths if p.exists() and p.stat().st_size > 0), None)
    if not path:
        return lookup

    if path.suffix == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    lookup[str(data.get("id", ""))] = {
                        "prompt": data.get("prompt_text", ""),
                        "response": data.get("response", ""),
                    }
                except Exception:
                    continue
    else:  # CSV
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                lookup[str(row.get("id", ""))] = {
                    "prompt": row.get("prompt_text", ""),
                    "response": row.get("response", row.get("model_response", "")),
                }
        except Exception:
            pass

    return lookup

# System prompt with English-first thinking
SYSTEM_PROMPT = (
    "You are a helpful expert assistant. "
    "Before responding: First think through the solution clearly in ENGLISH with detailed steps. "
    "Then provide your response in the same language as the user's query. "
    "Be comprehensive, practical, and safety-conscious."
)


def load_worst_rated_prompts(n_per_model: int = 5):
    """Load worst-rated prompts from judged outputs - select per model and join prompt text."""
    print("\n" + "="*80)
    print("STEP 1: LOADING WORST-RATED PROMPTS FROM ALL MODELS")
    print("="*80)

    judged_file = Path("outputs_local/judged_outputs_llamaguard.jsonl")
    if not judged_file.exists():
        print(f"❌ File not found: {judged_file}")
        return []

    rows = []
    with open(judged_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                score = data.get("score", {})
                rows.append({
                    "id": str(data.get("id", "")),
                    "model_full": data.get("model", ""),
                    "model": parse_model_name(data.get("model", "")),
                    "assertiveness": float(score.get("assertiveness", 5)),
                    "complexity": float(score.get("complexity", 5)),
                    "emotional_distance": float(score.get("emotional_distance", 5)),
                    "instructional_density": float(score.get("instructional_density", 5)),
                })
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ No responses loaded")
        return []

    df["quality_score"] = (df["complexity"] + df["instructional_density"]) / 2

    # Group per model and pick worst n
    selected = []
    models = sorted(df["model"].unique())
    print(f"\n✅ Loaded {len(df)} total responses")
    print(f"📊 Models found: {len(models)}")
    for m in models:
        model_df = df[df["model"] == m]
        print(f"   - {m}: {len(model_df)} responses")

    print("\n🎯 Selecting worst-rated prompts per model:")
    for m in models:
        model_df = df[df["model"] == m].nsmallest(n_per_model, "quality_score")
        print(f"   {m}: {len(model_df)} worst responses (quality avg: {model_df['quality_score'].mean():.2f})")
        selected.append(model_df)

    worst = pd.concat(selected, axis=0)
    print(f"\n✅ Total selected for regeneration: {len(worst)} prompts")

    # Attach original prompt/response
    results = []
    for model_name in worst["model"].unique():
        lookup = load_outputs_lookup(model_name)
        model_rows = worst[worst["model"] == model_name]
        for _, row in model_rows.iterrows():
            info = lookup.get(row["id"], {"prompt": "", "response": ""})
            results.append({
                "id": row["id"],
                "model": model_name,
                "prompt": info.get("prompt", ""),
                "original_response": info.get("response", ""),
                "assertiveness": row["assertiveness"],
                "complexity": row["complexity"],
                "emotional_distance": row["emotional_distance"],
                "instructional_density": row["instructional_density"],
                "quality_score": row["quality_score"],
            })

    # Drop any missing prompts
    filtered = [r for r in results if r["prompt"]]
    missing = len(results) - len(filtered)
    if missing:
        print(f"⚠️  Skipped {missing} entries with missing prompt text")

    return filtered

def call_gemini(prompt: str, model_id: str, api_key: str) -> Optional[str]:
    """Call Gemini with system prompt"""
    try:
        genai.configure(api_key=api_key)
        
        client = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=SYSTEM_PROMPT,
            safety_settings=[
                {
                    "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            ]
        )
        response = client.generate_content(prompt, stream=False)
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"      Gemini error: {str(e)[:50]}")
        return None

def call_openai(prompt: str, model_id: str, api_key: str) -> Optional[str]:
    """Call OpenAI with system prompt"""
    try:
        if not OpenAI:
            return None
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        print(f"      OpenAI error: {str(e)[:50]}")
        return None

LOCAL_MODEL_CACHE: Dict[str, tuple] = {}


def call_local_model(prompt: str, model_id: str) -> Optional[str]:
    """Call local Hugging Face model (cached per model_id)."""
    try:
        if model_id not in LOCAL_MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            LOCAL_MODEL_CACHE[model_id] = (tokenizer, model)
        else:
            tokenizer, model = LOCAL_MODEL_CACHE[model_id]
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part (after assistant tag if present)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        
        return response.strip()[:500] if response else None
    except Exception as e:
        print(f"      Local model error: {str(e)[:50]}")
        return None

def generate_response(prompt: str, model_name: str) -> Optional[str]:
    """Generate response using appropriate model"""
    
    if model_name not in MODEL_REGISTRY:
        print(f"      Unknown model: {model_name}")
        return None
    
    model_type, identifier, api_key = MODEL_REGISTRY[model_name]

    if model_type == 'skip':
        print(f"      Skipping gated model: {model_name}")
        return None
    
    if model_type == 'gemini':
        return call_gemini(prompt, identifier, api_key)
    elif model_type == 'openai':
        return call_openai(prompt, identifier, api_key)
    elif model_type == 'local':
        return call_local_model(prompt, identifier)
    
    return None

def regenerate_worst_responses(worst_prompts: List[Dict]):
    """Regenerate worst-rated prompts through their original models"""
    print("\n" + "="*80)
    print("STEP 2: REGENERATING WITH ENGLISH-FIRST THINKING")
    print("="*80)
    
    results = []
    # Clear output files at start to avoid stale data
    try:
        if OUTPUT_JSONL.exists():
            OUTPUT_JSONL.unlink()
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
    except Exception:
        pass
    
    # Group by model
    model_groups = {}
    for item in worst_prompts:
        model = item['model']
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(item)
    
    print(f"\n📊 Models to regenerate:")
    for model, items in sorted(model_groups.items()):
        print(f"   {model}: {len(items)} prompts")
    
    print(f"\n⚠️  Total to regenerate: {len(worst_prompts)} prompts")
    print(f"   Estimated time: ~{len(worst_prompts) * 0.8:.0f}-{len(worst_prompts) * 1.5:.0f} seconds (depending on model)")
    
    # Regenerate for each model
    total_idx = 0
    for model_name, items in sorted(model_groups.items()):
        print(f"\n🔄 Regenerating with {model_name}...")
        model_type, identifier, _ = MODEL_REGISTRY.get(model_name, ("unknown", None, None))
        
        for idx, item in enumerate(items, 1):
            total_idx += 1
            prompt = item['prompt']

            if not prompt:
                print(f"  [{total_idx}/{len(worst_prompts)}] {model_name} ❌ (missing prompt)")
                continue
            
            print(f"  [{total_idx}/{len(worst_prompts)}] {model_name}", end=" ", flush=True)
            
            start = time.time()
            response = generate_response(prompt, model_name)
            elapsed = time.time() - start
            
            if response:
                print(f"✅ ({elapsed:.1f}s)")
                record = {
                    'id': item['id'],
                    'prompt': prompt,
                    'model': model_name,
                    'original_response': item['original_response'],
                    'original_quality': (item['complexity'] + item['instructional_density']) / 2,
                    'regenerated_response': response,
                    'regeneration_time': elapsed,
                }
                results.append(record)

                # Incremental save to JSONL to avoid loss on interruption
                try:
                    with open(OUTPUT_JSONL, 'a') as f:
                        json.dump(record, f)
                        f.write('\n')
                except Exception:
                    pass
            else:
                print(f"❌ ({elapsed:.1f}s - Failed)")

        # Free local model weights between model groups to avoid OOM
        if model_type == 'local' and identifier in LOCAL_MODEL_CACHE:
            try:
                tok, mdl = LOCAL_MODEL_CACHE.pop(identifier)
                del tok, mdl
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    return results

def save_regenerated_responses(results: List[Dict]):
    """Save regenerated responses to JSONL for judging"""
    print("\n" + "="*80)
    print("STEP 3: SAVING REGENERATED RESPONSES")
    print("="*80)
    
    if len(results) == 0:
        print("⚠️  No results to save")
        return None
    
    # Write JSONL (overwrite with final aggregated results)
    with open(OUTPUT_JSONL, 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')

    # Write CSV
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Saved {len(results)} regenerated responses to:")
    print(f"   JSONL: {OUTPUT_JSONL}")
    print(f"   CSV:   {OUTPUT_CSV}")
    print(f"\n📊 Regeneration stats:")
    print(f"   Total regenerated: {len(results)}")
    print(f"   Total time: {sum(r['regeneration_time'] for r in results):.1f}s")
    
    return OUTPUT_JSONL

def main():
    print("\n" + "="*80)
    print("REGENERATE WORST-RATED PROMPTS WITH ENGLISH-FIRST THINKING")
    print("="*80)
    
    # Step 1: Load worst-rated prompts
    worst_prompts = load_worst_rated_prompts(n_per_model=5)
    
    if len(worst_prompts) == 0:
        print("❌ No worst prompts found")
        return
    
    # Step 2: Regenerate through respective models
    results = regenerate_worst_responses(worst_prompts)
    
    # Step 3: Save for judging
    output_file = save_regenerated_responses(results)
    
    print("\n" + "="*80)
    print("✅ STEP 1 COMPLETE: REGENERATION DONE")
    print("="*80)
    print(f"\n📁 Output: {output_file}")
    print("\nNext: Run judge_regenerated_responses.py to evaluate with llama-guard-3")

if __name__ == "__main__":
    main()
