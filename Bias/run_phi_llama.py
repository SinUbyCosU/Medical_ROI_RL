#!/usr/bin/env python3
"""
Generate responses from Phi-3.5-mini and Llama-3.1-8B on 600 prompts
Designed to be GPU-safe with memory management
"""

import json
import csv
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Model configuration
MODELS = {
    "phi35_mini": {
        "key": "phi35_mini",
        "name": "Phi-3.5-Mini",
        "repo": "microsoft/Phi-3.5-mini-instruct",
        "ollama_tag": "phi3.5",
        "output_file": "outputs_local/PromptPersona_Full_600_phi35_mini.jsonl",
        "vram_mb": 3800,
    },
    "llama31_8b": {
        "key": "llama31_8b",
        "name": "Llama-3.1-8B",
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "ollama_tag": "llama3.1",
        "output_file": "outputs_local/PromptPersona_Full_600_llama31_8b.jsonl",
        "vram_mb": 6000,  # 4-bit quantized
    },
}

PROMPT_FILE = "PromptPersona_Full_600.csv"


def load_prompts():
    """Load 600 prompts from CSV"""
    prompts = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                "id": row.get("id", ""),
                "domain": row.get("domain", ""),
                "topic": row.get("topic", ""),
                "gender": row.get("gender", ""),
                "language": row.get("language", ""),
                "persona_code": row.get("persona_code", ""),
                "prompt_text": row.get("prompt_text", ""),
            })
    return prompts


def get_gpu_memory():
    """Check available GPU memory"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except:
        pass
    return 0


def wait_for_gpu_memory(required_mb, max_wait_sec=300):
    """Wait until required GPU memory is available"""
    start = time.time()
    while time.time() - start < max_wait_sec:
        available = get_gpu_memory()
        if available >= required_mb:
            print(f"✓ GPU memory available: {available}MB >= {required_mb}MB required")
            return True
        print(f"⏳ Waiting for GPU memory: {available}MB < {required_mb}MB (waited {int(time.time()-start)}s)")
        time.sleep(10)
    return False


def generate_with_ollama(prompt_text, model_tag, timeout=120):
    """Generate response using Ollama"""
    try:
        cmd = [
            "ollama", "run", "--nowordwrap",
            model_tag,
            f"Respond to this prompt concisely:\n\n{prompt_text}"
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"⚠ Generation failed: {result.stderr[:100]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⚠ Generation timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"⚠ Generation error: {str(e)}")
        return None


def process_model(model_config, prompts):
    """Process all prompts for a single model"""
    model_key = model_config["key"]
    model_name = model_config["name"]
    ollama_tag = model_config["ollama_tag"]
    output_file = model_config["output_file"]
    required_vram = model_config["vram_mb"]
    
    print(f"\n{'='*80}")
    print(f"📊 Processing {model_name} ({ollama_tag})")
    print(f"{'='*80}")
    print(f"Required VRAM: {required_vram}MB")
    print(f"Output: {output_file}\n")
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    if Path(output_file).exists():
        existing = sum(1 for _ in open(output_file))
        if existing >= 600:
            print(f"✓ {output_file} already complete ({existing}/600)")
            return existing
    
    # Wait for GPU memory
    if not wait_for_gpu_memory(required_vram):
        print(f"❌ Could not acquire {required_vram}MB GPU memory")
        return 0
    
    count = 0
    start_time = time.time()
    
    for idx, prompt_obj in enumerate(prompts, 1):
        try:
            prompt_text = prompt_obj.get("prompt_text", "")
            if not prompt_text:
                continue
            
            # Generate response
            print(f"[{model_name}] {idx}/600 - ID {prompt_obj['id']}", end=" ... ")
            response = generate_with_ollama(prompt_text, ollama_tag)
            
            if response:
                # Save result
                result = {
                    "id": prompt_obj.get("id"),
                    "domain": prompt_obj.get("domain"),
                    "topic": prompt_obj.get("topic"),
                    "gender": prompt_obj.get("gender"),
                    "language": prompt_obj.get("language"),
                    "persona_code": prompt_obj.get("persona_code"),
                    "prompt_text": prompt_text,
                    "model_key": model_key,
                    "model_repo": model_config["repo"],
                    "response": response,
                    "latency_sec": 0,  # Ollama doesn't provide timing
                }
                
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                
                count += 1
                elapsed = time.time() - start_time
                rate = count / (elapsed / 60)  # responses per minute
                print(f"✓ ({count}/600, {rate:.1f} per min)")
            else:
                print(f"✗ Failed")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted")
            break
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n✅ {model_name} complete: {count}/600 in {int(elapsed)}s")
    return count


def main():
    print(f"\n{'='*80}")
    print("🚀 Running Phi-3.5-mini and Llama-3.1-8B on 600 prompts")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Load prompts
    print("📂 Loading prompts...")
    prompts = load_prompts()
    print(f"✓ Loaded {len(prompts)} prompts\n")
    
    total_count = 0
    
    try:
        # Process each model
        for model_key, model_config in MODELS.items():
            count = process_model(model_config, prompts)
            total_count += count
            
            # Brief pause between models to let GPU cool down
            if model_key != list(MODELS.keys())[-1]:
                print(f"\n⏳ Cooling down GPU for 30s before next model...")
                time.sleep(30)
    
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total responses generated: {total_count}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
