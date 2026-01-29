"""
Run Phi-3.5 Mini and Llama 3.1 8B on 600 prompts
Models run in parallel while evaluation continues
"""
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

# Models to run
MODELS = {
    "phi35_mini": {
        "repo": "microsoft/Phi-3.5-mini-instruct",
        "output": "outputs_local/PromptPersona_Full_600_phi35_mini.jsonl",
        "dtype": "bfloat16",
    },
    "llama31_8b": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "output": "outputs_local/PromptPersona_Full_600_llama31_8b.jsonl",
        "dtype": "bfloat16",
    },
}

PROMPTS_FILE = "outputs_local/PromptPersona_Full_600_qwen25_7b.jsonl"
MAX_NEW_TOKENS = 256


def load_prompts():
    """Load 600 prompts from existing output"""
    prompts = []
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append({
                "id": obj["id"],
                "prompt_text": obj["prompt_text"],
                "domain": obj.get("domain", ""),
                "topic": obj.get("topic", ""),
                "gender": obj.get("gender", ""),
                "language": obj.get("language", ""),
                "persona_code": obj.get("persona_code", ""),
            })
    return prompts


def build_text_pipeline(model_key, model_config):
    """Build inference pipeline"""
    print(f"\n{'='*80}")
    print(f"Loading model: {model_key}")
    print(f"Repo: {model_config['repo']}")
    print(f"{'='*80}\n")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["repo"],
            trust_remote_code=True,
            timeout=30,
        )
        
        # Use bfloat16 for efficiency
        dtype = torch.bfloat16 if model_config["dtype"] == "bfloat16" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config["repo"],
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            timeout=30,
        )
        
        pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=0.95,
        )
        
        print(f"✅ Model loaded: {model_key}")
        return pipeline
    except Exception as e:
        print(f"❌ Error loading {model_key}: {str(e)}")
        raise


def run_model(model_key, model_config, prompts):
    """Run model on all prompts"""
    print(f"\n{'='*80}")
    print(f"Running {model_key} on {len(prompts)} prompts")
    print(f"{'='*80}\n")
    
    pipeline = build_text_pipeline(model_key, model_config)
    output_file = model_config["output"]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file
    Path(output_file).write_text("")
    
    for idx, prompt in enumerate(prompts, 1):
        try:
            start_time = time.time()
            
            # Generate response
            outputs = pipeline(prompt["prompt_text"], return_full_text=False)
            response = outputs[0]["generated_text"]
            
            latency = time.time() - start_time
            
            # Save result
            result = {
                "id": prompt["id"],
                "domain": prompt["domain"],
                "topic": prompt["topic"],
                "gender": prompt["gender"],
                "language": prompt["language"],
                "persona_code": prompt["persona_code"],
                "prompt_text": prompt["prompt_text"],
                "model_key": model_key,
                "model_repo": model_config["repo"],
                "response": response,
                "latency_sec": round(latency, 3),
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            
            # Progress
            if idx % 50 == 0:
                print(f"[{model_key}] ✓ {idx}/{len(prompts)} ({100*idx//len(prompts)}%) - {latency:.2f}s/prompt")
        
        except Exception as e:
            print(f"  ⚠ Error on prompt {idx}: {str(e)}")
            # Save error record
            result = {
                "id": prompt["id"],
                "domain": prompt["domain"],
                "topic": prompt["topic"],
                "gender": prompt["gender"],
                "language": prompt["language"],
                "persona_code": prompt["persona_code"],
                "prompt_text": prompt["prompt_text"],
                "model_key": model_key,
                "model_repo": model_config["repo"],
                "response": f"ERROR: {str(e)}",
                "latency_sec": 0,
            }
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            continue
    
    # Final summary
    output_count = sum(1 for _ in open(output_file))
    print(f"\n✅ Completed {model_key}: {output_count}/{len(prompts)} responses")
    print(f"📁 Output: {output_file}\n")


def main():
    print("\n" + "="*80)
    print("Running Phi-3.5 Mini & Llama 3.1 8B on 600 prompts")
    print("="*80 + "\n")
    
    # Load prompts
    prompts = load_prompts()
    print(f"📊 Loaded {len(prompts)} prompts from {PROMPTS_FILE}\n")
    
    # Run models sequentially (can't run both on same GPU simultaneously at full precision)
    for model_key, model_config in MODELS.items():
        try:
            run_model(model_key, model_config, prompts)
        except Exception as e:
            print(f"❌ Failed to run {model_key}: {str(e)}")
            continue
    
    print("="*80)
    print("🎉 All models completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
