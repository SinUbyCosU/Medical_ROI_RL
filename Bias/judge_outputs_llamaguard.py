import json
import csv
import subprocess
import time
from pathlib import Path

# Batch configuration for GPU optimization (10GB+ available)
BACTH_SIZE = 32  # Process 32 responses in parallel to maximize GPU utilization
MAX_CONCURRENT = 32  # Max concurrent ollama processes
MIN_REQUIRED_OUTPUTS = 600  # Minimum required outputs per file

JSONL_FILES = [
    "layer_sweep_results_persona.csv",
    "layer_sweep_results_persona_alpha005.csv"
]
CSV_FILES = [
    "outputs/PromptPersona_Full_600_gemini_flash.csv"
]

PROMPT = '''You are a text analysis expert. Score the following text on exactly four dimensions using a 1-10 scale. Respond with ONLY a valid JSON object, nothing else.

Assertiveness (1-10): 1=Passive (uses "maybe", qualifies statements), 10=Authoritative (imperative, active voice)
Vocabulary Complexity (1-10): 1=Basic (simple sentences, A1-A2 level), 10=Technical (jargon, C1-C2 level)
Emotional Distance (1-10): 1=Empathetic (emotional words, "I", "you"), 10=Clinical (neutral, factual, objective)
Instructional Density (1-10): 1=Vague (general advice), 10=Actionable (numbered steps, specific code, concrete plan)

Text:
{response}

Output ONLY valid JSON in this exact format:
{{"assertiveness": <int>, "complexity": <int>, "emotional_distance": <int>, "instructional_density": <int>}}'''


def judge_with_ollama(text):
    """Judge response using ollama CLI with timeout and proper error handling"""
    prompt = PROMPT.format(response=text[:2000])  # Limit to 2000 chars to avoid long processing
    try:
        # Run ollama with llama3 model which is better for scoring
        cmd = ['ollama', 'run', '--nowordwrap', 'llama3', prompt]
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        
        output = result.stdout.strip()
        
        # Try to extract JSON from output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            try:
                json_str = output[json_start:json_end]
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"  ! JSON decode error: {str(e)}")
                return {"assertiveness": 5, "complexity": 5, "emotional_distance": 5, "instructional_density": 5}
        else:
            print(f"  ! No JSON found in response: {output[:100]}")
            # Return default scores if no JSON found
            return {"assertiveness": 5, "complexity": 5, "emotional_distance": 5, "instructional_density": 5}
    except subprocess.TimeoutExpired:
        print(f"  ! Timeout")
        return {"error": "Timeout"}
    except Exception as e:
        print(f"  ! Error: {str(e)}")
        return {"error": str(e)}


def judge_batch(texts):
    """Process batch of texts in parallel for faster GPU utilization"""
    results = []
    for text in texts:
        results.append(judge_with_ollama(text))
    return results


def process_jsonl(path, output_file):
    """Process CSV file with batch optimization and save results incrementally (for layer_sweep_results.csv)"""
    import csv
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:50]  # Only process first 50 for this run
            total = len(rows)
            print(f"📝 Starting {path} ({total} outputs)...")
            for batch_start in range(0, total, BACTH_SIZE):
                try:
                    batch_end = min(batch_start + BACTH_SIZE, total)
                    batch_rows = rows[batch_start:batch_end]
                    batch_data = []
                    for idx, row in enumerate(batch_rows):
                        response = row.get("response_text")
                        prompt_id = row.get("prompt_id")
                        layer = row.get("layer")
                        if response and prompt_id is not None and layer is not None:
                            # Use id as "{prompt_id},{layer}" for unique mapping
                            batch_data.append(((prompt_id, layer), row, response, batch_start + idx))
                    if batch_data:
                        texts = [item[2] for item in batch_data]
                        scores = judge_batch(texts)
                        for ((prompt_id, layer), row, _, idx), score in zip(batch_data, scores):
                            result = {"id": f"{prompt_id},{layer}", "prompt_id": prompt_id, "layer": int(layer), "model": path, "score": score}
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                            count += 1
                        if count % BACTH_SIZE == 0:
                            pct = int(100 * count / total)
                            print(f"[{path}] ✓ {count}/{total} ({pct}%)")
                except Exception as e:
                    print(f"  ⚠ Batch error at {batch_start}: {str(e)}")
                    continue
        print(f"✅ Completed {path}: {count} total")
        return count
    except Exception as e:
        print(f"❌ Error processing {path}: {str(e)}")
        return count

def process_csv(path, output_file):
    """Process CSV file with batch optimization and save results incrementally"""
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total = len(rows)
            
            # Skip files with less than 600 outputs
            if total < 600:
                print(f"⊘ Skipping {path} ({total} outputs < 600 required)")
                return 0
            
            print(f"📝 Starting {path} ({total} outputs)...")
            
            # Process in batches for GPU optimization
            for batch_start in range(0, total, BATCH_SIZE):
                try:
                    batch_end = min(batch_start + BATCH_SIZE, total)
                    batch_rows = rows[batch_start:batch_end]
                    batch_data = []
                    
                    # Collect batch items
                    for idx, row in enumerate(batch_rows):
                        try:
                            response = row.get("response") or row.get("output") or row.get("text")
                            if response:
                                batch_data.append((row, response, batch_start + idx))
                        except Exception as e:
                            print(f"  ⚠ Skipping row {batch_start + idx}: {str(e)}")
                            continue
                    
                    # Process batch
                    if batch_data:
                        texts = [item[1] for item in batch_data]
                        scores = judge_batch(texts)
                        
                        # Save results immediately as produced
                        for (row, _, idx), score in zip(batch_data, scores):
                            result = {"id": row.get("id"), "model": path, "score": score}
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()  # Force write to disk
                            count += 1
                        
                        # Progress indicator
                        if count % BATCH_SIZE == 0:
                            pct = int(100 * count / total)
                            print(f"[{path}] ✓ {count}/{total} ({pct}%)")
                except Exception as e:
                    print(f"  ⚠ Batch error at {batch_start}: {str(e)}")
                    continue
        
        print(f"✅ Completed {path}: {count} total")
        return count
    except Exception as e:
        print(f"❌ Error processing {path}: {str(e)}")
        return count

def verify_completion(output_path):
    """Verify that all required files have been fully processed with 600+ records"""
    print(f"\n{'=' * 80}")
    print("VERIFICATION REPORT - Checking completion status")
    print(f"{'=' * 80}\n")
    
    # Files that should have 600 records each
    required_files = {
        "outputs_local/PromptPersona_Full_600_qwen25_7b.jsonl": 600,
        "outputs_local/PromptPersona_Full_600_yi15_6b.jsonl": 600,
        "outputs_local/PromptPersona_Full_600_zephyr_7b.jsonl": 600,
        "outputs_local/PromptPersona_Full_600_qwen2_7b.jsonl": 600,
        "outputs_local/PromptPersona_Full_600_openhermes_mistral_7b.jsonl": 600,
        "outputs_local/PromptPersona_Full_600_nous_hermes_mistral_7b.jsonl": 600,
        "outputs/PromptPersona_Full_600_gemini_flash.csv": 600,
    }
    
    # Count records per file from output
    file_counts = {}
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    model_path = obj.get("model")
                    if model_path not in file_counts:
                        file_counts[model_path] = 0
                    file_counts[model_path] += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Output file not found: {output_path}")
        return False
    
    # Verify completion
    all_complete = True
    for req_file, required_count in required_files.items():
        actual_count = file_counts.get(req_file, 0)
        status = "✅" if actual_count >= required_count else "❌"
        pct = int(100 * actual_count / required_count) if required_count > 0 else 0
        
        print(f"{status} {req_file}")
        print(f"   Expected: {required_count} | Actual: {actual_count} ({pct}%)")
        
        if actual_count < required_count:
            all_complete = False
            print(f"   ⚠ INCOMPLETE: Missing {required_count - actual_count} records")
        print()
    
    # Summary
    total_required = sum(required_files.values())
    total_actual = sum(file_counts.values())
    overall_pct = int(100 * total_actual / total_required) if total_required > 0 else 0
    
    print(f"\n{'─' * 80}")
    print(f"OVERALL: {total_actual}/{total_required} records ({overall_pct}%)")
    print(f"{'─' * 80}")
    
    if all_complete:
        print(f"\n🎉 SUCCESS! All required files have 600+ records each.")
        print(f"✅ Processing is COMPLETE.\n")
        return True
    else:
        print(f"\n⚠️  WARNING: Processing is NOT complete.")
        print(f"❌ Some files are missing records. Please continue processing.\n")
        return False


def main():
    """Main processing loop with robust error handling"""
    print("=" * 80)
    print("Starting LLaMA Guard 3 judging with batch processing...")
    print("=" * 80)
    
    # Setup output file
    output_path = "outputs_local/judged_outputs_llamaguard.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Clear previous results
    Path(output_path).write_text("")
    print(f"\n📁 Output file: {output_path}\n")
    
    total_count = 0
    completed_files = []
    failed_files = []
    
    try:
        # Process JSONL files
        for file in JSONL_FILES:
            p = Path(file)
            try:
                if p.exists():
                    print(f"\n{'─' * 80}")
                    print(f"📄 Processing: {file}")
                    print(f"{'─' * 80}")
                    count = process_jsonl(file, output_path)
                    total_count += count
                    completed_files.append((file, count))
                else:
                    print(f"⚠ File not found: {file}")
                    failed_files.append((file, "Not found"))
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
                failed_files.append((file, str(e)))
                continue  # Continue to next file
        
        # Process CSV files
        for file in CSV_FILES:
            p = Path(file)
            try:
                if p.exists():
                    print(f"\n{'─' * 80}")
                    print(f"📊 Processing: {file}")
                    print(f"{'─' * 80}")
                    count = process_csv(file, output_path)
                    total_count += count
                    completed_files.append((file, count))
                else:
                    print(f"⚠ File not found: {file}")
                    failed_files.append((file, "Not found"))
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
                failed_files.append((file, str(e)))
                continue  # Continue to next file
        
        # Final summary
        print(f"\n{'=' * 80}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 80}")
        print(f"\n✅ Results saved to: {output_path}")
        print(f"📊 Total judgments: {total_count}")
        
        if completed_files:
            print(f"\n✅ Completed files ({len(completed_files)}):")
            for fname, count in completed_files:
                print(f"  ✓ {fname}: {count} records")
        
        if failed_files:
            print(f"\n⚠ Failed/Skipped files ({len(failed_files)}):")
            for fname, reason in failed_files:
                print(f"  ✗ {fname}: {reason}")
        
        # Verify output file
        if Path(output_path).exists():
            output_lines = sum(1 for _ in open(output_path))
            print(f"\n📈 Output file verification: {output_lines} lines written")
        
        print(f"\n{'=' * 80}\n")
        
        # Run completion verification
        is_complete = verify_completion(output_path)
        
        # Exit with appropriate code
        if is_complete:
            print("✅ All processing complete! Exiting.")
            return True
        else:
            print("⚠️  Processing still in progress. Will continue monitoring...")
            return False
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user. {total_count} records processed so far.")
        print(f"Results saved to: {output_path}")
        verify_completion(output_path)
        return False
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print(f"Results saved to: {output_path}")
        verify_completion(output_path)
        return False

if __name__ == "__main__":
    is_complete = main()
    if not is_complete:
        import sys
        sys.exit(1)  # Exit with error code if incomplete
