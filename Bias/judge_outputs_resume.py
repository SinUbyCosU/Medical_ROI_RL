import json
import csv
import subprocess
import time
from pathlib import Path
from collections import defaultdict

# Batch configuration
BATCH_SIZE = 8
MIN_REQUIRED_OUTPUTS = 600

# Files to process
JSONL_FILES = [
    "outputs_local/PromptPersona_Full_600_mistral_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_phi3_medium.jsonl",
    "outputs_local/PromptPersona_Full_600_qwen25_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_yi15_6b.jsonl",
    "outputs_local/PromptPersona_Full_600_zephyr_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_qwen2_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_openhermes_mistral_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_nous_hermes_mistral_7b.jsonl",
    "outputs_local/PromptPersona_Full_600_phi35_mini.jsonl",
    "outputs_local/PromptPersona_Full_600_llama31_8b.jsonl",
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


def get_processed_ids(output_file):
    """Get all IDs already processed"""
    processed = defaultdict(set)
    if Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    model = obj.get("model", "")
                    id_val = obj.get("id", "")
                    if model and id_val:
                        processed[model].add(id_val)
                except json.JSONDecodeError:
                    continue
    return processed


def judge_with_ollama(text):
    """Judge response using ollama CLI"""
    prompt = PROMPT.format(response=text[:2000])
    try:
        cmd = ['ollama', 'run', '--nowordwrap', 'llama3', prompt]
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        output = result.stdout.strip()
        
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            try:
                json_str = output[json_start:json_end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"assertiveness": 5, "complexity": 5, "emotional_distance": 5, "instructional_density": 5}
        else:
            return {"assertiveness": 5, "complexity": 5, "emotional_distance": 5, "instructional_density": 5}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


def judge_batch(texts):
    """Process batch of texts"""
    results = []
    for text in texts:
        results.append(judge_with_ollama(text))
    return results


def process_jsonl_resume(path, output_file, processed_ids):
    """Resume processing JSONL file from where it stopped"""
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total = len(lines)
            
            if total < MIN_REQUIRED_OUTPUTS:
                print(f"⊘ Skipping {path} ({total} outputs < {MIN_REQUIRED_OUTPUTS} required)")
                return 0
            
            already_processed = len(processed_ids.get(path, set()))
            print(f"📝 Resuming {path} ({already_processed}/{total} already done)...")
            
            # Process in batches
            for batch_start in range(0, total, BATCH_SIZE):
                try:
                    batch_end = min(batch_start + BATCH_SIZE, total)
                    batch_lines = lines[batch_start:batch_end]
                    batch_data = []
                    
                    for idx, line in enumerate(batch_lines):
                        try:
                            obj = json.loads(line)
                            obj_id = obj.get("id", "")
                            
                            # Skip if already processed
                            if obj_id in processed_ids.get(path, set()):
                                continue
                            
                            response = obj.get("response")
                            if response:
                                batch_data.append((obj, response, batch_start + idx))
                        except json.JSONDecodeError:
                            continue
                    
                    # Process batch
                    if batch_data:
                        texts = [item[1] for item in batch_data]
                        scores = judge_batch(texts)
                        
                        for (obj, _, idx), score in zip(batch_data, scores):
                            result = {"id": obj.get("id"), "model": path, "score": score}
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                            count += 1
                        
                        pct = int(100 * (already_processed + count) / total)
                        print(f"[{path}] ✓ {already_processed + count}/{total} ({pct}%)")
                except Exception as e:
                    print(f"  ⚠ Batch error at {batch_start}: {str(e)}")
                    continue
        
        print(f"✅ Completed {path}: {already_processed + count}/{total}")
        return count
    except Exception as e:
        print(f"❌ Error processing {path}: {str(e)}")
        return count


def process_csv_resume(path, output_file, processed_ids):
    """Resume processing CSV file from where it stopped"""
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total = len(rows)
            
            if total < MIN_REQUIRED_OUTPUTS:
                print(f"⊘ Skipping {path} ({total} outputs < {MIN_REQUIRED_OUTPUTS} required)")
                return 0
            
            already_processed = len(processed_ids.get(path, set()))
            print(f"📊 Resuming {path} ({already_processed}/{total} already done)...")
            
            for batch_start in range(0, total, BATCH_SIZE):
                try:
                    batch_end = min(batch_start + BATCH_SIZE, total)
                    batch_rows = rows[batch_start:batch_end]
                    batch_data = []
                    
                    for idx, row in enumerate(batch_rows):
                        try:
                            row_id = row.get("id", "")
                            
                            # Skip if already processed
                            if row_id in processed_ids.get(path, set()):
                                continue
                            
                            response = row.get("response") or row.get("output") or row.get("text")
                            if response:
                                batch_data.append((row, response, batch_start + idx))
                        except Exception:
                            continue
                    
                    if batch_data:
                        texts = [item[1] for item in batch_data]
                        scores = judge_batch(texts)
                        
                        for (row, _, idx), score in zip(batch_data, scores):
                            result = {"id": row.get("id"), "model": path, "score": score}
                            with open(output_file, "a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                            count += 1
                        
                        pct = int(100 * (already_processed + count) / total)
                        print(f"[{path}] ✓ {already_processed + count}/{total} ({pct}%)")
                except Exception as e:
                    print(f"  ⚠ Batch error at {batch_start}: {str(e)}")
                    continue
        
        print(f"✅ Completed {path}: {already_processed + count}/{total}")
        return count
    except Exception as e:
        print(f"❌ Error processing {path}: {str(e)}")
        return count


def verify_completion(output_file):
    """Verify all files are complete"""
    print(f"\n{'=' * 80}")
    print("VERIFICATION: Checking completion status...")
    print(f"{'=' * 80}\n")
    
    processed = get_processed_ids(output_file)
    all_complete = True
    
    for file in JSONL_FILES + CSV_FILES:
        p = Path(file)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                if file.endswith(".jsonl"):
                    total = sum(1 for _ in f)
                else:
                    total = sum(1 for _ in f) - 1  # Exclude header
            
            processed_count = len(processed.get(file, set()))
            
            if processed_count >= 600:
                print(f"✅ {file}: {processed_count}/600+ ✓ COMPLETE")
            else:
                print(f"⏳ {file}: {processed_count}/600 (INCOMPLETE)")
                all_complete = False
    
    print(f"\n{'=' * 80}")
    if all_complete:
        print("🎉 ALL FILES COMPLETE!")
    else:
        print("⏳ Processing still in progress...")
    print(f"{'=' * 80}\n")
    
    return all_complete


def main():
    print("=" * 80)
    print("RESUMING LLaMA Guard 3 judging from where it stopped...")
    print("=" * 80)
    
    output_path = "outputs_local/judged_outputs_llamaguard.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get already processed IDs
    processed_ids = get_processed_ids(output_path)
    print(f"\n📊 Already processed: {sum(len(v) for v in processed_ids.values())} entries\n")
    
    total_count = 0
    
    try:
        # Process JSONL files
        for file in JSONL_FILES:
            p = Path(file)
            try:
                if p.exists():
                    print(f"\n{'─' * 80}")
                    count = process_jsonl_resume(file, output_path, processed_ids)
                    total_count += count
                else:
                    print(f"⚠ File not found: {file}")
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
                continue
        
        # Process CSV files
        for file in CSV_FILES:
            p = Path(file)
            try:
                if p.exists():
                    print(f"\n{'─' * 80}")
                    count = process_csv_resume(file, output_path, processed_ids)
                    total_count += count
                else:
                    print(f"⚠ File not found: {file}")
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
                continue
        
        # Verify completion
        verify_completion(output_path)
        
        print(f"\n📈 New entries processed in this run: {total_count}")
        print(f"📁 Total entries in output: {sum(1 for _ in open(output_path))}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
