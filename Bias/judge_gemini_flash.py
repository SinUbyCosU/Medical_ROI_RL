#!/usr/bin/env python3
"""
Judge Gemini Flash CSV outputs with llama3 and append to judged_outputs_llamaguard.jsonl
"""

import json
import csv
import subprocess
import time
from pathlib import Path

GEMINI_CSV = "outputs/PromptPersona_Full_600_gemini_flash.csv"
OUTPUT_FILE = "outputs_local/judged_outputs_llamaguard.jsonl"
BATCH_SIZE = 8

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

def load_gemini_outputs():
    """Load gemini_flash CSV"""
    data = []
    with open(GEMINI_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'id': row.get('id'),
                'model_response': row.get('model_response', ''),
                'domain': row.get('domain'),
                'topic': row.get('topic'),
                'gender': row.get('gender'),
                'language': row.get('language'),
            })
    return data

def judge_batch(batch_data):
    """Judge a batch of responses"""
    scores = []
    for item in batch_data:
        response = item.get('model_response', '')
        if response:
            score = judge_with_ollama(response)
            scores.append(score)
        else:
            scores.append(None)
    return scores

def main():
    print("\n" + "="*80)
    print("GEMINI FLASH JUDGING")
    print("="*80)
    
    # Load gemini outputs
    print("\n📂 Loading Gemini Flash outputs...")
    gemini_data = load_gemini_outputs()
    print(f"✅ Loaded {len(gemini_data)} responses")
    
    # Get already processed IDs
    processed_ids = set()
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get('model') == 'outputs_local/PromptPersona_Full_600_gemini_flash.csv':
                        processed_ids.add(obj.get('id'))
                except:
                    pass
    
    print(f"📊 Already processed: {len(processed_ids)} entries")
    
    # Filter out already processed
    to_process = [item for item in gemini_data if item['id'] not in processed_ids]
    print(f"⏳ Remaining to process: {len(to_process)}")
    
    if not to_process:
        print("✅ All entries already processed!")
        return
    
    # Process in batches
    count = 0
    start_time = time.time()
    
    for batch_start in range(0, len(to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(to_process))
        batch_data = to_process[batch_start:batch_end]
        
        print(f"\n[Batch {batch_start//BATCH_SIZE + 1}] Processing {batch_start}-{batch_end}...", end=" ", flush=True)
        
        # Judge batch
        scores = judge_batch(batch_data)
        
        # Save results
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
            for item, score in zip(batch_data, scores):
                if score and 'error' not in score:
                    result = {
                        "id": item.get('id'),
                        "model": "outputs_local/PromptPersona_Full_600_gemini_flash.csv",
                        "score": score
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    count += 1
        
        elapsed = time.time() - start_time
        rate = count / (elapsed / 60) if elapsed > 0 else 0
        print(f"✓ ({count} total, {rate:.1f}/min)")
        
        time.sleep(1)  # Brief pause between batches
    
    elapsed = time.time() - start_time
    print(f"\n✅ Complete: {count}/{len(to_process)} scored in {int(elapsed)}s")
    print(f"📊 Total in judged_outputs: {sum(1 for _ in open(OUTPUT_FILE))} lines")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
