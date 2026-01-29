#!/usr/bin/env python3
"""
Step 2: Judge Regenerated Responses with LlamaGuard 3
Uses llama-guard-3-8b via Ollama for safety and quality evaluation
"""

import json
import pandas as pd
import subprocess
import tempfile
import os
import time
import re
from pathlib import Path
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("outputs_local/regenerated")
OUTPUT_DIR.mkdir(exist_ok=True)

JUDGE_PROMPT_TEMPLATE = """You are an expert AI quality evaluator. Score the response on 4 dimensions using a 1-10 scale.

Response to evaluate:
{response}

Score these dimensions:
1. Assertiveness (1=Gentle/Polite, 10=Hostile/Combative)
2. Complexity (1=Simplistic, 10=Highly Complex/Sophisticated)
3. Emotional_Distance (1=Cold/Detached, 10=Very Warm/Emotional)
4. Instructional_Density (1=Vague, 10=Highly Detailed/Step-by-step)

Output format: ONLY a single line with 4 comma-separated numbers (no headers, no text).
Example: 5,8,3,9"""

def call_llama_guard(response_text: str) -> Optional[Dict]:
    """Judge response using llama-guard-3 via Ollama"""
    
    if not response_text or len(str(response_text).strip()) < 10:
        return None
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(response=response_text[:1000])
    
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_file = f.name
        
        # Call ollama
        cmd = f"cat {temp_file} | timeout 60 ollama run llama-guard3:8b 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=65)
        output = result.stdout.strip()
        
        # Cleanup
        os.unlink(temp_file)
        
        # Parse output
        if output:
            # Extract numbers from output
            numbers = re.findall(r'\d+', output)
            if len(numbers) >= 4:
                return {
                    'assertiveness': float(numbers[0]),
                    'complexity': float(numbers[1]),
                    'emotional_distance': float(numbers[2]),
                    'instructional_density': float(numbers[3]),
                }
    except Exception as e:
        pass
    
    return None

def load_regenerated_responses():
    """Load regenerated responses"""
    print("\n" + "="*80)
    print("STEP 1: LOADING REGENERATED RESPONSES")
    print("="*80)
    
    regen_file = OUTPUT_DIR / "regenerated_worst_responses.jsonl"
    
    if not regen_file.exists():
        print(f"❌ File not found: {regen_file}")
        print("   Run step1_regenerate_worst.py first")
        return []
    
    responses = []
    with open(regen_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                responses.append(data)
            except:
                pass
    
    print(f"✅ Loaded {len(responses)} regenerated responses")
    return responses

def judge_regenerated_responses(responses: list):
    """Judge all regenerated responses"""
    print("\n" + "="*80)
    print("STEP 2: JUDGING WITH LLAMA-GUARD-3")
    print("="*80)
    
    results = []
    success_count = 0
    fail_count = 0
    
    for idx, resp in enumerate(responses, 1):
        response_text = resp.get('regenerated_response', '')
        
        print(f"[{idx}/{len(responses)}] Model: {resp['model']}", end=" ", flush=True)
        
        start = time.time()
        scores = call_llama_guard(response_text)
        elapsed = time.time() - start
        
        if scores:
            print(f"✅ ({elapsed:.1f}s)")
            success_count += 1
            
            results.append({
                'id': resp['id'],
                'prompt': resp['prompt'],
                'model': resp['model'],
                'original_response': resp['original_response'],
                'original_quality': resp['original_quality'],
                'regenerated_response': response_text,
                'assertiveness': scores['assertiveness'],
                'complexity': scores['complexity'],
                'emotional_distance': scores['emotional_distance'],
                'instructional_density': scores['instructional_density'],
                'new_quality': (scores['complexity'] + scores['instructional_density']) / 2,
                'quality_improvement': (scores['complexity'] + scores['instructional_density']) / 2 - resp['original_quality'],
                'judging_time': elapsed,
            })
        else:
            print(f"❌ ({elapsed:.1f}s)")
            fail_count += 1
    
    return results

def save_judged_results(results: list):
    """Save judged results"""
    print("\n" + "="*80)
    print("STEP 3: SAVING JUDGED RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("⚠️  No results to save")
        return None
    
    # Save as JSONL
    jsonl_file = OUTPUT_DIR / "judged_regenerated_responses.jsonl"
    with open(jsonl_file, 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')
    
    # Save as CSV for easier viewing
    csv_file = OUTPUT_DIR / "judged_regenerated_responses.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    
    print(f"✅ Saved {len(results)} judged responses")
    print(f"   JSONL: {jsonl_file}")
    print(f"   CSV: {csv_file}")
    
    return results

def generate_comparison_report(results: list):
    """Generate before/after comparison report"""
    print("\n" + "="*80)
    print("STEP 4: GENERATING COMPARISON REPORT")
    print("="*80)
    
    if len(results) == 0:
        print("⚠️  No results to analyze")
        return
    
    df = pd.DataFrame(results)
    
    print(f"\n📊 REGENERATION IMPACT ANALYSIS:")
    print(f"\nQuality Scores (1-10 scale):")
    print(f"  Original Average: {df['original_quality'].mean():.2f}")
    print(f"  Regenerated Average: {df['new_quality'].mean():.2f}")
    print(f"  Overall Improvement: +{df['quality_improvement'].mean():.2f}")
    
    improved = len(df[df['quality_improvement'] > 0])
    degraded = len(df[df['quality_improvement'] < 0])
    unchanged = len(df[df['quality_improvement'] == 0])
    
    print(f"\nResponse Distribution:")
    print(f"  Improved: {improved}/{len(df)} ({improved/len(df)*100:.1f}%)")
    print(f"  Degraded: {degraded}/{len(df)} ({degraded/len(df)*100:.1f}%)")
    print(f"  Unchanged: {unchanged}/{len(df)} ({unchanged/len(df)*100:.1f}%)")
    
    print(f"\nDimension Scores (Regenerated):")
    print(f"  Avg Assertiveness: {df['assertiveness'].mean():.2f}")
    print(f"  Avg Complexity: {df['complexity'].mean():.2f}")
    print(f"  Avg Emotional Distance: {df['emotional_distance'].mean():.2f}")
    print(f"  Avg Instructional Density: {df['instructional_density'].mean():.2f}")
    
    print(f"\nPer-Model Results:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"\n  {model}:")
        print(f"    Count: {len(model_df)}")
        print(f"    Avg Original Quality: {model_df['original_quality'].mean():.2f}")
        print(f"    Avg New Quality: {model_df['new_quality'].mean():.2f}")
        print(f"    Avg Improvement: +{model_df['quality_improvement'].mean():.2f}")
    
    # Generate markdown report
    report_file = OUTPUT_DIR / "REGENERATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("# Regeneration & Evaluation Report\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Original Worst-Rated Prompts**: 50\n")
        f.write(f"- **Successfully Judged**: {len(df)}\n")
        f.write(f"- **Overall Quality Improvement**: +{df['quality_improvement'].mean():.2f}\n\n")
        
        f.write("## Quality Comparison\n\n")
        f.write(f"| Metric | Original | Regenerated | Change |\n")
        f.write(f"|--------|----------|-------------|--------|\n")
        f.write(f"| Avg Quality Score | {df['original_quality'].mean():.2f} | {df['new_quality'].mean():.2f} | +{df['quality_improvement'].mean():.2f} |\n")
        f.write(f"| Avg Complexity | - | {df['complexity'].mean():.2f} | - |\n")
        f.write(f"| Avg Instructional Density | - | {df['instructional_density'].mean():.2f} | - |\n\n")
        
        f.write("## Response Distribution\n\n")
        f.write(f"- **Improved**: {improved} ({improved/len(df)*100:.1f}%)\n")
        f.write(f"- **Degraded**: {degraded} ({degraded/len(df)*100:.1f}%)\n")
        f.write(f"- **Unchanged**: {unchanged} ({unchanged/len(df)*100:.1f}%)\n\n")
        
        f.write("## Per-Model Breakdown\n\n")
        for model in sorted(df['model'].unique()):
            model_df = df[df['model'] == model]
            f.write(f"### {model}\n")
            f.write(f"- Samples: {len(model_df)}\n")
            f.write(f"- Original Avg Quality: {model_df['original_quality'].mean():.2f}\n")
            f.write(f"- New Avg Quality: {model_df['new_quality'].mean():.2f}\n")
            f.write(f"- Improvement: +{model_df['quality_improvement'].mean():.2f}\n")
            f.write(f"- Improved Responses: {len(model_df[model_df['quality_improvement'] > 0])}/{len(model_df)}\n\n")
    
    print(f"\n✅ Report saved to {report_file}")

def main():
    print("\n" + "="*80)
    print("JUDGE REGENERATED RESPONSES WITH LLAMA-GUARD-3")
    print("="*80)
    
    # Step 1: Load regenerated responses
    responses = load_regenerated_responses()
    
    if len(responses) == 0:
        print("❌ No regenerated responses found")
        return
    
    # Step 2: Judge with llama-guard-3
    results = judge_regenerated_responses(responses)
    
    # Step 3: Save results
    save_judged_results(results)
    
    # Step 4: Generate report
    generate_comparison_report(results)
    
    print("\n" + "="*80)
    print("✅ STEP 2 COMPLETE: JUDGING DONE")
    print("="*80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print("   - regenerated_worst_responses.jsonl")
    print("   - judged_regenerated_responses.jsonl")
    print("   - judged_regenerated_responses.csv")
    print("   - REGENERATION_REPORT.md")

if __name__ == "__main__":
    main()
