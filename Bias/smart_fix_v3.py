#!/usr/bin/env python3
"""
Smart Fix Mitigation V3: English-First Strategy for Hinglish Responses
Uses proper file-based prompt delivery to Ollama
"""

import json
import pandas as pd
import numpy as np
import os
import re
import time
import tempfile
from pathlib import Path
import glob
import warnings
import subprocess

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("smart_fix_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def count_steps(text):
    """Count numbered/bulleted steps"""
    if not text or pd.isna(text):
        return 0
    
    text = str(text).lower()
    
    # Count numbered lists (1., 2., 1), 2), etc)
    numbered = len(re.findall(r'\n\s*\d+[\.\)]\s+', text))
    bullets = len(re.findall(r'\n\s*[-•*]\s+', text))
    
    return max(numbered + bullets, 1)

def run_smart_fix(prompt):
    """Generate smart fix response using Ollama llama3 via file"""
    if pd.isna(prompt) or len(str(prompt).strip()) < 10:
        return None
    
    prompt_text = str(prompt)[:300]
    
    # Build the smart fix prompt
    smart_prompt = f"""You are an expert AI assistant.

Hinglish Query: "{prompt_text}"

INSTRUCTIONS:
1. THINKING: Write detailed technical solution in ENGLISH with numbered steps
2. TRANSLATION: Translate to Hinglish (Hindi + English) keeping all details

Format:
[THOUGHT]: Your English solution with steps
[RESPONSE]: Your Hinglish translation"""
    
    try:
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(smart_prompt)
            temp_file = f.name
        
        # Read and pipe to ollama
        cmd = f"cat {temp_file} | timeout 45 ollama run llama3 2>/dev/null"
        output = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=50).stdout
        
        # Cleanup
        os.unlink(temp_file)
        
        if "[RESPONSE]:" in output:
            response = output.split("[RESPONSE]:")[1].strip()
            if len(response) > 50:
                return response
        elif len(output) > 50:
            # If no marker, return the output if it's substantial
            return output[:500]
    except Exception as e:
        pass
    
    return None

def load_model_responses():
    """Load responses from original model JSONL files"""
    print("\n" + "="*80)
    print("LOADING MODEL RESPONSES")
    print("="*80)
    
    responses = []
    jsonl_files = glob.glob("outputs_local/*.jsonl")
    
    print(f"\nFound {len(jsonl_files)} model output files")
    
    for jsonl_file in jsonl_files:
        model_name = Path(jsonl_file).stem
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    responses.append({
                        'id': data.get('id', ''),
                        'text': data.get('text', ''),
                        'model': model_name
                    })
                except:
                    pass
    
    print(f"✅ Loaded {len(responses)} responses from models")
    return pd.DataFrame(responses)

def run_experiment(df, sample_size=20):
    """Run smart fix on low-quality responses"""
    print("\n" + "="*80)
    print(f"SMART FIX EXPERIMENT ({sample_size} samples)")
    print("="*80)
    
    # Count steps in original responses
    df['original_steps'] = df['text'].apply(count_steps)
    
    # Find low-quality responses (< 4 steps)
    low_quality = df[df['original_steps'] < 4].copy()
    print(f"\nFound {len(low_quality)} low-quality responses (< 4 steps)")
    
    if len(low_quality) == 0:
        print("⚠️  No low-quality responses found")
        return pd.DataFrame()
    
    # Sample
    sample = low_quality.sample(min(sample_size, len(low_quality)), random_state=42)
    
    results = []
    
    for idx, (i, row) in enumerate(sample.iterrows(), 1):
        prompt = row['text']
        original_steps = row['original_steps']
        
        print(f"\n[{idx}/{sample_size}] Original: {original_steps} steps")
        print(f"  Prompt: {str(prompt)[:100]}...")
        print(f"  Running smart fix...", end=" ", flush=True)
        
        start = time.time()
        fixed = run_smart_fix(prompt)
        elapsed = time.time() - start
        
        if fixed:
            fixed_steps = count_steps(fixed)
            improvement = fixed_steps - original_steps
            success = improvement > 0
            print(f"✅ ({elapsed:.1f}s)")
            print(f"  Result: {fixed_steps} steps (improvement: +{improvement})")
            
            results.append({
                'id': row['id'],
                'model': row['model'],
                'original_text': prompt,
                'original_steps': original_steps,
                'fixed_text': fixed,
                'fixed_steps': fixed_steps,
                'improvement': improvement,
                'success': success,
                'time_seconds': elapsed
            })
        else:
            print(f"❌ ({elapsed:.1f}s)")
            print(f"  Failed to generate response")
    
    return pd.DataFrame(results)

def analyze_and_report(results_df):
    """Generate analysis report"""
    print("\n" + "="*80)
    print("ANALYSIS & REPORT")
    print("="*80)
    
    if len(results_df) == 0:
        print("\n⚠️  No results to analyze")
        return
    
    success_count = results_df['success'].sum()
    total_count = len(results_df)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    avg_original = results_df['original_steps'].mean()
    avg_fixed = results_df['fixed_steps'].mean()
    avg_improvement = results_df['improvement'].mean()
    
    print(f"\n📊 RESULTS:")
    print(f"  Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    print(f"  Original Steps (avg): {avg_original:.2f}")
    print(f"  Fixed Steps (avg): {avg_fixed:.2f}")
    print(f"  Average Improvement: +{avg_improvement:.2f} steps")
    print(f"  Total Time: {results_df['time_seconds'].sum():.1f}s")
    
    # Save results
    csv_file = OUTPUT_DIR / "smart_fix_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n✅ Saved results to {csv_file}")
    
    # Generate markdown report
    report_file = OUTPUT_DIR / "SMART_FIX_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("# Smart Fix Mitigation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Experiment Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Sample Size**: {total_count} responses\n")
        f.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")
        
        f.write("## Quality Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Original Steps (avg) | {avg_original:.2f} |\n")
        f.write(f"| Fixed Steps (avg) | {avg_fixed:.2f} |\n")
        f.write(f"| Average Improvement | +{avg_improvement:.2f} steps |\n")
        f.write(f"| Total Execution Time | {results_df['time_seconds'].sum():.1f}s |\n\n")
        
        f.write("## Individual Results\n\n")
        for idx, row in results_df.iterrows():
            f.write(f"### Sample {idx + 1}\n")
            f.write(f"- **Status**: {'✅ Success' if row['success'] else '❌ Failed'}\n")
            f.write(f"- **Model**: {row['model']}\n")
            f.write(f"- **Original Steps**: {row['original_steps']}\n")
            f.write(f"- **Fixed Steps**: {row['fixed_steps']}\n")
            f.write(f"- **Improvement**: {row['improvement']:+.0f} steps\n")
            f.write(f"- **Time**: {row['time_seconds']:.1f}s\n\n")
    
    print(f"✅ Saved report to {report_file}")

def main():
    print("\n" + "="*80)
    print("SMART FIX MITIGATION: English-First Strategy for Hinglish")
    print("="*80)
    
    # Load responses
    df = load_model_responses()
    
    if len(df) == 0:
        print("❌ No responses loaded")
        return
    
    # Run experiment
    results = run_experiment(df, sample_size=20)
    
    # Analyze
    analyze_and_report(results)
    
    print("\n" + "="*80)
    print("✅ SMART FIX EXPERIMENT COMPLETE")
    print("="*80)
    print(f"📁 Results: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
