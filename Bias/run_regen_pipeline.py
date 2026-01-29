#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

def run_step(cmd, title, timeout=3600):
    print("\n" + "="*80)
    print(f"STEP: {title}")
    print("="*80)
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        if result.returncode == 0:
            print(f"\n✅ {title} - SUCCESS\n")
            return True
        else:
            print(f"\n⚠️  {title} - Code {result.returncode}\n")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  {title} - TIMEOUT\n")
        return False
    except Exception as e:
        print(f"\n⚠️  {title} - ERROR: {e}\n")
        return False

print("\n" + "="*80)
print("REGENERATION & JUDGING PIPELINE")
print("="*80)

steps = [
    ("cd /root/Bias && source /root/.venv/bin/activate && python step1_regenerate_worst.py 2>&1 | tee regen_step1.log", 
     "Step 1/2: Regenerate Worst-Rated Prompts", 1800),
    ("cd /root/Bias && source /root/.venv/bin/activate && python step2_judge_regenerated.py 2>&1 | tee regen_step2.log",
     "Step 2/2: Judge Regenerated with LlamaGuard-3", 1200),
]

results = []
for cmd, title, timeout in steps:
    success = run_step(cmd, title, timeout)
    results.append((title, success))
    time.sleep(2)

print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)
for title, success in results:
    status = "✅" if success else "⚠️"
    print(f"{status} {title}")

print("\n📁 Output: /root/Bias/outputs_local/regenerated/")
