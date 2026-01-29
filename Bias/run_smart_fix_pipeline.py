#!/usr/bin/env python3
"""
Smart Fix Pipeline: Run Low-Quality Response Fixes + Score Comparison
Runs all steps sequentially in a single process with error resilience
"""

import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("smart_fix_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_command(cmd, description, timeout=3600):
    """Run command and handle errors gracefully"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - SUCCESS")
            return True
        else:
            print(f"\n⚠️  {description} - returned code {result.returncode}")
            print("   Continuing despite error...")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  {description} - TIMEOUT after {timeout}s")
        print("   Continuing despite timeout...")
        return False
    except Exception as e:
        print(f"\n⚠️  {description} - ERROR: {e}")
        print("   Continuing despite error...")
        return False

def main():
    print("\n" + "="*80)
    print("SMART FIX PIPELINE - RUNNING ALL STEPS")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = [
        (
            "cd /root/Bias && source /root/.venv/bin/activate && python smart_fix_low_quality.py 2>&1 | tee smart_fix_step1.log",
            "Step 1/2: Smart Fix Low-Quality Responses",
            600  # 10 minutes timeout
        ),
        (
            "cd /root/Bias && source /root/.venv/bin/activate && python compare_score_improvements.py 2>&1 | tee score_comparison_step2.log",
            "Step 2/2: Compare Score Improvements",
            300   # 5 minutes timeout
        ),
    ]
    
    results = []
    
    for cmd, description, timeout in steps:
        success = run_command(cmd, description, timeout)
        results.append((description, success))
        time.sleep(2)  # Brief pause between steps
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for step, success in results:
        status = "✅ SUCCESS" if success else "⚠️  PARTIAL/ERROR"
        print(f"{status}: {step}")
    
    overall_success = all(s for _, s in results)
    
    print("\n" + "="*80)
    if overall_success:
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY")
    else:
        print("⚠️  SOME STEPS HAD ISSUES - CHECK LOGS")
    print("="*80)
    
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print("   - smart_fix_low_quality_results.csv")
    print("   - SMART_FIX_LOW_QUALITY_REPORT.md")
    print("   - SCORE_IMPROVEMENT_REPORT.md")
    print("   - score_improvement_comparison.png")
    print("   - score_improvement_stats.json")
    
    print(f"\n📋 Log files:")
    print("   - smart_fix_step1.log")
    print("   - score_comparison_step2.log")

if __name__ == "__main__":
    main()
