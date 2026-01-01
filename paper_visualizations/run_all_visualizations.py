#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

def run_script(script_path, description):
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path.name}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True
        )

        print(result.stdout)

        if result.stderr:
            print(f"⚠ Warnings/Errors:\n{result.stderr}")

        if result.returncode != 0:
            print(f"❌ Script failed with return code {result.returncode}")
            return False
        else:
            print(f"✓ {description} completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Error running {script_path.name}: {e}")
        return False

def main():
    base_dir = Path("D:/UAV_Project/paper_visualizations")

    scripts = [
        (base_dir / "figure_A_simple_comparison.py", "Figure A - Feature Map Comparison"),
        (base_dir / "figure_B_qualitative_comparison.py", "Figure B - Qualitative Comparison with Zoom"),
        (base_dir / "figure_C_error_analysis.py", "Figure C - Error Analysis (False Positives/Negatives)"),
        (base_dir / "figure_D_hyperparameter_analysis.py", "Figure D - Training Metrics & Confusion Matrix"),
    ]

    print("\n" + "="*80)
    print(" " * 20 + "SeaHunter Paper Visualization Generator")
    print("="*80)
    print(f"\nOutput directory: {base_dir}")
    print(f"Total scripts to run: {len(scripts)}\n")

    results = []
    for script_path, description in scripts:
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            results.append((description, False))
            continue

        success = run_script(script_path, description)
        results.append((description, success))

    print("\n" + "="*80)
    print(" " * 30 + "SUMMARY")
    print("="*80 + "\n")

    for description, success in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{status:12} | {description}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n{'='*80}")
    print(f"Completed: {successful}/{total} scripts ran successfully")
    print(f"{'='*80}\n")

    if successful == total:
        print("🎉 All visualizations generated successfully!")
        print(f"📁 Check the output in: {base_dir}")
    else:
        print("⚠ Some visualizations failed. Please check the errors above.")

if __name__ == "__main__":
    main()
