# -*- coding: utf-8 -*-
import os
import sys
import datetime
import numpy as np
from typing import List

from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from SolverVerification.patch_tests import PatchTestRunner, TestResult

def print_result_table(results: List[TestResult]):
    """Prints a formatted table of test results."""
    print("\n" + "="*110)
    print(f" {'[WHT] Solver Verification Results':^108}")
    print("="*110)
    header = f"{'Test Case':<25} | {'Element':<8} | {'Quantity':<20} | {'Theory':>12} | {'FEM':>12} | {'Error%':>8} | {'Result':<8}"
    print(header)
    print("-" * 110)

    passed_count = 0
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        if res.passed: passed_count += 1
        
        row = (f"{res.name:<25} | {res.element_type:<8} | {res.quantity:<20} | "
               f"{res.theory:12.4g} | {res.fem:12.4g} | {res.error_pct:8.2f}% | {status:<8}")
        print(row)
    
    print("-" * 110)
    print(f" Total: {len(results)} | Passed: {passed_count} | Failed: {len(results) - passed_count}")
    print("="*110 + "\n")

def save_result_to_md(results: List[TestResult]):
    """Saves test results to a Markdown file in the results directory."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"verification_report_{timestamp}.md"
    filepath = results_dir / filename
    
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# [WHT] Solver Verification Report\n\n")
        f.write(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Summary:** Total {len(results)} tests, {passed_count} PASSED, {failed_count} FAILED\n\n")
        
        f.write("| Test Case | Element | Quantity | Theory | FEM | Error% | Result |\n")
        f.write("| :--- | :--- | :--- | ---: | ---: | ---: | :---: |\n")
        
        for res in results:
            status = "✅ PASS" if res.passed else "❌ FAIL"
            f.write(f"| {res.name} | {res.element_type} | {res.quantity} | {res.theory:.4g} | {res.fem:.4g} | {res.error_pct:.2f}% | {status} |\n")
        
        f.write(f"\n---\n")
        f.write(f"**Final Result:** {'PASS' if failed_count == 0 else 'FAIL'}\n")

    print(f" -> [Success] Verification report saved to: {filepath}")

def main():
    runner = PatchTestRunner()
    all_results = []

    print(" -> Running Patch Tests for QUAD4...")
    all_results.extend(runner.test_3pt_bending(etype='QUAD4'))
    all_results.extend(runner.test_4pt_bending(etype='QUAD4'))
    all_results.extend(runner.test_twisting(etype='QUAD4'))
    all_results.extend(runner.test_frequency(etype='QUAD4'))
    all_results.extend(runner.test_membrane_uniaxial(etype='QUAD4'))

    print(" -> Running Patch Tests for TRIA3...")
    all_results.extend(runner.test_3pt_bending(etype='TRIA3'))
    all_results.extend(runner.test_4pt_bending(etype='TRIA3'))
    all_results.extend(runner.test_twisting(etype='TRIA3'))
    all_results.extend(runner.test_frequency(etype='TRIA3'))
    all_results.extend(runner.test_membrane_uniaxial(etype='TRIA3'))

    print_result_table(all_results)
    save_result_to_md(all_results)

    # Detailed debug for failed cases
    for res in all_results:
        if not res.passed and "Frequency" in res.name:
            print(f" -> [Debug] {res.name} ({res.element_type}) failed. Details: {res.details}")
            # The frequencies are from the result of the test run, but we don't have them here.
            # I'll update patch_tests.py to store them in 'details'.

    failed_tests = [r for r in all_results if not r.passed]
    if failed_tests:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
