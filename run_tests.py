"""
author: Jayesh Pandey
summary: Test runner script that discovers and executes all tests in the tests/ directory.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import unittest


def main() -> int:
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)

    failures = 0
    total = 0

    print("=== RUN TESTS ===")
    for modinfo in pkgutil.iter_modules(["tests"]):
        if not modinfo.name.startswith("test_"):
            continue
        module_name = f"tests.{modinfo.name}"
        module = importlib.import_module(module_name)
        suite = loader.loadTestsFromModule(module)
        if suite.countTestCases() == 0:
            continue
        total += 1
        print(f"\n--- {module_name} ---")
        result = runner.run(suite)
        if not result.wasSuccessful():
            failures += 1

    print("\n=== SUMMARY ===")
    if failures == 0:
        print(f"PASS ({total} modules)")
        return 0
    print(f"FAIL ({failures}/{total} modules failed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

