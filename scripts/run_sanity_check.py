"""Sanity check script for DriveSense-VLM project structure and imports.

Verifies:
  1. All drivesense modules import without error
  2. All YAML configs parse without error
  3. Directory structure matches the expected layout
  4. Stub functions correctly raise NotImplementedError

Usage:
    python scripts/run_sanity_check.py

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import yaml

# ── Project root detection ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Add src/ to sys.path so drivesense is importable without pip install
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ── Checks registry ───────────────────────────────────────────────────────────

CheckResult = tuple[str, bool, str]  # (label, passed, detail)
results: list[CheckResult] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    """Record a check result."""
    results.append((label, passed, detail))


# ── 1. Module Import Checks ────────────────────────────────────────────────────

MODULES_TO_CHECK = [
    "drivesense",
    "drivesense.data",
    "drivesense.data.nuscenes_loader",
    "drivesense.data.dada_loader",
    "drivesense.data.annotation",
    "drivesense.data.dataset",
    "drivesense.data.transforms",
    "drivesense.training",
    "drivesense.training.sft_trainer",
    "drivesense.training.callbacks",
    "drivesense.inference",
    "drivesense.inference.merge_lora",
    "drivesense.inference.quantize",
    "drivesense.inference.tensorrt_vit",
    "drivesense.inference.serve",
    "drivesense.eval",
    "drivesense.eval.grounding",
    "drivesense.eval.reasoning",
    "drivesense.eval.production",
    "drivesense.eval.robustness",
    "drivesense.utils",
    "drivesense.utils.config",
    "drivesense.utils.logging",
    "drivesense.utils.visualization",
]

for module_name in MODULES_TO_CHECK:
    try:
        importlib.import_module(module_name)
        check(f"import {module_name}", True)
    except ImportError as exc:
        check(f"import {module_name}", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        check(f"import {module_name}", False, f"Unexpected: {exc}")


# ── 2. Config YAML Checks ──────────────────────────────────────────────────────

CONFIG_FILES = [
    "configs/model.yaml",
    "configs/data.yaml",
    "configs/training.yaml",
    "configs/inference.yaml",
    "configs/eval.yaml",
]

for config_rel in CONFIG_FILES:
    config_path = PROJECT_ROOT / config_rel
    try:
        with config_path.open() as f:
            content = yaml.safe_load(f)
        assert isinstance(content, dict), "Parsed YAML is not a dict"
        check(f"parse {config_rel}", True)
    except FileNotFoundError:
        check(f"parse {config_rel}", False, "File not found")
    except yaml.YAMLError as exc:
        check(f"parse {config_rel}", False, str(exc))
    except AssertionError as exc:
        check(f"parse {config_rel}", False, str(exc))


# ── 3. Directory Structure Checks ─────────────────────────────────────────────

EXPECTED_PATHS = [
    "configs/",
    "src/drivesense/",
    "src/drivesense/data/",
    "src/drivesense/training/",
    "src/drivesense/inference/",
    "src/drivesense/eval/",
    "src/drivesense/utils/",
    "scripts/",
    "slurm/",
    "notebooks/",
    "demo/",
    "tests/",
    "README.md",
    "CLAUDE.md",
    "pyproject.toml",
    ".gitignore",
    "LICENSE",
]

for rel_path in EXPECTED_PATHS:
    full_path = PROJECT_ROOT / rel_path
    exists = full_path.exists()
    check(f"exists {rel_path}", exists, "" if exists else "Missing")


# ── 4. Stub NotImplementedError Checks ────────────────────────────────────────

# (module, callable_name) pairs that must raise NotImplementedError
STUB_CHECKS: list[tuple[str, str]] = [
    ("drivesense.data.nuscenes_loader", "NuScenesRarityFilter"),
    ("drivesense.data.dada_loader", "DADALoader"),
    ("drivesense.data.annotation", "AnnotationPipeline"),
    ("drivesense.data.dataset", "DriveSenseDataset"),
    ("drivesense.training.sft_trainer", "train"),
    ("drivesense.eval.grounding", "compute_iou"),
    ("drivesense.inference.serve", "DriveSenseServer"),
    # Note: drivesense.utils.config.load_config is fully implemented (not a stub)
]

for module_name, callable_name in STUB_CHECKS:
    try:
        mod = importlib.import_module(module_name)
        obj = getattr(mod, callable_name, None)
        if obj is None:
            check(f"stub {module_name}.{callable_name}", False, "Not found in module")
            continue

        # Determine if it's a class (check __init__) or a function
        if inspect.isclass(obj):
            try:
                obj()  # Attempt instantiation with no args
                check(f"stub {module_name}.{callable_name}", False, "__init__ did not raise")
            except NotImplementedError:
                check(f"stub {module_name}.{callable_name}", True)
            except TypeError:
                # Required positional args — check __init__ source for NotImplementedError
                src = inspect.getsource(obj.__init__)
                if "NotImplementedError" in src:
                    check(f"stub {module_name}.{callable_name}", True)
                else:
                    check(f"stub {module_name}.{callable_name}", False, "__init__ has no NotImplementedError")
        elif callable(obj):
            try:
                obj()  # type: ignore[call-arg]
                check(f"stub {module_name}.{callable_name}", False, "Did not raise NotImplementedError")
            except NotImplementedError:
                check(f"stub {module_name}.{callable_name}", True)
            except TypeError:
                # Required args — check source
                src = inspect.getsource(obj)
                if "NotImplementedError" in src:
                    check(f"stub {module_name}.{callable_name}", True)
                else:
                    check(f"stub {module_name}.{callable_name}", False, "No NotImplementedError in source")
        else:
            check(f"stub {module_name}.{callable_name}", False, "Not callable")

    except ImportError as exc:
        check(f"stub {module_name}.{callable_name}", False, f"Import failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        check(f"stub {module_name}.{callable_name}", False, f"Unexpected: {exc}")


# ── 5. Package Version Check ───────────────────────────────────────────────────
try:
    import drivesense
    version = drivesense.__version__
    check("drivesense.__version__", isinstance(version, str) and len(version) > 0, version)
except Exception as exc:  # noqa: BLE001
    check("drivesense.__version__", False, str(exc))


# ── Print Results Table ────────────────────────────────────────────────────────

print()
print("=" * 70)
print(f"  DriveSense-VLM Sanity Check — Project Root: {PROJECT_ROOT.name}")
print("=" * 70)
print(f"  {'Status':<6}  {'Check'}")
print(f"  {'------':<6}  {'-----'}")

failed = 0
for label, passed, detail in results:
    icon = "✓" if passed else "✗"
    suffix = f"  [{detail}]" if detail and not passed else ""
    print(f"  {icon:<6}  {label}{suffix}")
    if not passed:
        failed += 1

print("=" * 70)
total = len(results)
passed_count = total - failed
print(f"  Results: {passed_count}/{total} passed", end="")
if failed == 0:
    print("  — All checks passed!")
else:
    print(f"  — {failed} check(s) failed")
print("=" * 70)
print()

sys.exit(0 if failed == 0 else 1)
