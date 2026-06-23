#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAMES = {
    "app.py",
    "backend.py",
}

FORBIDDEN_DIR_PARTS = {
    "__pycache__",
    "backend",
    "app",
}

FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
}

ALLOWED_OUTPUT_FILES = {
    REPO_ROOT / "outputs" / ".gitkeep",
}


def is_forbidden(path: Path) -> str | None:
    rel = path.relative_to(REPO_ROOT)
    parts = set(rel.parts)
    if path.name in FORBIDDEN_NAMES:
        return "legacy app/backend entry point"
    if parts & FORBIDDEN_DIR_PARTS:
        return "legacy app/backend directory or Python cache"
    if path.suffix in FORBIDDEN_SUFFIXES:
        return "Python bytecode"
    if rel.parts and rel.parts[0] == "outputs" and path.is_file() and path not in ALLOWED_OUTPUT_FILES:
        return "generated output artifact"
    return None


def main() -> int:
    violations: list[tuple[Path, str]] = []
    for path in REPO_ROOT.rglob("*"):
        reason = is_forbidden(path)
        if reason:
            violations.append((path, reason))

    if violations:
        print("Repository hygiene check failed:")
        for path, reason in violations[:100]:
            print(f"  [{reason}] {path.relative_to(REPO_ROOT)}")
        if len(violations) > 100:
            print(f"  ... {len(violations) - 100} more")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
