from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def require_file(path: str | Path, purpose: str) -> str:
    resolved = resolve_repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing {purpose}: {resolved}\n"
            f"Place the file there or edit the corresponding command/config path."
        )
    return str(resolved)


def require_dir(path: str | Path, purpose: str) -> str:
    resolved = resolve_repo_path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Missing {purpose}: {resolved}\n"
            f"Create/link the directory there or edit the corresponding command/config path."
        )
    return str(resolved)
