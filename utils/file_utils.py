from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def scan_pdfs(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path, default):
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload) -> None:
    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def copy_file(src: str | Path, dst_dir: str | Path) -> Path:
    destination = ensure_directory(dst_dir) / Path(src).name
    shutil.copy2(src, destination)
    return destination


def reset_directory_contents(path: str | Path) -> None:
    target = ensure_directory(path)
    for entry in target.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def relative_paths(paths: Iterable[Path], root: str | Path) -> list[str]:
    base = Path(root)
    return [str(path.relative_to(base)) for path in paths]
