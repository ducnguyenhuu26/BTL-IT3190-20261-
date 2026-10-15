"""Byte-preserving historical code snapshots; prediction stages remain strict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_directory(root: Path, lock_path: Path) -> Path:
    return root / "provenance" / file_sha256(lock_path)[:12]


def _source(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Code snapshot paths must stay inside the project")
    return path


def archive_code(root: Path, lock_path: Path, references=()) -> Path:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock["code_hashes"]
    destination = snapshot_directory(root, lock_path)
    files = dict(expected)
    for name in references:
        files.setdefault(name, file_sha256(_source(root, name)))
    for name, digest in files.items():
        source = _source(root, name)
        if file_sha256(source) != digest:
            raise ValueError(f"Cannot archive changed historical source: {name}")
        target = _source(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != digest:
                raise ValueError(f"Existing code snapshot differs: {target}")
        else:
            shutil.copyfile(source, target)
    manifest = {"lock_sha256": file_sha256(lock_path), "code_hashes": expected,
                "reference_hashes": {name: digest for name, digest in files.items() if name not in expected},
                "purpose": "Original bytes for historical report audits and optimization regression benchmarks"}
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing snapshot manifest differs")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return destination


def verify_code(root: Path, lock_path: Path, expected: dict[str, str], *, allow_snapshot=False) -> str:
    if all(_source(root, name).is_file() and file_sha256(_source(root, name)) == digest
           for name, digest in expected.items()):
        return "working_tree"
    if not allow_snapshot:
        raise ValueError("Working code differs from the locked prediction implementation")
    directory = snapshot_directory(root, lock_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["lock_sha256"] != file_sha256(lock_path) or manifest["code_hashes"] != expected:
        raise ValueError("Historical code snapshot belongs to another lock")
    for name, digest in expected.items():
        if file_sha256(_source(directory, name)) != digest:
            raise ValueError(f"Historical code checksum differs: {name}")
    return "historical_snapshot"
