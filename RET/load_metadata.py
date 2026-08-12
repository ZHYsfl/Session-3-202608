"""Load paper metadata JSON files from a repository-relative directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import Metadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _numeric_json_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort PMID filenames numerically and other JSON names lexicographically."""
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def load_metadata(storage_path: str | Path) -> list[Metadata]:
    """Return all metadata records stored under a repository path.

    Args:
        storage_path: Directory path relative to the GitHub project root, for
            example ``"RET/data"``. An absolute path is also accepted for
            deployed environments.

    Returns:
        The complete metadata list, ordered by numeric PMID filename.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path does not point to a directory.
        ValueError: If a JSON object is malformed or its ``id`` does not match
            the JSON filename.
    """
    data_dir = Path(storage_path)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Metadata directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Metadata path is not a directory: {data_dir}")

    metadata_list: list[Metadata] = []
    for json_path in sorted(data_dir.glob("*.json"), key=_numeric_json_sort_key):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load metadata JSON: {json_path}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Metadata JSON must contain an object: {json_path}")

        metadata_id = payload.get("id")
        if str(metadata_id) != json_path.stem:
            raise ValueError(
                f"Metadata id does not match filename: {json_path.name} "
                f"(id={metadata_id!r})"
            )

        metadata_list.append(Metadata.from_dict(payload))

    return metadata_list


def load_metadata_from_dir(storage_path: str | Path) -> list[Metadata]:
    """Compatibility name matching ``RET/retrieval_v1.py``."""
    return load_metadata(storage_path)


__all__ = ["Metadata", "load_metadata", "load_metadata_from_dir"]
