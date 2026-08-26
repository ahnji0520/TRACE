#!/usr/bin/env python3
"""Remove empty failed-input artifacts produced by persona processor runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAILED_PATTERNS = (
    'failed_*inputs*.json',
    '*.failed.json',
)


def is_empty_failure_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size == 0:
        return True
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    return data == []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    removed = []
    for root in args.roots:
        if not root.exists():
            continue
        candidates = []
        for pattern in FAILED_PATTERNS:
            candidates.extend(root.rglob(pattern))
        for path in sorted(set(candidates)):
            if is_empty_failure_file(path):
                removed.append(path)
                if not args.dry_run:
                    path.unlink()
    for path in removed:
        print(f'[CLEAN] removed empty failure file: {path}')
    print(f'[INFO] empty failure files removed: {len(removed)}')


if __name__ == '__main__':
    main()
