#!/usr/bin/env python3
"""Merge base extraction shards and retry outputs into a canonical final JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {exc}') from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_json_array(path: Path) -> list[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return json.loads(path.read_text(encoding='utf-8'))


def latest_failed_count(output_dir: Path, kind: str, model_family: str) -> tuple[Path | None, int]:
    retry_out_pat = re.compile(rf'^extracted_{re.escape(kind)}_only\.{re.escape(model_family)}\.retry(\d+)\.jsonl$')
    retry_rounds = []
    for path in output_dir.glob(f'extracted_{kind}_only.{model_family}.retry*.jsonl'):
        match = retry_out_pat.match(path.name)
        if match:
            retry_rounds.append(int(match.group(1)))

    failed_pat = re.compile(rf'^failed_{re.escape(kind)}_inputs\.{re.escape(model_family)}\.retry(\d+)\.json$')
    retry_failed = []
    for path in output_dir.glob(f'failed_{kind}_inputs.{model_family}.retry*.json'):
        match = failed_pat.match(path.name)
        if match:
            retry_failed.append((int(match.group(1)), path))

    if retry_rounds:
        latest_round = max(retry_rounds)
        latest_failed = [path for round_no, path in retry_failed if round_no == latest_round]
        if latest_failed:
            path = latest_failed[0]
            return path, len(load_json_array(path))
        return None, 0

    base_failed = []
    for path in output_dir.glob(f'failed_{kind}_inputs.shard*.json'):
        base_failed.extend(load_json_array(path))
    return None, len(base_failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=['behavior', 'dialogue'], required=True)
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--base-name', required=True)
    parser.add_argument('--final-name', required=True)
    parser.add_argument('--allow-pending-failures', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: dict[str, dict[str, Any]] = {}
    paths = []
    base_path = args.output_dir / args.base_name
    if base_path.exists():
        paths.append(base_path)

    pat = re.compile(rf'^extracted_{re.escape(args.kind)}_only\.{re.escape(args.model_family)}\.retry(\d+)\.jsonl$')
    retry_paths = []
    for path in args.output_dir.glob(f'extracted_{args.kind}_only.{args.model_family}.retry*.jsonl'):
        match = pat.match(path.name)
        if match:
            retry_paths.append((int(match.group(1)), path))
    paths.extend(path for _, path in sorted(retry_paths))

    if not paths:
        raise SystemExit(f'No extraction outputs found in {args.output_dir}')

    for path in paths:
        for row in load_jsonl(path):
            records[str(row['id'])] = row

    failed_path, pending = latest_failed_count(args.output_dir, args.kind, args.model_family)
    if pending and not args.allow_pending_failures:
        src = f' ({failed_path})' if failed_path else ''
        raise SystemExit(f'{pending} {args.kind} extraction failures remain{src}')

    rows = sorted(records.values(), key=lambda row: (str(row['user_id']), int(row['session_num']), str(row['id'])))
    final_path = args.output_dir / args.final_name
    write_jsonl(final_path, rows)
    print(f'[INFO] merged extraction files: {len(paths)}')
    print(f'[INFO] final rows: {len(rows)} -> {final_path}')
    print(f'[INFO] pending failures: {pending}')


if __name__ == '__main__':
    main()
