#!/usr/bin/env python3
"""Rebuild persona-update resume shards from canonical unique rows.

`update.py --resume` restores state from `updated_personas_logs.shard*.jsonl`.
This helper first folds normal shard rows and successful retry rows into one
unique JSONL, then rewrites shard files using the same sorted-user assignment
that `update.py` uses.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output/llama-3.1-8B/upd"
DEFAULT_FILTER_DATA_PATH = (
    Path(__file__).resolve().parents[5]
    / "data/evocrs/persona_processor/filter_reficr.json"
)
SHARD_RE = re.compile(r"^updated_personas_logs\.shard(\d+)\.jsonl$")
RETRY_RE = re.compile(r"^updated_personas_logs\.[^.]+\.retry(\d+)\.jsonl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unique merged update JSONL and regenerate resume shard files."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filter-data-path", type=Path, default=DEFAULT_FILTER_DATA_PATH)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--merged-name",
        default="updated_personas_logs.llama.all.jsonl",
        help="Name of the canonical unique merged JSONL to write in output-dir.",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Do not create .bak files before rewriting existing outputs.",
    )
    parser.add_argument(
        "--from-merged",
        action="store_true",
        help="Use the canonical merged JSONL as the only source, then regenerate shards.",
    )
    parser.add_argument(
        "--cleanup-shards",
        action="store_true",
        help="Remove shard JSONL files after writing the merged JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without modifying files.",
    )
    parser.set_defaults(backup=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], backup: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sorted_shard_paths(output_dir: Path) -> list[Path]:
    paths = []
    for path in output_dir.glob("updated_personas_logs.shard*.jsonl"):
        match = SHARD_RE.match(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(paths)]


def sorted_retry_paths(output_dir: Path) -> list[Path]:
    paths = []
    for path in output_dir.glob("updated_personas_logs.*.retry*.jsonl"):
        match = RETRY_RE.match(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(paths)]


def row_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["user_id"]), int(row["session_num"]), str(row["id"]))


def load_shard_user_ids(filter_data_path: Path, num_shards: int) -> dict[str, int]:
    filter_data = json.loads(filter_data_path.read_text(encoding="utf-8"))
    eligible_user_ids = sorted(
        str(uid)
        for uid, sessions in filter_data.items()
        if sessions and max(int(session) for session in sessions) > 1
    )
    return {uid: idx % num_shards for idx, uid in enumerate(eligible_user_ids)}


def add_rows(
    rows_by_id: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    source_name: str,
    source_rank: int,
    source_by_id: dict[str, tuple[int, str]],
) -> tuple[int, int]:
    added = 0
    replaced = 0

    for row in rows:
        row_id = str(row["id"])
        previous = source_by_id.get(row_id)
        if previous is None:
            added += 1
        elif source_rank >= previous[0]:
            replaced += 1
        else:
            continue

        rows_by_id[row_id] = row
        source_by_id[row_id] = (source_rank, source_name)

    return added, replaced


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not args.filter_data_path.exists():
        raise SystemExit(f"Missing filter data: {args.filter_data_path}")

    rows_by_id: dict[str, dict[str, Any]] = {}
    source_by_id: dict[str, tuple[int, str]] = {}
    added_total = 0
    replaced_total = 0

    merged_path = args.output_dir / args.merged_name
    shard_paths = sorted_shard_paths(args.output_dir)
    retry_paths = [] if args.from_merged else sorted_retry_paths(args.output_dir)

    if args.from_merged:
        if not merged_path.exists() or merged_path.stat().st_size == 0:
            raise SystemExit(f"Missing canonical merged JSONL: {merged_path}")
        added, replaced = add_rows(
            rows_by_id,
            load_jsonl(merged_path),
            source_name=merged_path.name,
            source_rank=0,
            source_by_id=source_by_id,
        )
        added_total += added
        replaced_total += replaced
    else:
        if not shard_paths and not retry_paths:
            raise SystemExit(f"No update shard or retry JSONL files found in {args.output_dir}")

        for path in shard_paths:
            added, replaced = add_rows(
                rows_by_id,
                load_jsonl(path),
                source_name=path.name,
                source_rank=0,
                source_by_id=source_by_id,
            )
            added_total += added
            replaced_total += replaced

        for retry_rank, path in enumerate(retry_paths, start=1):
            added, replaced = add_rows(
                rows_by_id,
                load_jsonl(path),
                source_name=path.name,
                source_rank=retry_rank,
                source_by_id=source_by_id,
            )
            added_total += added
            replaced_total += replaced

    user_to_shard = load_shard_user_ids(args.filter_data_path, args.num_shards)
    shard_rows: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(args.num_shards)}
    skipped_rows = []

    merged_rows = sorted(rows_by_id.values(), key=row_sort_key)
    for row in merged_rows:
        user_id = str(row["user_id"])
        shard_idx = user_to_shard.get(user_id)
        if shard_idx is None:
            skipped_rows.append(row)
            continue
        shard_rows[shard_idx].append(row)

    if skipped_rows:
        sample_ids = ", ".join(str(row["id"]) for row in skipped_rows[:5])
        raise SystemExit(
            f"{len(skipped_rows)} rows could not be assigned to shards from filter data. "
            f"Sample ids: {sample_ids}"
        )

    if not args.dry_run:
        write_jsonl(merged_path, merged_rows, backup=args.backup)
        if not args.cleanup_shards:
            for shard_idx in range(args.num_shards):
                shard_path = args.output_dir / f"updated_personas_logs.shard{shard_idx}.jsonl"
                write_jsonl(shard_path, sorted(shard_rows[shard_idx], key=row_sort_key), backup=args.backup)

        for shard_path in shard_paths:
            match = SHARD_RE.match(shard_path.name)
            if match and (args.cleanup_shards or int(match.group(1)) >= args.num_shards):
                if args.backup:
                    shutil.copy2(shard_path, shard_path.with_suffix(shard_path.suffix + ".bak"))
                shard_path.unlink()

    print(f"[INFO] source mode: {'merged' if args.from_merged else 'shards+retries'}")
    print(f"[INFO] source shard files: {len(shard_paths)}")
    print(f"[INFO] source retry files: {len(retry_paths)}")
    print(f"[INFO] unique merged rows: {len(merged_rows)}")
    print(f"[INFO] rows added while folding: {added_total}")
    print(f"[INFO] duplicate rows replaced by later sources: {replaced_total}")
    print(f"[INFO] merged output: {merged_path}")
    for shard_idx in range(args.num_shards):
        print(
            f"[INFO] shard{shard_idx}: {len(shard_rows[shard_idx])} rows "
            f"-> {args.output_dir / f'updated_personas_logs.shard{shard_idx}.jsonl'}"
        )
    removable_shards = [
        path
        for path in shard_paths
        if SHARD_RE.match(path.name)
        and (args.cleanup_shards or int(SHARD_RE.match(path.name).group(1)) >= args.num_shards)
    ]
    if removable_shards:
        action = "would remove" if args.dry_run else "removed"
        label = "shard" if args.cleanup_shards else "stale shard"
        print(f"[INFO] {action} {label} files: {', '.join(path.name for path in removable_shards)}")
    if args.dry_run:
        print("[INFO] dry run only; no files were written")


if __name__ == "__main__":
    main()
