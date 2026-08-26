
#!/usr/bin/env python3
"""Fold successful persona-update retry outputs back into shard logs.

Retry generation writes valid rows to updated_personas_logs.llama.retryN.jsonl,
but resume mode reads updated_personas_logs.shard*.jsonl. This helper applies
the retry rows to the shard files and rebuilds the merged all.jsonl file.
"""

import argparse
import json
import re
from pathlib import Path


RETRY_RE = re.compile(r"^updated_personas_logs\.[^.]+\.retry(\d+)\.jsonl$")
SHARD_RE = re.compile(r"^updated_personas_logs\.shard(\d+)\.jsonl$")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_array(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return json.loads(path.read_text(encoding="utf-8"))


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


def latest_pending_failures(output_dir: Path) -> tuple[Path | None, int]:
    retry_failed = []
    for path in output_dir.glob("updated_personas_logs.*.retry*.failed.json"):
        stem = path.name.replace(".failed.json", ".jsonl")
        match = RETRY_RE.match(stem)
        if match:
            retry_failed.append((int(match.group(1)), path))

    if retry_failed:
        latest = sorted(retry_failed)[-1][1]
        return latest, len(load_json_array(latest))

    base_failed = []
    for path in output_dir.glob("failed_update_inputs.shard*.json"):
        base_failed.extend(load_json_array(path))
    return None, len(base_failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output/llama-3.1-8B/upd",
        help="Directory containing update shard logs and retry outputs.",
    )
    parser.add_argument(
        "--merged-name",
        default="updated_personas_logs.llama.all.jsonl",
        help="Merged JSONL filename to rebuild after applying retries.",
    )
    parser.add_argument(
        "--final-name",
        default=None,
        help="Optional final JSONL filename to write only when pending failures are zero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    shard_paths = sorted_shard_paths(output_dir)
    retry_paths = sorted_retry_paths(output_dir)
    if not shard_paths:
        raise SystemExit(f"No shard logs found in {output_dir}")

    shard_rows: dict[Path, list[dict]] = {}
    id_to_location: dict[str, tuple[Path, int]] = {}
    user_to_shard: dict[str, Path] = {}

    for shard_path in shard_paths:
        rows = load_jsonl(shard_path)
        shard_rows[shard_path] = rows
        for idx, row in enumerate(rows):
            row_id = row["id"]
            user_id = str(row["user_id"])
            if row_id in id_to_location:
                raise SystemExit(f"Duplicate id already exists in shard logs: {row_id}")
            id_to_location[row_id] = (shard_path, idx)
            user_to_shard.setdefault(user_id, shard_path)

    applied = 0
    replaced = 0
    appended = 0

    for retry_path in retry_paths:
        for row in load_jsonl(retry_path):
            row_id = row["id"]
            user_id = str(row["user_id"])
            if row_id in id_to_location:
                shard_path, idx = id_to_location[row_id]
                shard_rows[shard_path][idx] = row
                replaced += 1
            else:
                shard_path = user_to_shard.get(user_id)
                if shard_path is None:
                    raise SystemExit(
                        f"Cannot assign retry row {row_id}: user {user_id} was not found in shard logs"
                    )
                idx = len(shard_rows[shard_path])
                shard_rows[shard_path].append(row)
                id_to_location[row_id] = (shard_path, idx)
                appended += 1
            user_to_shard[user_id] = shard_path
            applied += 1

    merged_rows = []
    for shard_path in shard_paths:
        rows = shard_rows[shard_path]
        merged_rows.extend(rows)
        if not args.dry_run:
            write_jsonl(shard_path, rows)

    merged_ids = [row["id"] for row in merged_rows]
    if len(merged_ids) != len(set(merged_ids)):
        raise SystemExit("Duplicate ids detected after applying retries")

    merged_path = output_dir / args.merged_name
    if not args.dry_run:
        write_jsonl(merged_path, merged_rows)

    failed_path, pending_failed = latest_pending_failures(output_dir)
    if pending_failed == 0 and args.final_name:
        final_path = output_dir / args.final_name
        if not args.dry_run:
            write_jsonl(final_path, merged_rows)
    else:
        final_path = None

    print(f"[INFO] shard files: {len(shard_paths)}")
    print(f"[INFO] retry files: {len(retry_paths)}")
    print(f"[INFO] retry rows applied: {applied} (replaced={replaced}, appended={appended})")
    print(f"[INFO] merged rows: {len(merged_rows)}")
    print(f"[INFO] merged unique ids: {len(set(merged_ids))}")
    if failed_path is not None:
        print(f"[INFO] latest retry failures: {pending_failed} ({failed_path})")
    else:
        print(f"[INFO] base shard failures: {pending_failed}")
    if final_path is not None:
        print(f"[INFO] wrote clean final output: {final_path}")
    elif args.final_name:
        print("[INFO] final output not written because pending failures remain")


if __name__ == "__main__":
    main()
