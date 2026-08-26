#!/usr/bin/env python3
"""Build Llama updated-persona ReFICR top-50 eval data with demographics.

For a target sample at session K+1, behavioral/dialogue persona updates are
taken from the persona-processor output for session K. The demographic
dialogue persona from each user's GT session 1 sample is prepended to all
later-session samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[4]
GT_PATH = SUPPLEMENTARY_ROOT / "data/evocrs/ranker/eval/reficr_candidates/reficr_ranking_test_top50_GT.json"
UPDATED_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/persona_processor/results/llama-3.1-8B/updated_personas_logs.llama.all.jsonl"
)
BASELINE_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/ranker/eval/reficr_candidates/llama-3.1-8b/"
    / "reficr_ranking_test_top50_inferenced_updated.json.error"
)
OUTPUT_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/ranker/eval/reficr_candidates/llama-3.1-8b/"
    / "inferenced_with_demographic.json"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_sample_id(sample_id: str) -> tuple[str, int]:
    user_id, session_text = sample_id.rsplit("_", 1)
    return user_id, int(session_text)


def load_updated_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["user_id"]), int(row["session_num"]))
            if key in records:
                raise ValueError(f"Duplicate updated persona for {key} at {path}:{line_num}")
            records[key] = row
    return records


def parse_json_string(value: Any, field_name: str, sample_id: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON string in {field_name} for {sample_id}") from exc


def persona_from_updated(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    sample_id = str(row.get("id", "unknown"))
    memory = parse_json_string(row.get("updated_memory", {}), "updated_memory", sample_id)
    if not isinstance(memory, dict):
        raise ValueError(f"updated_memory must be an object for {sample_id}")

    behavior = memory.get("behavior_persona", {})
    dialogue_persona = memory.get("dialogue_persona", [])
    if not isinstance(behavior, dict):
        raise ValueError(f"updated_memory.behavior_persona must be an object for {sample_id}")
    if not isinstance(dialogue_persona, list):
        raise ValueError(f"updated_memory.dialogue_persona must be a list for {sample_id}")
    return behavior, dialogue_persona


def build_session1_dialogue_map(gt_data: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for sample in gt_data:
        user_id, session_num = split_sample_id(str(sample["id"]))
        if session_num != 1:
            continue
        dialogue_persona = sample.get("dialogue_persona", [])
        if not isinstance(dialogue_persona, list):
            raise ValueError(f"dialogue_persona must be a list for {sample['id']}")
        result[user_id] = dialogue_persona
    return result


def main() -> None:
    gt_data = load_json(GT_PATH)
    baseline_ids = {str(sample["id"]) for sample in load_json(BASELINE_PATH)}
    updated_map = load_updated_map(UPDATED_PATH)
    session1_dialogue_map = build_session1_dialogue_map(gt_data)

    output: list[dict[str, Any]] = []
    missing: list[str] = []

    for sample in gt_data:
        sample_id = str(sample["id"])
        if sample_id not in baseline_ids:
            continue

        user_id, session_num = split_sample_id(sample_id)
        if session_num <= 1:
            continue

        prev_key = (user_id, session_num - 1)
        updated_row = updated_map.get(prev_key)
        if updated_row is None:
            missing.append(f"{sample['id']} needs updated persona {prev_key}")
            continue

        behavior, dialogue_persona = persona_from_updated(updated_row)
        item = dict(sample)
        item["behavioral_persona"] = behavior
        item["dialogue_persona"] = session1_dialogue_map.get(user_id, []) + dialogue_persona
        output.append(item)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(output)} samples -> {OUTPUT_PATH}")
    if missing:
        print(f"Skipped {len(missing)} samples without previous-session updated persona.")
        for item in missing[:10]:
            print(f"  - {item}")
        if len(missing) > 10:
            print(f"  ... {len(missing) - 10} more")


if __name__ == "__main__":
    main()
