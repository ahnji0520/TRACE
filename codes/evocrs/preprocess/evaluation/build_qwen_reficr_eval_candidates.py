#!/usr/bin/env python3
"""Build Qwen persona variants of ReFICR top-50 ranker eval files.

The generated files keep the GT eval template's target, dialogue history, and
top-50 candidate items unchanged. Only the persona fields are replaced with
Qwen persona-processor outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEMPLATE_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/ranker/eval/reficr_candidates/llama-3.1-8b/reficr_ranking_test_top50_inferenced_updated.json"
)
DEFAULT_INITIAL_PERSONA_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/ranker/eval/reficr_candidates/reficr_ranking_test_top50_GT.json"
)
DEFAULT_EXTRACTED_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/persona_processor/results/qwen3-8B/extracted_personas_only.qwen.jsonl"
)
DEFAULT_UPDATED_PATH = (
    SUPPLEMENTARY_ROOT
    / "data/evocrs/persona_processor/results/qwen3-8B/updated_personas_logs.qwen.final.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    SUPPLEMENTARY_ROOT / "data/evocrs/ranker/eval/reficr_candidates/qwen3-8b"
)


OUTPUT_NAMES = {
    "updated": "reficr_ranking_test_top50_inferenced_updated.json",
    "prev": "reficr_ranking_test_top50_prev_session_extracted.json",
    "session1": "reficr_ranking_test_top50_session1_extracted.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Qwen-persona ReFICR top-50 ranker eval JSON files."
    )
    parser.add_argument("--template-path", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument(
        "--initial-persona-path",
        type=Path,
        default=DEFAULT_INITIAL_PERSONA_PATH,
        help="JSON file containing session-1 initial dialogue persona per user.",
    )
    parser.add_argument("--extracted-path", type=Path, default=DEFAULT_EXTRACTED_PATH)
    parser.add_argument("--updated-path", type=Path, default=DEFAULT_UPDATED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Skip samples whose required Qwen persona record is missing. "
            "By default, missing records fail the run."
        ),
    )
    parser.add_argument(
        "--updated-missing",
        choices=["fallback-extracted", "error", "skip"],
        default="fallback-extracted",
        help=(
            "How to handle a missing previous-session updated-memory record. "
            "fallback-extracted keeps the sample and uses the previous-session "
            "extracted persona."
        ),
    )
    parser.add_argument(
        "--no-prepend-initial-dialogue",
        dest="prepend_initial_dialogue",
        action="store_false",
        help="Do not prepend GT session-1 initial dialogue persona.",
    )
    parser.set_defaults(prepend_initial_dialogue=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_string(value: Any, *, field_name: str, sample_id: str) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON string in {field_name} for {sample_id}") from exc


def load_jsonl_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            user_id = str(row.get("user_id", ""))
            session_num = int(row["session_num"])
            key = (user_id, session_num)

            if key in records:
                raise ValueError(f"Duplicate persona record for {key} in {path}:{line_num}")

            records[key] = row

    return records


def persona_from_extracted(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    sample_id = str(row.get("id", "unknown"))
    behavior = parse_json_string(
        row.get("extracted_behavior", {}),
        field_name="extracted_behavior",
        sample_id=sample_id,
    )
    dialogue = parse_json_string(
        row.get("extracted_dialogue", {}),
        field_name="extracted_dialogue",
        sample_id=sample_id,
    )

    if not isinstance(behavior, dict):
        raise ValueError(f"extracted_behavior must parse to an object for {sample_id}")
    if not isinstance(dialogue, dict):
        raise ValueError(f"extracted_dialogue must parse to an object for {sample_id}")

    dialogue_persona = dialogue.get("dialogue_persona", [])
    if not isinstance(dialogue_persona, list):
        raise ValueError(f"extracted_dialogue.dialogue_persona must be a list for {sample_id}")

    return behavior, dialogue_persona


def persona_from_updated(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    sample_id = str(row.get("id", "unknown"))
    memory = parse_json_string(
        row.get("updated_memory", {}),
        field_name="updated_memory",
        sample_id=sample_id,
    )

    if not isinstance(memory, dict):
        raise ValueError(f"updated_memory must parse to an object for {sample_id}")

    behavior = memory.get("behavior_persona", {})
    dialogue_persona = memory.get("dialogue_persona", [])

    if not isinstance(behavior, dict):
        raise ValueError(f"updated_memory.behavior_persona must be an object for {sample_id}")
    if not isinstance(dialogue_persona, list):
        raise ValueError(f"updated_memory.dialogue_persona must be a list for {sample_id}")

    return behavior, dialogue_persona


def with_persona(
    sample: dict[str, Any],
    behavior: dict[str, Any],
    dialogue_persona: list[str],
    initial_dialogue_persona: list[str] | None = None,
) -> dict[str, Any]:
    item = dict(sample)
    item["behavioral_persona"] = behavior
    item["dialogue_persona"] = (initial_dialogue_persona or []) + dialogue_persona
    return item


def split_sample_id(sample_id: str) -> tuple[str, int]:
    try:
        user_id, session_text = sample_id.rsplit("_", 1)
        return user_id, int(session_text)
    except ValueError as exc:
        raise ValueError(f"Invalid sample id, expected '<user_id>_<session>': {sample_id}") from exc


def require_record(
    records: dict[tuple[str, int], dict[str, Any]],
    key: tuple[str, int],
    source_name: str,
) -> dict[str, Any]:
    try:
        return records[key]
    except KeyError as exc:
        user_id, session_num = key
        raise KeyError(
            f"Missing {source_name} persona record for user={user_id}, session={session_num}"
        ) from exc


def build_initial_dialogue_map(
    data: list[dict[str, Any]],
) -> dict[str, list[str]]:
    initial_dialogue: dict[str, list[str]] = {}

    for sample in data:
        user_id, session_num = split_sample_id(str(sample["id"]))
        if session_num != 1:
            continue

        dialogue_persona = sample.get("dialogue_persona", [])
        if not isinstance(dialogue_persona, list):
            raise ValueError(f"dialogue_persona must be a list for {sample['id']}")
        initial_dialogue[user_id] = dialogue_persona

    return initial_dialogue


def append_or_skip(
    output: list[dict[str, Any]],
    skipped: list[str],
    sample: dict[str, Any],
    build_fn,
    allow_missing: bool,
) -> None:
    try:
        output.append(build_fn())
    except KeyError as exc:
        if not allow_missing:
            raise
        skipped.append(f"{sample['id']}: {exc}")


def build_outputs(
    template_data: list[dict[str, Any]],
    extracted_map: dict[tuple[str, int], dict[str, Any]],
    updated_map: dict[tuple[str, int], dict[str, Any]],
    initial_dialogue_map: dict[str, list[str]],
    allow_missing: bool,
    updated_missing: str,
    prepend_initial_dialogue: bool,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    outputs = {name: [] for name in OUTPUT_NAMES}
    skipped: list[str] = []

    for sample in template_data:
        user_id, session_num = split_sample_id(str(sample["id"]))
        if session_num <= 1:
            continue

        prev_key = (user_id, session_num - 1)
        session1_key = (user_id, 1)
        initial_dialogue = (
            initial_dialogue_map.get(user_id, []) if prepend_initial_dialogue else []
        )

        if prev_key in updated_map:
            outputs["updated"].append(
                with_persona(
                    sample,
                    *persona_from_updated(updated_map[prev_key]),
                    initial_dialogue_persona=initial_dialogue,
                )
            )
        elif updated_missing == "fallback-extracted":
            outputs["updated"].append(
                with_persona(
                    sample,
                    *persona_from_extracted(
                        require_record(extracted_map, prev_key, "extracted fallback")
                    ),
                    initial_dialogue_persona=initial_dialogue,
                )
            )
            skipped.append(
                f"{sample['id']}: missing updated {prev_key}; used previous extracted persona"
            )
        elif updated_missing == "skip" or allow_missing:
            skipped.append(f"{sample['id']}: missing updated {prev_key}; skipped updated")
        else:
            require_record(updated_map, prev_key, "updated")

        append_or_skip(
            outputs["prev"],
            skipped,
            sample,
            lambda: with_persona(
                sample,
                *persona_from_extracted(require_record(extracted_map, prev_key, "extracted")),
                initial_dialogue_persona=initial_dialogue,
            ),
            allow_missing,
        )
        append_or_skip(
            outputs["session1"],
            skipped,
            sample,
            lambda: with_persona(
                sample,
                *persona_from_extracted(require_record(extracted_map, session1_key, "extracted")),
                initial_dialogue_persona=initial_dialogue,
            ),
            allow_missing,
        )

    return outputs, skipped


def save_outputs(outputs: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in OUTPUT_NAMES.items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(outputs[key], f, ensure_ascii=False, indent=4)
        print(f"Saved {len(outputs[key])} samples -> {path}")


def main() -> None:
    args = parse_args()

    template_data = load_json(args.template_path)
    initial_persona_data = load_json(args.initial_persona_path)
    extracted_map = load_jsonl_map(args.extracted_path)
    updated_map = load_jsonl_map(args.updated_path)
    initial_dialogue_map = build_initial_dialogue_map(initial_persona_data)

    outputs, skipped = build_outputs(
        template_data=template_data,
        extracted_map=extracted_map,
        updated_map=updated_map,
        initial_dialogue_map=initial_dialogue_map,
        allow_missing=args.allow_missing,
        updated_missing=args.updated_missing,
        prepend_initial_dialogue=args.prepend_initial_dialogue,
    )
    save_outputs(outputs, args.output_dir)

    if skipped:
        print(f"Skipped {len(skipped)} missing persona assignments.")
        for item in skipped[:10]:
            print(f"  - {item}")
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
