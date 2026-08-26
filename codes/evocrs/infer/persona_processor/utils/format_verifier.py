import json
from typing import Any, Dict, List, Tuple


"""
# NOTE: Schema Validator

# {
#     "positive_preference": {
#         "content_material": str,
#         "production_context": str,
#     },
#     "negative_preference": {
#         "content_material": str,
#         "production_context": str,
#     },
#     "usage_patterns": {
#         "temporal_patterns": str,
#         "immersion_style": str,
#     },
#     "decision_drivers": {
#         "selection_priority": str,
#         "motivations": str,
#     },
# }



# {
#     "dialogue_persona": [str, ...],
#     "memory": {
#         "episodic_memory": [str, ...],
#         "recommendation_outcomes": [
#             {
#                 "item": str,
#                 "reaction": str,
#                 "reason": str,
#             },
#             ...
#         ]
#     }
# }


# {
#   "updated_behavioral_persona": {...},
#   "updated_dialogue_persona": [...],
#   "updated_memory": {
#     "episodic_memory": [...],
#     "recommendation_history": [...]
#   }
# }

"""




def _expect_dict(obj: Any, path: str, errors: List[str]) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{path} must be a dict, got {type(obj).__name__}")
        return False
    return True


def _expect_list(obj: Any, path: str, errors: List[str]) -> bool:
    if not isinstance(obj, list):
        errors.append(f"{path} must be a list, got {type(obj).__name__}")
        return False
    return True


def _expect_str(obj: Any, path: str, errors: List[str]) -> bool:
    if not isinstance(obj, str):
        errors.append(f"{path} must be a string, got {type(obj).__name__}")
        return False
    return True


def _require_keys(obj: Dict[str, Any], required_keys: List[str], path: str, errors: List[str]) -> bool:
    ok = True
    for key in required_keys:
        if key not in obj:
            errors.append(f"{path}.{key} is missing")
            ok = False
    return ok


def _parse_json_string(value: Any, path: str, errors: List[str]) -> Any:
    if not isinstance(value, str):
        errors.append(f"{path} must be a JSON string, got {type(value).__name__}")
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        errors.append(f"{path} is not valid JSON string: {e}")
        return None


def _validate_behavior_schema(behavior: Any, errors: List[str], base_path: str = "extracted_behavior") -> bool:
    if not _expect_dict(behavior, base_path, errors):
        return False

    required = [
        "positive_preference",
        "negative_preference",
        "usage_patterns",
        "decision_drivers",
    ]
    _require_keys(behavior, required, base_path, errors)

    nested_schema = {
        "positive_preference": ["content_material", "production_context"],
        "negative_preference": ["content_material", "production_context"],
        "usage_patterns": ["temporal_patterns", "immersion_style"],
        "decision_drivers": ["selection_priority", "motivations"],
    }

    for section, fields in nested_schema.items():
        if section not in behavior:
            continue
        section_obj = behavior[section]
        section_path = f"{base_path}.{section}"

        if not _expect_dict(section_obj, section_path, errors):
            continue

        _require_keys(section_obj, fields, section_path, errors)

        for field in fields:
            if field in section_obj:
                _expect_str(section_obj[field], f"{section_path}.{field}", errors)

    return len(errors) == 0


def _validate_dialogue_schema(dialogue: Any, errors: List[str], base_path: str = "extracted_dialogue") -> bool:
    if not _expect_dict(dialogue, base_path, errors):
        return False

    required = ["dialogue_persona", "memory"]
    _require_keys(dialogue, required, base_path, errors)

    if "dialogue_persona" in dialogue:
        dialogue_persona = dialogue["dialogue_persona"]
        if _expect_list(dialogue_persona, f"{base_path}.dialogue_persona", errors):
            for i, item in enumerate(dialogue_persona):
                _expect_str(item, f"{base_path}.dialogue_persona[{i}]", errors)

    if "memory" in dialogue:
        memory = dialogue["memory"]
        memory_path = f"{base_path}.memory"

        if _expect_dict(memory, memory_path, errors):
            memory_required = ["episodic_memory", "recommendation_outcomes"]
            _require_keys(memory, memory_required, memory_path, errors)

            if "episodic_memory" in memory:
                episodic_memory = memory["episodic_memory"]
                if _expect_list(episodic_memory, f"{memory_path}.episodic_memory", errors):
                    for i, item in enumerate(episodic_memory):
                        _expect_str(item, f"{memory_path}.episodic_memory[{i}]", errors)

            if "recommendation_outcomes" in memory:
                outcomes = memory["recommendation_outcomes"]
                outcomes_path = f"{memory_path}.recommendation_outcomes"

                if _expect_list(outcomes, outcomes_path, errors):
                    for i, outcome in enumerate(outcomes):
                        outcome_path = f"{outcomes_path}[{i}]"

                        if not _expect_dict(outcome, outcome_path, errors):
                            continue

                        outcome_required = ["item", "reaction", "reason"]
                        _require_keys(outcome, outcome_required, outcome_path, errors)

                        for key in outcome_required:
                            if key in outcome:
                                _expect_str(outcome[key], f"{outcome_path}.{key}", errors)

    return len(errors) == 0






def validate_behavior_record(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    parsed: Dict[str, Any] = {}

    if not _expect_dict(record, "record", errors):
        return False, errors, parsed

    required = ["id", "user_id", "session_num", "extracted_behavior"]
    _require_keys(record, required, "record", errors)

    if "id" in record:
        _expect_str(record["id"], "record.id", errors)

    if "user_id" in record:
        _expect_str(record["user_id"], "record.user_id", errors)

    if "session_num" in record and not isinstance(record["session_num"], int):
        errors.append(f"record.session_num must be an int, got {type(record['session_num']).__name__}")

    behavior = None
    if "extracted_behavior" in record:
        behavior = _parse_json_string(record["extracted_behavior"], "record.extracted_behavior", errors)
        if behavior is not None:
            parsed["extracted_behavior"] = behavior
            _validate_behavior_schema(behavior, errors)

    return len(errors) == 0, errors, parsed


def validate_dialogue_record(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    parsed: Dict[str, Any] = {}

    if not _expect_dict(record, "record", errors):
        return False, errors, parsed

    required = ["id", "user_id", "session_num", "extracted_dialogue"]
    _require_keys(record, required, "record", errors)

    if "id" in record:
        _expect_str(record["id"], "record.id", errors)

    if "user_id" in record:
        _expect_str(record["user_id"], "record.user_id", errors)

    if "session_num" in record and not isinstance(record["session_num"], int):
        errors.append(f"record.session_num must be an int, got {type(record['session_num']).__name__}")

    dialogue = None
    if "extracted_dialogue" in record:
        dialogue = _parse_json_string(record["extracted_dialogue"], "record.extracted_dialogue", errors)
        if dialogue is not None:
            parsed["extracted_dialogue"] = dialogue
            _validate_dialogue_schema(dialogue, errors)

    return len(errors) == 0, errors, parsed


def validate_llm_record(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    parsed: Dict[str, Any] = {}

    if not _expect_dict(record, "record", errors):
        return False, errors, parsed

    required = ["id", "user_id", "session_num", "extracted_behavior", "extracted_dialogue"]
    _require_keys(record, required, "record", errors)

    if "id" in record:
        _expect_str(record["id"], "record.id", errors)

    if "user_id" in record:
        _expect_str(record["user_id"], "record.user_id", errors)

    if "session_num" in record and not isinstance(record["session_num"], int):
        errors.append(f"record.session_num must be an int, got {type(record['session_num']).__name__}")

    behavior = None
    dialogue = None

    if "extracted_behavior" in record:
        behavior = _parse_json_string(record["extracted_behavior"], "record.extracted_behavior", errors)
        if behavior is not None:
            parsed["extracted_behavior"] = behavior
            _validate_behavior_schema(behavior, errors)

    if "extracted_dialogue" in record:
        dialogue = _parse_json_string(record["extracted_dialogue"], "record.extracted_dialogue", errors)
        if dialogue is not None:
            parsed["extracted_dialogue"] = dialogue
            _validate_dialogue_schema(dialogue, errors)

    return len(errors) == 0, errors, parsed

def _canonicalize_updated_memory_payload(memory_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
    1) raw LLM output schema
       {
         "updated_behavioral_persona": ...,
         "updated_dialogue_persona": ...,
         "updated_memory": {
             "episodic_memory": ...,
             "recommendation_history": ...
         }
       }

    2) normalized stored schema
       {
         "behavior_persona": ...,
         "dialogue_persona": ...,
         "memory": {
             "session_events": ...,
             "recommendation_history": ...
         }
       }

    Return canonical schema:
       {
         "behavior_persona": ...,
         "dialogue_persona": ...,
         "memory": {
             "session_events": ...,
             "recommendation_history": ...
         }
       }
    """
    if not isinstance(memory_obj, dict):
        return memory_obj

    # Case 1: already normalized
    if "behavior_persona" in memory_obj and "dialogue_persona" in memory_obj and "memory" in memory_obj:
        memory = memory_obj.get("memory", {})
        if isinstance(memory, dict) and "episodic_memory" in memory and "session_events" not in memory:
            memory["session_events"] = memory.pop("episodic_memory")
        if isinstance(memory, dict) and "recommendation_outcomes" in memory and "recommendation_history" not in memory:
            memory["recommendation_history"] = memory.pop("recommendation_outcomes")
        return memory_obj

    # Case 2: raw LLM schema
    if (
        "updated_behavioral_persona" in memory_obj
        and "updated_dialogue_persona" in memory_obj
        and "updated_memory" in memory_obj
    ):
        raw_memory = memory_obj.get("updated_memory", {})
        if not isinstance(raw_memory, dict):
            raw_memory = {}

        session_events = raw_memory.get("session_events", raw_memory.get("episodic_memory", []))
        recommendation_history = raw_memory.get(
            "recommendation_history",
            raw_memory.get("recommendation_outcomes", [])
        )

        return {
            "behavior_persona": memory_obj.get("updated_behavioral_persona", {}),
            "dialogue_persona": memory_obj.get("updated_dialogue_persona", []),
            "memory": {
                "session_events": session_events,
                "recommendation_history": recommendation_history,
            },
        }

    return memory_obj


def _validate_updated_memory_schema(memory_obj: Any, errors: List[str], base_path: str = "updated_memory") -> bool:
    if not _expect_dict(memory_obj, base_path, errors):
        return False

    memory_obj = _canonicalize_updated_memory_payload(memory_obj)

    required = ["behavior_persona", "dialogue_persona", "memory"]
    _require_keys(memory_obj, required, base_path, errors)

    if "behavior_persona" in memory_obj:
        _validate_behavior_schema(
            memory_obj["behavior_persona"],
            errors,
            f"{base_path}.behavior_persona"
        )

    if "dialogue_persona" in memory_obj:
        dialogue_persona = memory_obj["dialogue_persona"]
        if _expect_list(dialogue_persona, f"{base_path}.dialogue_persona", errors):
            for i, item in enumerate(dialogue_persona):
                _expect_str(item, f"{base_path}.dialogue_persona[{i}]", errors)

    if "memory" in memory_obj:
        memory = memory_obj["memory"]
        memory_path = f"{base_path}.memory"

        if _expect_dict(memory, memory_path, errors):
            memory_required = ["session_events", "recommendation_history"]
            _require_keys(memory, memory_required, memory_path, errors)

            if "session_events" in memory:
                session_events = memory["session_events"]
                if _expect_list(session_events, f"{memory_path}.session_events", errors):
                    for i, item in enumerate(session_events):
                        _expect_str(item, f"{memory_path}.session_events[{i}]", errors)

            if "recommendation_history" in memory:
                recommendation_history = memory["recommendation_history"]
                history_path = f"{memory_path}.recommendation_history"

                if _expect_list(recommendation_history, history_path, errors):
                    for i, item in enumerate(recommendation_history):
                        item_path = f"{history_path}[{i}]"
                        if not _expect_dict(item, item_path, errors):
                            continue

                        required_keys = ["item", "reaction", "reason"]
                        _require_keys(item, required_keys, item_path, errors)

                        for key in required_keys:
                            if key in item:
                                _expect_str(item[key], f"{item_path}.{key}", errors)

    return len(errors) == 0


def validate_updated_memory_record(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    parsed: Dict[str, Any] = {}

    if not _expect_dict(record, "record", errors):
        return False, errors, parsed

    required = ["id", "user_id", "session_num", "updated_memory"]
    _require_keys(record, required, "record", errors)

    if "id" in record:
        _expect_str(record["id"], "record.id", errors)

    if "user_id" in record:
        _expect_str(record["user_id"], "record.user_id", errors)

    if "session_num" in record and not isinstance(record["session_num"], int):
        errors.append(f"record.session_num must be an int, got {type(record['session_num']).__name__}")

    if "updated_memory" in record:
        updated_memory = _parse_json_string(record["updated_memory"], "record.updated_memory", errors)
        if updated_memory is not None:
            canonical_memory = _canonicalize_updated_memory_payload(updated_memory)
            _validate_updated_memory_schema(canonical_memory, errors)

            if len(errors) == 0:
                parsed["updated_memory"] = canonical_memory

    return len(errors) == 0, errors, parsed