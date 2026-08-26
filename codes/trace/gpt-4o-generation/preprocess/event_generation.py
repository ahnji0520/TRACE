import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from dataset_generation.prompt.event_generation_prompt import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

DEBUG = False


# 1. Implementation note
SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
else:
    print(f"Warning: .env file not found: {ENV_PATH}")
    api_key = None

# 2. Implementation note
client = OpenAI(api_key=api_key)

def get_llm_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return {}


def get_llm_responses_parallel(prompts: List[str], max_workers: int=10) -> List[dict]:
    results = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(get_llm_response, prompt): idx for idx, prompt in enumerate(prompts)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"❌ Error in batch call {idx}: {e}")
                results[idx] = {}
    return results


def load_existing_results(output_path):
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def format_prompt(template, user_profile_str:str, watch_log_str:str, existing_events=None, last_event=None, candidates=None):
    template = template.replace("[[$User_Profile$]]", user_profile_str)
    template = template.replace("[[$Watch_Log_Item$]]", watch_log_str)

    if existing_events is not None:
        if not isinstance(existing_events, str):
            existing_events = json.dumps(existing_events)
        template = template.replace("[[$Existing_Events$]]", existing_events)                  
    if last_event is not None:
        if not isinstance(last_event, str):
            last_event = json.dumps(last_event)
        template = template.replace("[[$Last_Event$]]", last_event)
    if candidates is not None:
        if not isinstance(candidates, str):
            candidates = json.dumps(candidates)
        template = template.replace("[[$Candidates$]]", candidates)


    if "[[$" in template:
        print(template)
        raise ValueError("Template still contains unreplaced variables")

    return template



def collect_session_level_targets(user_sessions: dict, personas: dict):
    """
    user_id -> [ (session_key, item_data) ]  # one target per session, ordered by session_key
    """
    user_plan = {}

    for user_id, sessions in user_sessions.items():
        
        profile_info = personas.get(user_id, {}).get("static", {})
        user_profile_str = json.dumps({
            "user_age": profile_info.get("user_age"),
            "num_kids": 1 if profile_info.get("has_kids") else 0,
            "relationship_status": profile_info.get("relationship_status"),
            "occupation": profile_info.get("user_occupation")
        }, indent=2, ensure_ascii=False)

        sorted_session_keys = sorted(sessions.keys(), key=lambda x: int(x))

        per_user = []
        for s_key in sorted_session_keys:
            session = sessions[s_key]
            watch_list = session.get("watch", [])

            # Implementation note.
            chosen = None
            for item in reversed(watch_list):
                if item.get("target"):
                    chosen = item
                    break

            if chosen is None:
                continue

            watch_log_str = json.dumps(chosen, indent=2, ensure_ascii=False)
            per_user.append((s_key, {
                "user_id": user_id,
                "sessions_key": s_key,
                "item": chosen,
                "user_profile_str": user_profile_str,
                "watch_log_str": watch_log_str,
            }))

        if per_user:
            user_plan[user_id] = per_user

    return user_plan

        
def generate_events_round_robin(max_workers):
    output_path = os.path.join("data", "events.json")

    data_dir = "data"
    persona_path = os.path.join(data_dir, "dialogue_persona_init_dict.json")
    session_data_path = os.path.join(data_dir, "behavioral_log_sessions.json")
    output_path = os.path.join(data_dir, "events.json")

    
    # Load data
    with open(persona_path, 'r', encoding='utf-8') as f:
        personas = json.load(f)

    # Implementation note.
    with open(session_data_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('NaN', 'null')
        user_sessions = json.loads(content)


    existing_results = load_existing_results(output_path)
    print(f"📂 Loaded {len(existing_results)} existing user results")

    # DEBUG Mode
    if DEBUG:
        user_sessions = dict(list(user_sessions.items())[:10])


    user_plan = collect_session_level_targets(user_sessions, personas)
    user_existing_events = {uid: [] for uid in user_plan.keys()}

    # Implementation note.
    for user_id, seq in user_plan.items():
        saved = existing_results.get(user_id, {})
        for session_key, item_data in seq:
            if session_key in saved:
                # Implementation note.
                item_data["item"]["event"] = saved[session_key]
                # Implementation note.
                user_existing_events[user_id].append(saved[session_key])
            else:
                break

    # Implementation note.
    max_rounds = max((len(v) for v in user_plan.values()), default=0)

    for round_idx in range(max_rounds):
        batch = []
        for user_id, seq in user_plan.items():
            if round_idx >= len(seq):
                continue

            session_key, item_data = seq[round_idx]
            if item_data["item"].get("event") is not None:
                continue  # Implementation note.
            batch.append((user_id, session_key, item_data))

        if not batch:
            continue

        is_init_round = (round_idx == 0)

        # Implementation note.
        if is_init_round:
            template_gen = (INITIAL_EVENT_GENERATION_INTRO +
                            EVENT_GENERATION_RULES +
                            INITIAL_EVENT_GENERATION_TASK +
                            INITIAL_EVENT_GENERATION_OUTPUT +
                            INITIAL_EVENT_GENERATION_INPUTS)

            template_sel = (INITIAL_EVENT_SELECTION_TASK +
                            INITIAL_EVENT_GENERATION_INPUTS +
                            EVENT_SELECTION_INPUT)
        else:
            template_gen = (SUB_EVENT_GENERATION_INTRO +
                            EVENT_GENERATION_RULES +
                            SUB_EVENT_GENERATION_TASK +
                            SUB_EVENT_GENERATION_OUTPUT +
                            SUB_EVENT_GENERATION_INPUTS)

            template_sel = (SUB_EVENT_SELECTION_TASK +
                            SUB_EVENT_GENERATION_INPUTS +
                            EVENT_SELECTION_INPUT)

        # 1) Implementation note
        gen_prompts = []
        gen_meta = []  # (user_id, item_data)
        for user_id, session_key, item_data in batch:
            existing_events = user_existing_events[user_id]
            last_event = existing_events[-1] if existing_events else None

            if is_init_round:
                prompt = format_prompt(
                    template_gen,
                    user_profile_str=item_data["user_profile_str"],
                    watch_log_str=item_data["watch_log_str"]
                )
            else:
                prompt = format_prompt(
                    template_gen,
                    user_profile_str=item_data["user_profile_str"],
                    watch_log_str=item_data["watch_log_str"],
                    existing_events=existing_events,
                    last_event=last_event
                )

            gen_prompts.append(prompt)
            gen_meta.append((user_id, item_data))

        gen_results = get_llm_responses_parallel(gen_prompts, max_workers)

        # 2) Selection prompts
        sel_prompts = []
        sel_meta = []  # (user_id, item_data)
        for i, (user_id, item_data) in enumerate(gen_meta):
            existing_events = user_existing_events[user_id]
            last_event = existing_events[-1] if existing_events else None

            if is_init_round:
                prompt = format_prompt(
                    template_sel,
                    user_profile_str=item_data["user_profile_str"],
                    watch_log_str=item_data["watch_log_str"],
                    candidates=gen_results[i]
                )
            else:
                prompt = format_prompt(
                    template_sel,
                    user_profile_str=item_data["user_profile_str"],
                    watch_log_str=item_data["watch_log_str"],
                    existing_events=existing_events,
                    last_event=last_event,
                    candidates=gen_results[i]
                )

            sel_prompts.append(prompt)
            sel_meta.append((user_id, item_data))

        sel_results = get_llm_responses_parallel(sel_prompts, max_workers)

        # 3) Implementation note
        for i, (user_id, item_data) in enumerate(sel_meta):
            selected_event = sel_results[i].get("refined_event", "No event selected.")
            item_data["item"]["event"] = selected_event
            user_existing_events[user_id].append(selected_event)

        print(f"✅ Session Round {round_idx+1}/{max_rounds} complete: {len(batch)} users processed")

    # Implementation note.
    final_results = existing_results.copy()
    for user_id, seq in user_plan.items():
        final_results.setdefault(user_id, {})
        for session_key, item_data in seq:
            event = item_data["item"].get("event")
            if event is not None:
                final_results[user_id][session_key] = event

    save_results(final_results, output_path)
    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    generate_events_round_robin(max_workers=50)