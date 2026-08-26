import os
import json
import time
import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
from diskcache import Cache
from openai import OpenAI

# Import existing prompts
from dataset_generation.prompt.behavioral_persona_extract_prompt import (
    CONTENT_MATERIAL_EXTRACT,
    PRODUCTION_CONTEXT_EXTRACT,
    TEMPORAL_PATTERNS_EXTRACT,
    IMMERSION_STYLE_EXTRACT,
    SELECTION_PRIORITY_EXTRACT,
    MOTIVATIONS_EXTRACT
)
from dataset_generation.prompt.behavioral_persona_update_prompt import (
    CONTENT_MATERIAL_UPDATE,
    PRODUCTION_CONTEXT_UPDATE,
    TEMPORAL_PATTERNS_UPDATE,
    IMMERSION_STYLE_UPDATE,
    SELECTION_PRIORITY_UPDATE,
    MOTIVATIONS_UPDATE
)

# -----------------------
# Setup
# -----------------------
SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
load_dotenv(ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cache = Cache(".cache")

# Global LLM concurrency limit; controls actual concurrent calls separately from worker count
MAX_INFLIGHT_LLM = int(os.getenv("MAX_INFLIGHT_LLM", "40"))
llm_sema = threading.Semaphore(MAX_INFLIGHT_LLM)


# -----------------------
# File Utils (atomic save, jsonl)
# -----------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json_atomic(path: str, obj):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic

def load_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_jsonl(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_path(run_dir, uid, step):
    return os.path.join(run_dir, "extracts", str(uid), f"{step:06d}.json")

def checkpoint_path(run_dir, uid):
    return os.path.join(run_dir, "checkpoints", f"{uid}.json")

def history_path(run_dir, uid):
    return os.path.join(run_dir, "history", f"{uid}.jsonl")


# -----------------------
# LLM util (cache + retry)
# -----------------------
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_llm_json_with_cache(prompt: str):
    if prompt in cache:
        return cache[prompt]

    with llm_sema:
        resp = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )

    result = json.loads(resp.choices[0].message.content)
    cache[prompt] = result

    # print(resp)
    # if hasattr(resp, "usage") and hasattr(resp.usage, "cached_tokens"):
    #     print(f"✅ Cached tokens used: {resp.usage.cached_tokens}")
    return result


# -----------------------
# Stats update (cheap)
# -----------------------
def calculate_updated_stats(old_stats, session_data):
    watch_records = session_data.get("watch", [])
    new_pcts = [r["watched_pct"] for r in watch_records]
    new_pcts = [0.0 if pct is None else float(pct) for pct in new_pcts]

    old_mean = float(old_stats.get("watched_pct_avg", 0))
    old_var = float(old_stats.get("watched_pct_var", 0))
    old_n = int(old_stats.get("total_count", 0))

    new_mean = float(np.mean(new_pcts))
    new_var = float(np.var(new_pcts))

    new_n = len(new_pcts)
    total_n = old_n + new_n
    if total_n == 0:
        return old_stats

    # Combined mean
    total_mean = (old_mean * old_n + new_mean * new_n) / total_n
    # Combined variance
    m2_total = (old_n* old_var) + (new_n * new_var)
    m2_total += old_n * (old_mean - total_mean) ** 2
    m2_total += new_n * (new_mean - total_mean) ** 2

    total_var = m2_total / total_n if total_n > 0 else 0

    weekday_freq = old_stats.get("weekday_freq", {}).copy()
    for record in watch_records:
        day = record.get("watched_weekday")
        if day:
            weekday_freq[day] = weekday_freq.get(day, 0) + 1

    return {
        "watched_pct_avg": round(float(total_mean), 2),
        "watched_pct_var": round(float(total_var), 2),
        "total_count": total_n,
        "weekday_freq": weekday_freq
    }


# -----------------------
# Phase split meta
# -----------------------
@dataclass(frozen=True)
class SessionMeta:
    user_id: str
    step: int
    session_key: str
    stats_limited_str: str
    stats_full_str: str
    window_content_logs_json: str
    window_production_logs_json: str
    window_union_logs_json: str
    window_usage_logs_json: str
    updated_stats: dict


def build_session_metas_for_user(user_id: str, sessions: dict, init_persona: dict, context_k: int = 2):
    """
    Build extract inputs without LLM calls while preserving user session order and stats/context windows.
    Create per-session EXTRACT input metadata.
    """
    sorted_keys = sorted(sessions.keys(), key=lambda x: int(x))
    metas = []

    current_stats = copy.deepcopy(init_persona.get("historical_stats", {
        "watched_pct_avg": 0, "watched_pct_var": 0, "total_count": 0, "weekday_freq": {}
    }))

    context_window = []
    for step, s_key in enumerate(sorted_keys):
        session = sessions[s_key]

        # stats update
        current_stats = calculate_updated_stats(current_stats, session)
        

        stats_limited_str = json.dumps({
            "watched_pct_avg": current_stats["watched_pct_avg"],
            "watched_pct_var": current_stats["watched_pct_var"]
        }, ensure_ascii=False)
        stats_full_str = json.dumps(current_stats, ensure_ascii=False)

        # context window update
        context_window.append(session)
        if len(context_window) > context_k:
            context_window.pop(0)

        # window logs build
        window_content_logs, window_production_logs, window_union_logs = [], [], []
        window_usage_logs = {}

        for s in context_window:
            date = s["date"]
            window_usage_logs[date] = {
                "day": s["weekday"],
                "watched_pct": [r["watched_pct"] for r in s["watch"]]
            }
            for r in s["watch"]:
                window_content_logs.append({k: r.get(k) for k in [
                    "watched_pct", "content_type", "title", "genres", "description", "keywords"
                ]})
                window_production_logs.append({k: r.get(k) for k in [
                    "watched_pct", "content_type", "title", "release_year", "countries", "actors", "directors"
                ]})
                window_union_logs.append({k: r.get(k) for k in [
                    "watched_pct", "content_type", "title", "genres", "description", "keywords",
                    "release_year", "countries", "actors", "directors"
                ]})
        
        metas.append(SessionMeta(
            user_id=user_id,
            step=step,
            session_key=s_key,
            stats_limited_str=stats_limited_str,
            stats_full_str=stats_full_str,
            window_content_logs_json=json.dumps(window_content_logs, ensure_ascii=False),
            window_production_logs_json=json.dumps(window_production_logs, ensure_ascii=False),
            window_union_logs_json=json.dumps(window_union_logs, ensure_ascii=False),
            window_usage_logs_json=json.dumps(window_usage_logs, ensure_ascii=False),
            updated_stats=copy.deepcopy(current_stats),
        ))

    return metas


# -----------------------
# Phase A: EXTRACT
# -----------------------
def extract_worker(meta: SessionMeta):
    
    ext_content = get_llm_json_with_cache(
        CONTENT_MATERIAL_EXTRACT.format(
            user_completion_stats=meta.stats_limited_str,
            log_data=meta.window_content_logs_json
        )
    )
    ext_prod = get_llm_json_with_cache(
        PRODUCTION_CONTEXT_EXTRACT.format(
            user_completion_stats=meta.stats_limited_str,
            log_data=meta.window_production_logs_json
        )
    )
    ext_temp = get_llm_json_with_cache(
        TEMPORAL_PATTERNS_EXTRACT.format(
            user_stats=meta.stats_full_str,
            log_data=meta.window_usage_logs_json
        )
    )
    ext_immer = get_llm_json_with_cache(
        IMMERSION_STYLE_EXTRACT.format(
            user_stats=meta.stats_full_str,
            log_data=meta.window_usage_logs_json
        )
    )
    
    merged_ext_pref = {**ext_content, **ext_prod}
    
    ext_priority = get_llm_json_with_cache(
        SELECTION_PRIORITY_EXTRACT.format(
            preference=json.dumps(merged_ext_pref, ensure_ascii=False),
            user_completion_stats=meta.stats_limited_str,
            log_data=meta.window_union_logs_json
        )
    )
    ext_motives = get_llm_json_with_cache(
        MOTIVATIONS_EXTRACT.format(
            preference=json.dumps(merged_ext_pref, ensure_ascii=False),
            user_completion_stats=meta.stats_limited_str,
            log_data=meta.window_union_logs_json
        )
    )

    extracted_formatted = {
        "historical_stats": meta.updated_stats,
        "positive_preference": {
            "content_material": ext_content.get("positive_content_material"),
            "production_context": ext_prod.get("positive_production_context")
        },
        "negative_preference": {
            "content_material": ext_content.get("negative_content_material"),
            "production_context": ext_prod.get("negative_production_context")
        },
        "usage_patterns": {
            "temporal_patterns": ext_temp.get("temporal_patterns"),
            "immersion_style": ext_immer.get("immersion_style")
        },
        "decision_drivers": {
            "selection_priority": ext_priority.get("selection_priority"),
            "motivations": ext_motives.get("behavioral_motivation")
        }
    }
    
    return meta.user_id, meta.step, meta.session_key, extracted_formatted


def extract_and_persist(meta: SessionMeta, run_dir: str):
    out_path = extract_path(run_dir, meta.user_id, meta.step)
    if os.path.exists(out_path):
        return meta.user_id, meta.step, "skipped"

    uid, step, s_key, extracted = extract_worker(meta)
    payload = {
        "user_id": uid,
        "step": step,
        "session_key": s_key,
        "extracted": extracted,
        "saved_at": time.time()
    }
    
    save_json_atomic(out_path, payload)
    return uid, step, "saved"


def load_extracted(run_dir: str, uid: str, step: int):
    p = extract_path(run_dir, uid, step)
    data = load_json_if_exists(p)
    if data is None:
        return None
    return data["session_key"], data["extracted"]


# -----------------------
# Phase B: UPDATE
# -----------------------
def update_with_extracted(current_persona: dict, extracted: dict):
    
    # Access extracted directly because it now has the same structure as persona
    
    # 1. content/prod/temp/immer update
    res_content = get_llm_json_with_cache(CONTENT_MATERIAL_UPDATE.format(
        current_positive_content=current_persona["positive_preference"]["content_material"],
        current_negative_content=current_persona["negative_preference"]["content_material"],
        new_positive_content=extracted["positive_preference"]["content_material"],
        new_negative_content=extracted["negative_preference"]["content_material"]
    ))
    
    res_prod = get_llm_json_with_cache(PRODUCTION_CONTEXT_UPDATE.format(
        current_positive_production=current_persona["positive_preference"]["production_context"],
        current_negative_production=current_persona["negative_preference"]["production_context"],
        new_positive_production=extracted["positive_preference"]["production_context"],
        new_negative_production=extracted["negative_preference"]["production_context"]
    ))

    res_temp = get_llm_json_with_cache(TEMPORAL_PATTERNS_UPDATE.format(
        current_temporal_persona=current_persona["usage_patterns"]["temporal_patterns"],
        new_temporal_persona=extracted["usage_patterns"]["temporal_patterns"]
    ))

    res_immer = get_llm_json_with_cache(IMMERSION_STYLE_UPDATE.format(
        current_immersion_persona=current_persona["usage_patterns"]["immersion_style"],
        new_immersion_persona=extracted["usage_patterns"]["immersion_style"]
    ))

    # 2. priority/motive update
    updated_pref_ref = {
        "positive": f"{res_content.get('positive_content_material')} {res_prod.get('positive_production_context')}",
        "negative": f"{res_content.get('negative_content_material')} {res_prod.get('negative_production_context')}"
    }

    res_priority = get_llm_json_with_cache(SELECTION_PRIORITY_UPDATE.format(
        current_selection_priority=current_persona["decision_drivers"]["selection_priority"],
        new_selection_priority=extracted["decision_drivers"]["selection_priority"],
        updated_preferences=json.dumps(updated_pref_ref, ensure_ascii=False)
    ))

    res_motives = get_llm_json_with_cache(MOTIVATIONS_UPDATE.format(
        current_motivation_persona=current_persona["decision_drivers"]["motivations"],
        new_motivation_persona=extracted["decision_drivers"]["motivations"],
        updated_preferences=json.dumps(updated_pref_ref, ensure_ascii=False)
    ))

    # 3. Combine final results as before
    return {
        "historical_stats": extracted["historical_stats"], # Updated stats are already returned by extract_worker
        "positive_preference": {
            "content_material": res_content.get("positive_content_material"),
            "production_context": res_prod.get("positive_production_context")
        },
        "negative_preference": {
            "content_material": res_content.get("negative_content_material"),
            "production_context": res_prod.get("negative_production_context")
        },
        "usage_patterns": {
            "temporal_patterns": res_temp.get("temporal_patterns"),
            "immersion_style": res_immer.get("immersion_style")
        },
        "decision_drivers": {
            "selection_priority": res_priority.get("selection_priority"),
            "motivations": res_motives.get("behavioral_motivation")
        }
    }


def load_checkpoint(run_dir: str, uid: str):
    return load_json_if_exists(checkpoint_path(run_dir, uid))

def save_checkpoint(run_dir: str, uid: str, last_done_step: int, persona: dict):
    payload = {
        "user_id": uid,
        "last_done_step": last_done_step,
        "persona": persona,
        "saved_at": time.time()
    }
    save_json_atomic(checkpoint_path(run_dir, uid), payload)
                                                   
def compile_history_from_jsonl(run_dir: str, user_ids: list[str], init_personas: dict, output_path: str):
    out = {}
    for uid in user_ids:
        out[uid] = {}

        # --- Implementation note ---
        if uid in init_personas:
            initial_data = init_personas[uid]
            out[uid]["0"] = {
                "extracted": initial_data,      # Initial persona data referenced in the request
                "updated_persona": initial_data # Set step 0 identically because there is no update
            }
        # ------------------------------------------------------

        p = history_path(run_dir, uid)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    session_idx = str(rec["step"] + 1) # Sessions 1, 2, ...
                    
                    out[uid][session_idx] = {
                        "extracted": rec["extracted"],
                        "updated_persona": rec["updated_persona"]
                    }
        
        # Update the final key
        if out[uid]:
            last_step = max(out[uid].keys(), key=int)
            out[uid]["final"] = out[uid][last_step]["updated_persona"]

    save_json_atomic(output_path, out)


# -----------------------
# Main runner (2-Phase + Resume)
# -----------------------
def run_two_phase_resume(
    run_name="default_run",
    max_workers_extract=50,
    max_workers_update=50,
    context_k=2,
    limit_users=None,
    write_history_jsonl=True
):
    data_dir = "data"
    run_dir = os.path.join(data_dir, "runs", run_name)
    ensure_dir(run_dir)

    # load data
    init_b_personas = json.load(open(f"{data_dir}/behavioral_persona_init.json", "r"))
    sessions_data = json.load(open(f"{data_dir}/behavioral_log_sessions.json", "r"))

    user_ids = list(sessions_data.keys())
    if limit_users is not None:
        user_ids = user_ids[:limit_users]

    # Preprocess: create metadata
    metas_by_user = {}
    all_metas = []
    for uid in user_ids:
        try:
            metas = build_session_metas_for_user(
                uid, sessions_data[uid], init_b_personas[uid], context_k=context_k
            )
        except Exception as e:
            print("session_data", uid in sessions_data.keys())
            print("init_b_personas", uid in init_b_personas.keys())
            raise e
        metas_by_user[uid] = metas
        all_metas.extend(metas)

    # Phase A: EXTRACT only missing items
    pending = [m for m in all_metas if not os.path.exists(extract_path(run_dir, m.user_id, m.step))]
    print(f"🚀 Phase A(EXTRACT): total={len(all_metas)}, pending={len(pending)}, workers={max_workers_extract}")

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers_extract) as ex:
            futs = [ex.submit(extract_and_persist, m, run_dir) for m in pending]
            for _ in tqdm(as_completed(futs), total=len(futs), desc="Phase A - Extract"):
                pass

    # Phase B: UPDATE with checkpoint-based resume
    current_state = {}
    start_step = {}
    for uid in user_ids:
        ck = load_checkpoint(run_dir, uid)
        if ck is None:
            current_state[uid] = copy.deepcopy(init_b_personas[uid])
            start_step[uid] = 0
        else:
            current_state[uid] = ck["persona"]
            start_step[uid] = ck["last_done_step"] + 1

    max_step = max(len(metas_by_user[uid]) for uid in user_ids) - 1
    print(f"🚀 Phase B(UPDATE): max_step={max_step}, workers={max_workers_update}")

    for step in range(0, max_step + 1):
        wave_users = []
        for uid in user_ids:
            if step < start_step[uid]:
                continue
            if step >= len(metas_by_user[uid]):
                continue
            wave_users.append(uid)

        if not wave_users:
            continue

        # Modify the _update_job function inside Phase B
        def _update_job(uid_):
            loaded = load_extracted(run_dir, uid_, step) 
            if loaded is None:
                raise RuntimeError(f"Missing extract: uid={uid_}, step={step}")
            s_key, extracted = loaded  # Retrieve the extraction result here.

            persona_before = copy.deepcopy(current_state[uid_])
            # Run persona update
            new_state = update_with_extracted(current_state[uid_], extracted)
            
            # Return both the extraction result and the updated state.
            return uid_, step, s_key, extracted, new_state 

        # Inside the execution loop where fut.result() is received
        with ThreadPoolExecutor(max_workers=max_workers_update) as ex:
            futs = [ex.submit(_update_job, uid) for uid in wave_users]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Phase B - Update"):
                uid, step_, s_key, ext, after = fut.result() # Add ext

                current_state[uid] = after
                save_checkpoint(run_dir, uid, step_, after)

                if write_history_jsonl:
                    append_jsonl(history_path(run_dir, uid), {
                        "step": step_,
                        "session_key": s_key,
                        "extracted": ext,        # Save extraction results
                        "updated_persona": after, # Save update results
                        "saved_at": time.time()
                    })

    # Save final personas
    final_out = os.path.join(run_dir, "final_personas.json")
    save_json_atomic(final_out, {uid: current_state[uid] for uid in user_ids})

    # Create history in the existing format
    # history_out = os.path.join(run_dir, "behavioral_persona_history.json")
    # compile_history_from_jsonl(run_dir, user_ids, history_out)
    history_out = os.path.join(run_dir, "behavioral_persona_history.json")
    compile_history_from_jsonl(run_dir, user_ids, init_b_personas, history_out)

    print(f"Done.")
    print(f" - final_personas: {final_out}")
    print(f" - behavioral_persona_history: {history_out}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--max_workers_extract", type=int, default=50)
    parser.add_argument("--max_workers_update", type=int, default=50)
    parser.add_argument("--context_k", type=int, default=3)
    parser.add_argument("--limit_users", type=int, default=3)
    parser.add_argument("--write_history_jsonl", action="store_true")
    args = parser.parse_args()

    import pprint
    pprint.pprint(vars(args))

    # Example run
    run_two_phase_resume(
        run_name=args.run_name,
        max_workers_extract=args.max_workers_extract,
        max_workers_update=args.max_workers_update,
        context_k=args.context_k,
        limit_users=args.limit_users,
        write_history_jsonl=args.write_history_jsonl
    )
