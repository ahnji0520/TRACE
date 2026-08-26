# import sys
# import os
from pathlib import Path
# import json
# import numpy as np
# from openai import OpenAI
# from dotenv import load_dotenv

# 1. path setup
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# 2. Implementation note
# from src.dataset_generation.prompt.behavioral_persona_extract_prompt import (
#     CONTENT_MATERIAL_EXTRACT,
#     PRODUCTION_CONTEXT_EXTRACT,
#     TEMPORAL_PATTERNS_EXTRACT,
#     IMMERSION_STYLE_EXTRACT,
#     SELECTION_PRIORITY_EXTRACT,
#     MOTIVATIONS_EXTRACT
# )

# 3. Implementation note
# ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
# load_dotenv(ENV_PATH)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def calculate_initial_stats(watch_logs):
# Implementation note.
#     all_pcts = []
#     weekday_freq = {}
#     total_count = 0
    
#     for date, data in watch_logs.items():
#         day = data.get('weekday')
#         records = data.get('records', [])
#         record_count = len(records)
        
# Implementation note.
#         weekday_freq[day] = weekday_freq.get(day, 0) + record_count
#         total_count += record_count
        
#         for record in records:
#             all_pcts.append(record.get('watched_pct', 0))
            
#     if not all_pcts:
#         return {"watched_pct_avg": 0, "watched_pct_var": 0, "total_count": 0, "weekday_freq": {}}
    
#     return {
#         "watched_pct_avg": round(float(np.mean(all_pcts)), 2),
#         "watched_pct_var": round(float(np.var(all_pcts)), 2),
#         "total_count": total_count,
#         "weekday_freq": weekday_freq
#     }

# def get_llm_json(prompt):
# Implementation note.
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0.2
#         )
#         return json.loads(response.choices[0].message.content)
#     except Exception as e:
# Implementation note.
#         return {}

# def run_behavioral_persona_init():
#     input_path = os.path.join(project_root, "data/behavioral_log_init.json")
#     output_path = os.path.join(project_root, "data/behavioral_persona_init.json")
    
#     if not os.path.exists(input_path):
# Implementation note.
#         return

#     with open(input_path, 'r', encoding='utf-8') as f:
#         content = f.read().replace('NaN', 'null')
#         raw_data = json.loads(content)
        
#     final_persona_data = {}

# Implementation note.
#     target_users = list(raw_data.items())[:20]
# Implementation note.

#     for user_id, user_logs in target_users:
#         print(f"🚀 Initializing Persona for User: {user_id}")
        
#         watch_data = user_logs.get('watch', {})
        
# --- Implementation note ---
#         full_stats = calculate_initial_stats(watch_data)
        
#         stats_limited_str = json.dumps({
#             "watched_pct_avg": full_stats["watched_pct_avg"],
#             "watched_pct_var": full_stats["watched_pct_var"]
#         })
#         stats_full_str = json.dumps(full_stats)

# --- Implementation note ---
#         usage_logs = {date: {"day": d["weekday"], "watched_pct": [r["watched_pct"] for r in d["records"]]} 
#                       for date, d in watch_data.items()}
        
#         content_logs = []
#         production_logs = []
#         union_logs = []
#         for d in watch_data.values():
#             for r in d["records"]:
#                 content_logs.append({k: r.get(k) for k in ["watched_pct", "content_type", "title", "genres", "description", "keywords"]})
#                 production_logs.append({k: r.get(k) for k in ["watched_pct", "content_type", "title", "release_year", "countries", "actors", "directors"]})
#                 union_logs.append({k: r.get(k) for k in ["watched_pct", "content_type", "title", "genres", "description", "keywords", "release_year", "countries", "actors", "directors"]})

# --- Implementation note ---
        
#         pref_content = get_llm_json(CONTENT_MATERIAL_EXTRACT.format(
#             user_completion_stats=stats_limited_str, 
#             log_data=json.dumps(content_logs)
#         ))
#         pref_prod = get_llm_json(PRODUCTION_CONTEXT_EXTRACT.format(
#             user_completion_stats=stats_limited_str, 
#             log_data=json.dumps(production_logs)
#         ))
        
#         usage_temp = get_llm_json(TEMPORAL_PATTERNS_EXTRACT.format(
#             user_stats=stats_full_str, 
#             log_data=json.dumps(usage_logs)
#         ))
#         usage_immer = get_llm_json(IMMERSION_STYLE_EXTRACT.format(
#             user_stats=stats_full_str,
#             log_data=json.dumps(usage_logs)
#         ))
        
#         merged_pref_str = json.dumps({**pref_content, **pref_prod})
        
#         dec_priority = get_llm_json(SELECTION_PRIORITY_EXTRACT.format(
#             preference=merged_pref_str, 
#             user_completion_stats=stats_limited_str,
#             log_data=json.dumps(union_logs)
#         ))
#         dec_motives = get_llm_json(MOTIVATIONS_EXTRACT.format(
#             preference=merged_pref_str, 
#             user_completion_stats=stats_limited_str, 
#             log_data=json.dumps(union_logs)
#         ))

# --- Implementation note ---
#         final_persona_data[user_id] = {
#             "historical_stats": full_stats,
#             "positive_preference": {
#                 "content_material": pref_content.get("positive_content_material"),
#                 "production_context": pref_prod.get("positive_production_context")
#             },
#             "negative_preference": {
#                 "content_material": pref_content.get("negative_content_material"),
#                 "production_context": pref_prod.get("negative_production_context")
#             },
#             "usage_patterns": {
#                 "temporal_patterns": usage_temp.get("temporal_patterns"),
#                 "immersion_style": usage_immer.get("immersion_style")
#             },
#             "decision_drivers": {
#                 "selection_priority": dec_priority.get("selection_priority"),
#                 "motivations": dec_motives.get("behavioral_motivation")
#             }
#         }

# file save
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_persona_data, f, indent=4, ensure_ascii=False)
    
# Implementation note.

# if __name__ == "__main__":
#     run_behavioral_persona_init()

import sys
import os
import json
import threading
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 1. Implementation note
current_dir = os.path.dirname(os.path.abspath(__file__))
# Implementation note.
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. Implementation note
from src.dataset_generation.prompt.behavioral_persona_extract_prompt import (
    CONTENT_MATERIAL_EXTRACT,
    PRODUCTION_CONTEXT_EXTRACT,
    TEMPORAL_PATTERNS_EXTRACT,
    IMMERSION_STYLE_EXTRACT,
    SELECTION_PRIORITY_EXTRACT,
    MOTIVATIONS_EXTRACT
)

# 3. Implementation note
SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
load_dotenv(ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Implementation note.
MAX_INFLIGHT_LLM = int(os.getenv("MAX_INFLIGHT_LLM", "40"))
llm_sema = threading.Semaphore(MAX_INFLIGHT_LLM)

# -----------------------
# Implementation note.
# -----------------------

def calculate_initial_stats(watch_logs):
    """Calculate cumulative stats from initial behavior logs"""
    all_pcts = []
    weekday_freq = {}
    total_count = 0
    
    for date, data in watch_logs.items():
        day = data.get('weekday')
        records = data.get('records', [])
        record_count = len(records)
        
        # Implementation note.
        weekday_freq[day] = weekday_freq.get(day, 0) + record_count
        total_count += record_count
        
        for record in records:
            all_pcts.append(record.get('watched_pct', 0))
            
    if not all_pcts:
        return {"watched_pct_avg": 0, "watched_pct_var": 0, "total_count": 0, "weekday_freq": {}}
    
    return {
        "watched_pct_avg": round(float(np.mean(all_pcts)), 2),
        "watched_pct_var": round(float(np.var(all_pcts)), 2),
        "total_count": total_count,
        "weekday_freq": weekday_freq
    }

def get_llm_json(prompt):
    """Call the OpenAI API and return a JSON response with semaphore control"""
    with llm_sema:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"API call failed: {e}")
            return {}

# -----------------------
# Implementation note.
# -----------------------

def process_single_user(user_id, user_logs):
    """Process one user log and extract the initial persona"""
    try:
        watch_data = user_logs.get('watch', {})
        
        # Implementation note.
        full_stats = calculate_initial_stats(watch_data)
        
        stats_limited_str = json.dumps({
            "watched_pct_avg": full_stats["watched_pct_avg"],
            "watched_pct_var": full_stats["watched_pct_var"]
        }, ensure_ascii=False)
        stats_full_str = json.dumps(full_stats, ensure_ascii=False)

        # Implementation note.
        usage_logs = {date: {"day": d["weekday"], "watched_pct": [r["watched_pct"] for r in d["records"]]} 
                      for date, d in watch_data.items()}
        
        content_logs, production_logs, union_logs = [], [], []
        for d in watch_data.values():
            for r in d["records"]:
                content_logs.append({k: r.get(k) for k in ["watched_pct", "content_type", "title", "genres", "description", "keywords"]})
                production_logs.append({k: r.get(k) for k in ["watched_pct", "content_type", "title", "release_year", "countries", "actors", "directors"]})
                union_logs.append({k: r.get(k) for k in ["watched_pct", "content_type", "title", "genres", "description", "keywords", "release_year", "countries", "actors", "directors"]})

        # Implementation note.
        pref_content = get_llm_json(CONTENT_MATERIAL_EXTRACT.format(
            user_completion_stats=stats_limited_str, 
            log_data=json.dumps(content_logs, ensure_ascii=False)
        ))
        pref_prod = get_llm_json(PRODUCTION_CONTEXT_EXTRACT.format(
            user_completion_stats=stats_limited_str, 
            log_data=json.dumps(production_logs, ensure_ascii=False)
        ))
        usage_temp = get_llm_json(TEMPORAL_PATTERNS_EXTRACT.format(
            user_stats=stats_full_str, 
            log_data=json.dumps(usage_logs, ensure_ascii=False)
        ))
        usage_immer = get_llm_json(IMMERSION_STYLE_EXTRACT.format(
            user_stats=stats_full_str,
            log_data=json.dumps(usage_logs, ensure_ascii=False)
        ))
        
        merged_pref_str = json.dumps({**pref_content, **pref_prod}, ensure_ascii=False)
        
        dec_priority = get_llm_json(SELECTION_PRIORITY_EXTRACT.format(
            preference=merged_pref_str, 
            user_completion_stats=stats_limited_str,
            log_data=json.dumps(union_logs, ensure_ascii=False)
        ))
        dec_motives = get_llm_json(MOTIVATIONS_EXTRACT.format(
            preference=merged_pref_str, 
            user_completion_stats=stats_limited_str, 
            log_data=json.dumps(union_logs, ensure_ascii=False)
        ))

        # Implementation note.
        result = {
            "historical_stats": full_stats,
            "positive_preference": {
                "content_material": pref_content.get("positive_content_material"),
                "production_context": pref_prod.get("positive_production_context")
            },
            "negative_preference": {
                "content_material": pref_content.get("negative_content_material"),
                "production_context": pref_prod.get("negative_production_context")
            },
            "usage_patterns": {
                "temporal_patterns": usage_temp.get("temporal_patterns"),
                "immersion_style": usage_immer.get("immersion_style")
            },
            "decision_drivers": {
                "selection_priority": dec_priority.get("selection_priority"),
                "motivations": dec_motives.get("behavioral_motivation")
            }
        }
        return user_id, result

    except Exception as e:
        print(f"Failed to process user {user_id}: {e}")
        return user_id, None

def run_behavioral_persona_init(max_workers=50):
    input_path = os.path.join(project_root, "data/behavioral_log_init.json")
    output_path = os.path.join(project_root, "data/behavioral_persona_init.json")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        # Implementation note.
        content = f.read().replace('NaN', 'null')
        raw_data = json.loads(content)
        
    final_persona_data = {}
    # Implementation note.
    target_users = list(raw_data.items())[:20]
    
    print(f"Starting initial persona generation for {len(target_users)} users with {max_workers} workers.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_user, uid, logs): uid for uid, logs in target_users}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting initial personas"):
            uid, result = future.result()
            if result:
                final_persona_data[uid] = result

    # Implementation note.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_persona_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nDone. Saved data for {len(final_persona_data)} users: {output_path}")

if __name__ == "__main__":
    run_behavioral_persona_init(max_workers=50)