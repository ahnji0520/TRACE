import json
import os
from pathlib import Path
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Keep existing imports
from dataset_generation.dialogue_generation.dialogue_persona_and_memory_updater import update_dialogue_persona
from src.dataset_generation.prompt.dialogue_generation_prompt import DIALOGUE_GENERATION, DIALOGUE_GENERATION_2

SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
load_dotenv(ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Lock for synchronizing shared resources
file_lock = threading.Lock()

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().replace('NaN', 'null')
        return json.loads(content)

def process_single_user(user_id, sessions, d_personas, b_persona_history, events_data, final_dataset, output_path):
    """
    Process all sessions for one user sequentially within a thread.
    """
    # Skip users that were already processed
    with file_lock:
        if user_id in final_dataset:
            return f"⏩ User {user_id} already processed."

    print(f"🚀 Processing User {user_id}...")
    
    # Initialize memory independently per user
    current_memory = {
        "episodic_memory": [],
        "recommendation_outcomes": []
    }
    
    user_sessions_list = []
    sorted_keys = sorted(sessions.keys(), key=lambda x: int(x))
    
    # Use a copy of the user persona for state tracking
    local_d_persona = copy.deepcopy(d_personas[user_id])

    for s_idx in sorted_keys:
        session = sessions[s_idx]

        # [1] Memory filtering logic
        filtered_memory = copy.deepcopy(current_memory)
        if "recommendation_outcomes" in filtered_memory:
            filtered_memory["recommendation_outcomes"] = [
                outcome for outcome in filtered_memory["recommendation_outcomes"]
                if outcome.get("reaction") == "Accept"
            ]
        
        # [2] Data preprocessing
        source_items = [ {k: v for k, v in item.items() if k not in ['watched_date', 'watched_weekday']} 
                        for item in session['watch'] if item.get('in_session') and not item.get('target') ]
        target_item = next(({k: v for k, v in item.items() if k not in ['watched_date', 'watched_weekday']} 
                           for item in session['watch'] if item.get('target')), None)

        # [3] Load the behavioral persona from the previous session
        prev_history_key = str(int(s_idx) - 1)
        if prev_history_key not in b_persona_history.get(user_id, {}):
            continue
            
        session_behavioral_persona = b_persona_history[user_id][prev_history_key]["updated_persona"]
        full_stats = session_behavioral_persona.get('historical_stats', {})
        limited_stats = {
            "watched_pct_avg": full_stats.get("watched_pct_avg", 0),
            "watched_pct_var": full_stats.get("watched_pct_var", 0)
        }

        # [4] Generate dialogue
        event = events_data.get(user_id, {}).get(str(s_idx), "")
        day_str = f"{session['date']}, {session['weekday']}"
        
        # For saving the input persona
        input_d_persona_snapshot = copy.deepcopy(local_d_persona)

        system_prompt = DIALOGUE_GENERATION
        user_prompt = DIALOGUE_GENERATION_2.format(
            dialogue_persona=json.dumps(local_d_persona, indent=2),
            behavioral_persona=json.dumps(session_behavioral_persona, indent=2),
            day=day_str,
            event=event,
            memory=json.dumps(filtered_memory, indent=2, ensure_ascii=False),
            historical_completion_stats=json.dumps(limited_stats),
            source_items=json.dumps(source_items, indent=2),
            target_item=json.dumps(target_item, indent=2)
        )

        try:
            # OpenAI prompt caching is applied automatically when the same system prompt and user prompt prefix are maintained.
            res = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {'role': 'system', 'content' : system_prompt},
                    {"role": "user", "content": user_prompt}
                ], 
                response_format={"type": "json_object"}
            )
            llm_response = json.loads(res.choices[0].message.content)
        except Exception as e:
            return f"❌ Error for User {user_id}, Session {s_idx}: {e}"

        dialogue_content = llm_response.get("Dialogue", [])
        design_strategy = llm_response.get("Dialogue Design Strategy", "")

        # [5] Update dialogue persona and memory by calling the function
        updated_d_persona, next_memory, extracted_d_persona = update_dialogue_persona(
            current_d_persona=local_d_persona,
            dialogue_json=dialogue_content,
            day=day_str,
        )

        # Save results
        session_entry = {
            "session_id": s_idx,
            "date": session['date'],
            "weekday": session['weekday'],
            "dialogue_persona": {
                "input": input_d_persona_snapshot,
                "extracted": extracted_d_persona,
                "updated": updated_d_persona
            },
            "behavioral_persona": session_behavioral_persona,
            "memory": copy.deepcopy(current_memory),
            "dialogue_design_strategy": design_strategy,
            "dialogue": dialogue_content
        }
        user_sessions_list.append(session_entry)
        
        # Update state for the next session
        new_episode = next_memory.get("session_event")
        if new_episode:
            current_memory["episodic_memory"].append(new_episode)
        current_memory["recommendation_outcomes"] = next_memory.get("recommendation_outcomes", [])
        local_d_persona = updated_d_persona

    # Merge into the shared dictionary and save after the user task completes
    with file_lock:
        final_dataset[user_id] = user_sessions_list
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=4, ensure_ascii=False)
    
    return f"💾 User {user_id} saved."

def run_main_pipeline(args):
    data_dir = "data"
    output_path = f"{data_dir}/final_crs_dataset_0122.json"
    
    # [1] Load data
    d_personas = load_json(f"{data_dir}/dialogue_persona_init.json")
    b_persona_history = load_json(f"{data_dir}/behavioral_persona_history.json")
    sessions_data = load_json(f"{data_dir}/behavioral_log_sessions.json")
    events_data = load_json(f"{data_dir}/events.json")
    
    # [2] Load existing result file
    final_dataset = load_json(output_path) or {}
    if final_dataset:
        print(f"Loaded existing dataset. Users: {len(final_dataset)}")

    # Run parallel processing
    user_ids = list(sessions_data.keys())
    if args.limit_users is not None:
        user_ids = user_ids[:args.limit_users]

    print(f"Starting parallel processing. Max workers: 50")
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [
            executor.submit(
                process_single_user, 
                user_id = u_id, 
                sessions=sessions_data[u_id], 
                d_personas =d_personas, 
                b_persona_history = b_persona_history, 
                events_data = events_data, 
                final_dataset = final_dataset, 
                output_path = output_path
            ) 
            for u_id in user_ids
        ]
        
        # Show progress
        for future in tqdm(futures, desc="Overall Progress"):
            result = future.result()

    print(f"All tasks completed. Final dataset: {output_path}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pprint import pprint
    parser = ArgumentParser()
    parser.add_argument("--limit_users", type=int, default=None)
    args = parser.parse_args()
    pprint(vars(args))
    run_main_pipeline(args)