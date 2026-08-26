import json
import os
import copy

def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Merged 9 components with memory-specific differences: {filepath}")

def merge_9_components_fixed():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../../"))
    data_dir = os.path.join(project_root, "data")

    BEHAVIORAL_PATH = os.path.join(data_dir, "behavioral_persona_history.json")
    DIALOGUE_FILE_PATH = os.path.join(data_dir, "final_crs_dataset_ours.json")
    OUTPUT_PATH = os.path.join(data_dir, "persona_update_9_components.json")

    b_data = load_json(BEHAVIORAL_PATH)
    d_data = load_json(DIALOGUE_FILE_PATH)
    
    merged_output = {}

    for user_id, d_sessions in d_data.items():
        if user_id not in b_data: continue
            
        merged_output[user_id] = []
        user_b_history = b_data[user_id]
        
        # [Initial state setup]
        # 1. Behavioral
        initial_beh = user_b_history.get("0", {}).get("updated_persona", {})
        prev_beh_after = {k: v for k, v in initial_beh.items() if k != 'historical_stats'}
        
        # 2. For cumulative memory management; outcomes are accumulated directly
        stacked_outcomes = []
        # Episodic memory is already cumulative in the source, so store the previous state and compute the delta
        prev_episodic_after = []

        for i, d_session in enumerate(d_sessions):
            s_id = d_session.get("session_id")
            if s_id not in user_b_history: continue
            b_session = user_b_history[s_id]

            # --- 1. Behavioral Persona ---
            beh_before = copy.deepcopy(prev_beh_after)
            beh_current = {k: v for k, v in b_session.get('extracted', {}).items() if k != 'historical_stats'}
            beh_after = {k: v for k, v in b_session.get('updated_persona', {}).items() if k != 'historical_stats'}

            # --- 2. Dialogue Persona ---
            dia_before = d_session.get('dialogue_persona', {}).get('input', [])
            dia_current = d_session.get('dialogue_persona', {}).get('extracted', [])
            dia_after = d_session.get('dialogue_persona', {}).get('updated', [])

            # --- 3. Memory: merge the two types of data ---
            # A. Episodic Memory; the source is already cumulative
            ep_after = d_session.get('memory', {}).get('episodic_memory', [])
            ep_before = copy.deepcopy(prev_episodic_after)
            ep_current = ep_after[len(ep_before):] # Extract only the delta
            
            # B. Recommendation Outcomes; the source contains only the session and is accumulated directly
            out_before = copy.deepcopy(stacked_outcomes)
            out_current = d_session.get('memory', {}).get('recommendation_outcomes', [])
            out_after = out_before + out_current
            
            # Assemble nine components
            session_entry = {
                "session_id": s_id,
                "date": d_session.get("date"),
                "weekday": d_session.get("weekday"),
                "behavioral_persona": {
                    "before_update": beh_before,
                    "current_extraction": beh_current,
                    "after_update": beh_after
                },
                "dialogue_persona": {
                    "before_update": dia_before,
                    "current_extraction": dia_current,
                    "after_update": dia_after
                },
                "memory": {
                    "before_update": {
                        "episodic_memory": ep_before,
                        "recommendation_outcomes": out_before
                    },
                    "current_extraction": {
                        "episodic_memory": ep_current,
                        "recommendation_outcomes": out_current
                    },
                    "after_update": {
                        "episodic_memory": ep_after,
                        "recommendation_outcomes": out_after
                    }
                }
            }
            merged_output[user_id].append(session_entry)

            # State transition; becomes Before for the next session
            prev_beh_after = beh_after
            prev_episodic_after = ep_after
            stacked_outcomes = out_after

    save_json(merged_output, OUTPUT_PATH)

if __name__ == "__main__":
    merge_9_components_fixed()