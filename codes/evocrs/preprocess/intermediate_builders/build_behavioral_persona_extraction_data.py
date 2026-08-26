import json
import os

def load_json(filepath):
    """Load a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: file not found - {filepath}")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save result data as a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Merge and window processing completed. Save path: {filepath}")

def merge_with_windowing():
    # 1. path setup
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    # Move up to reach the project root (KT-Agent)
    project_root = os.path.abspath(os.path.join(base_dir, "../../../../"))
    data_dir = os.path.join(project_root, "data")

    LOG_PATH = os.path.join(data_dir, "behavioral_log_sessions.json")
    PERSONA_PATH = os.path.join(data_dir, "behavioral_persona_history.json")
    OUTPUT_PATH = os.path.join(data_dir, "behavioral_persona_extract_data.json")

    # List of keys to exclude
    EXCLUDE_KEYS = ["target", "in_session"]

    # 2. Load data
    logs = load_json(LOG_PATH)
    personas = load_json(PERSONA_PATH)
    
    merged_data = {}

    # 3. Merge data and process windows
    # [Updated] Loop by user to define user_id and user_sessions.
    for user_id, user_sessions in logs.items():
        merged_data[user_id] = {}
        
        # [Updated] Define s_keys by sorting session keys numerically ("1", "2", "3", ...).
        s_keys = sorted(user_sessions.keys(), key=int)
        
        for i in range(len(s_keys)):
            curr_key = s_keys[i]
            session_content = user_sessions[curr_key]
            
            # (1) Process behavioral-log windows with size 3: i-2, i-1, and the current session
            windowed_watch = []
            start_idx = max(0, i - 2) 
            
            for idx in range(start_idx, i + 1):
                win_key = s_keys[idx]
                win_watch = user_sessions[win_key].get("watch", [])
                
                for item in win_watch:
                    # Filter out unnecessary date information and related fields
                    filtered_item = {k: v for k, v in item.items() if k not in EXCLUDE_KEYS}
                    windowed_watch.append(filtered_item)
            
            # New session object containing windowed logs and basic information
            processed_session = {
                "date": session_content.get("date"),
                "weekday": session_content.get("weekday"),
                "watch": windowed_watch
            }

            # (2) Merge persona history
            if user_id in personas and curr_key in personas[user_id]:
                processed_session["persona_history"] = personas[user_id][curr_key]
            else:
                # Initialize with an empty dictionary when persona data is missing
                processed_session["persona_history"] = {}

            # Add to the final dictionary
            merged_data[user_id][curr_key] = processed_session

    # 4. Save results
    save_json(merged_data, OUTPUT_PATH)

if __name__ == "__main__":
    merge_with_windowing()