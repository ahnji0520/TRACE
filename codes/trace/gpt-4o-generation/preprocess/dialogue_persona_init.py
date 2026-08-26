import json
import os

def create_dialogue_persona_init():
    input_path = "data/original_user_data.json"
    output_path = "data/dialogue_persona_init.json"
    
    # 1. Implementation note
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Implementation note.
            f.seek(0)
            content = f.read().replace('NaN', 'null')
            data = json.loads(content)

    # 2. Implementation note
    new_data = {}
    
    for user_id, user_info in data.items():
        new_data[user_id] = {
            "static": {
                "user_age": user_info.get("user_age"),
                "user_gender": user_info.get("user_gender"),
                "user_occupation": user_info.get("user_occupation"),
                "has_kids": user_info.get("has_kids"),
                "relationship_status": user_info.get("relationship_status"),
                "comm_style": user_info.get("comm_style", {})
            },
            "dynamic": []  # Implementation note.
        }

    # 3. Implementation note
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f"Initial dialogue persona file created: {output_path}")

if __name__ == "__main__":
    create_dialogue_persona_init()