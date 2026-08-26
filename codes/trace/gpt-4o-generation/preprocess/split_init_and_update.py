import json
import os

def split_by_item_count():
    # 1. file path setup
    input_file = "data/original_user_data.json"
    data_dir = "data"
    
    init_output = os.path.join(data_dir, "behavioral_log_init.json")
    update_output = os.path.join(data_dir, "behavioral_log_update.json")

    if not os.path.exists(input_file):
        print(f"Error: file not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    init_data = {}
    update_data = {}

    for user_id, content in data.items():
        watch_sessions = content.get("watch", {})
        sorted_dates = sorted(watch_sessions.keys())

        # Flatten all watched items while preserving date and weekday metadata.
        all_items = []
        for date in sorted_dates:
            day_info = watch_sessions[date].get("weekday", "")
            for record in watch_sessions[date].get("records", []):
                all_items.append({
                    "date": date,
                    "weekday": day_info,
                    "record": record
                })

        total_item_count = len(all_items)
        if total_item_count == 0:
            continue

        # Use the first 20 percent of items as initialization data.
        split_point = max(1, round(total_item_count * 0.2))
        
        init_items = all_items[:split_point]
        update_items = all_items[split_point:]

        # Rebuild the watch dictionary grouped by date.
        def rebuild_watch(item_list):
            new_watch = {}
            for item in item_list:
                d = item["date"]
                if d not in new_watch:
                    new_watch[d] = {
                        "weekday": item["weekday"],
                        "records": []
                    }
                new_watch[d]["records"].append(item["record"])
            return new_watch

        init_data[user_id] = {"watch": rebuild_watch(init_items)}
        update_data[user_id] = {"watch": rebuild_watch(update_items)}

    # 4. file save
    with open(init_output, 'w', encoding='utf-8') as f:
        json.dump(init_data, f, indent=4, ensure_ascii=False)
    with open(update_output, 'w', encoding='utf-8') as f:
        json.dump(update_data, f, indent=4, ensure_ascii=False)

    print("--- Split by item count completed ---")
    print(f"For 25 total items, the split is saved as Init(5) / Update(20).")

if __name__ == "__main__":
    split_by_item_count()