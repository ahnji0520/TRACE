# import json
# import random
# import os

# def split_sessions():
#     input_path = "data/behavioral_log_update.json"
#     output_path = "data/behavioral_log_sessions.json"

#     if not os.path.exists(input_path):
# Implementation note.
#         return

#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     final_result = {}

#     for user_id, user_data in data.items():
#         watch_data = user_data.get("watch", {})
#         if not watch_data:
#             continue

# 1. Implementation note
#         all_items = []
#         for date in sorted(watch_data.keys()):
#             day_info = watch_data[date]
#             for record in day_info.get("records", []):
# Implementation note.
#                 item = record.copy()
#                 item["watched_date"] = date
#                 item["watched_weekday"] = day_info.get("weekday", "")
#                 all_items.append(item)

# 2. Implementation note
#         temp_sessions = []
#         current_chunk = []
#         for item in all_items:
#             current_chunk.append(item)
#             if (item.get("watched_pct") or 0) >= 80.0:
#                 temp_sessions.append(current_chunk)
#                 current_chunk = []
        
# Implementation note.
#         if not temp_sessions:
#             continue

# 3. Implementation note
#         merged_sessions = []
# Implementation note.

#         for sess in temp_sessions:
#             if not merged_sessions:
#                 merged_sessions.append(sess)
#                 is_already_merged = False
#                 continue
            
#             prev_sess = merged_sessions[-1]
            
# Implementation note.
#             if not is_already_merged and (len(sess) == 1 or sess[0]["watched_date"] == prev_sess[-1]["watched_date"]):
#                 merged_sessions[-1].extend(sess)
# Implementation note.
#             else:
#                 merged_sessions.append(sess)
# Implementation note.

# 4. Set in-session targets and sample up to three items
#         user_sessions = {}
#         for idx, sess in enumerate(merged_sessions):
#             target_item = sess[-1]
#             source_items = sess[:-1]

# Set target items
#             target_item["target"] = True
#             target_item["in_session"] = True

# Implementation note.
#             if len(source_items) > 3:
# 4 Implementation note
#                 sampled_indices = set(random.sample(range(len(source_items)), 3))
#                 for s_idx, s_item in enumerate(source_items):
#                     s_item["target"] = False
#                     s_item["in_session"] = True if s_idx in sampled_indices else False
#             else:
# 3 Implementation note
#                 for s_item in source_items:
#                     s_item["target"] = False
#                     s_item["in_session"] = True

# Implementation note.
#             user_sessions[str(idx + 1)] = {
#                 "date": target_item["watched_date"],
#                 "weekday": target_item["watched_weekday"],
#                 "watch": sess
#             }

#         final_result[user_id] = user_sessions

# 5. Save
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_result, f, indent=2, ensure_ascii=False)
    
# Implementation note.

# if __name__ == "__main__":
#     split_sessions()

import json
import random
import os

def split_sessions():
    input_path = "data/behavioral_log_update.json"
    output_path = "data/behavioral_log_sessions.json"

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        # Handle NaN values if needed
        data = json.load(f)

    final_result = {}

    for user_id, user_data in data.items():
        watch_data = user_data.get("watch", {})
        if not watch_data:
            continue

        # 1. Sort all items chronologically and flatten them
        all_items = []
        for date in sorted(watch_data.keys()):
            day_info = watch_data[date]
            for record in day_info.get("records", []):
                item = record.copy()
                item["watched_date"] = date
                item["watched_weekday"] = day_info.get("weekday", "")
                all_items.append(item)

        # ---------------------------------------------------------
        # 2. Step 1: split at 80 percent completion and merge to satisfy the minimum size of two
        # ---------------------------------------------------------
        step1_sessions = []
        accumulated_items = [] # Accumulate items below 80 percent completion or with fewer than two items
        temp_chunk = []

        for item in all_items:
            temp_chunk.append(item)
            # Complete one watch chunk when watch progress is at least 80 percent
            if (item.get("watched_pct") or 0) >= 80.0:
                accumulated_items.extend(temp_chunk)
                temp_chunk = []
                
                # Register an independent session candidate only when at least two items accumulated
                if len(accumulated_items) >= 2:
                    step1_sessions.append(accumulated_items)
                    accumulated_items = []
        
        # Handle remaining items that do not reach 80 percent or two items at the end
        remaining = accumulated_items + temp_chunk
        if remaining:
            if step1_sessions:
                step1_sessions[-1].extend(remaining) # Merge into the previous session
            else:
                step1_sessions.append(remaining) # Keep this if the data is too sparse

        # ---------------------------------------------------------
        # 3. Step 2: final merge based on the last item date (daily consolidation)
        # ---------------------------------------------------------
        final_merged_sessions = []
        for sess in step1_sessions:
            if not final_merged_sessions:
                final_merged_sessions.append(sess)
                continue
            
            prev_sess = final_merged_sessions[-1]
            
            # Merge when the previous session last date equals the current session last date
            if sess[-1]["watched_date"] == prev_sess[-1]["watched_date"]:
                final_merged_sessions[-1].extend(sess)
            else:
                final_merged_sessions.append(sess)

        # ---------------------------------------------------------
        # 4. Set in-session targets and sample up to three items
        # ---------------------------------------------------------
        user_sessions = {}
        for idx, sess in enumerate(final_merged_sessions):
            target_item = sess[-1]
            source_items = sess[:-1]

            # Set target items
            target_item["target"] = True
            target_item["in_session"] = True

            # Process and sample source items while adjusting the context window
            if len(source_items) > 3:
                # Randomly select three items; index-based sorted sampling is recommended to preserve order
                indices = sorted(random.sample(range(len(source_items)), 3))
                sampled_indices = set(indices)
                for s_idx, s_item in enumerate(source_items):
                    s_item["target"] = False
                    s_item["in_session"] = True if s_idx in sampled_indices else False
            else:
                for s_item in source_items:
                    s_item["target"] = False
                    s_item["in_session"] = True

            # Build session metadata
            user_sessions[str(idx + 1)] = {
                "date": target_item["watched_date"],
                "weekday": target_item["watched_weekday"],
                "watch": sess
            }

        final_result[user_id] = user_sessions

    # 5. Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"2-step session merge completed: {output_path}")

if __name__ == "__main__":
    split_sessions()