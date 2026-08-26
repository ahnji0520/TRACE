import json
import random
import string
from typing import *

def drop_initial_items(value, n=8):
    if n <= 0:
        return value
    if isinstance(value, list):
        return value[n:]
    if isinstance(value, tuple):
        return list(value[n:])
    return value

### PERSONA & MEMORY Processor Preprocess ####
def make_behavioral_persona_extract(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT, data):
    """
    Iterate over dictionary-form session data and build a behavioral persona extraction dataset.
    Use the 'persona_history -> extracted' field as the response target.
    """
    dataset = []

    for user_id, sessions in data.items():
        sorted_session_ids = sorted(sessions.keys(), key=lambda x: int(x))
        
        for s_id in sorted_session_ids:
            session_data = sessions[s_id]

            persona_extracted = session_data.get('persona_history', {}).get('extracted', {})
            if not persona_extracted:
                continue # Skip sessions without data

            hist_stats = persona_extracted.get('historical_stats', {})
            behavioral_log = session_data.get('watch', [])

            target_response = {
                k: v for k, v in persona_extracted.items() if k != 'historical_stats'
            }


            formatted_prompt = [{'role' : 'system', "content" : SYSTEM_PROMPT},
            {'role': 'user' , 'content' : USER_PROMPT.format(historical_stats = hist_stats, behavioral_log = json.dumps(behavioral_log, ensure_ascii=False, indent=2))}, 
            {'role' : 'assistant', 'content' : ASSISTANT_PROMPT.format(response=json.dumps(target_response, ensure_ascii=False, separators=(",", ":")))}]
                        
            instance = {
                "id": f"{user_id}_{s_id}", # ID starts from 1; this is correct
                "messages": formatted_prompt # A valid list is now inserted
            }
            
            dataset.append(instance)
                
    return dataset
    

# def make_dialogue_persona_and_memory_extract(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT, data):
#     """
# Build an instruction-tuning dataset that extracts the dialogue persona and memory from the current session 
# for use as the next-session input.
#     """
#     dataset = []

#     for user_id, sessions in data.items():
# Each user has a list of sessions
#         for i in range(len(sessions)):
#             curr_session = sessions[i]
    
#             # --- [1] Input: Day (Date + Weekday) ---
#             date = curr_session.get('date', '')
#             weekday = curr_session.get('weekday', '')
# Implementation note.

#             # --- [2] Input: Dialogue (Cleaned) ---
# Extract only speaker and text, excluding label and tagged_items
#             dialogue_list = []
#             for turn in curr_session.get('dialogue', []):
#                 speaker = turn.get('speaker', '')
#                 text = turn.get('text', '').strip()
#                 if speaker and text:
#                     dialogue_list.append(f"{speaker}: {text}")
#             dialogue_str = "\n".join(dialogue_list)

# --- Implementation note ---
#             target_persona = curr_session.get('dialogue_persona', {}).get('extracted', [])

# --- Implementation note ---
# Memory uses the current session output that is fed into the next session
#             target_memory = {}
#             if i + 1 < len(sessions):
#                 target_memory = sessions[i+1].get('memory', {})
#             else:
# Skip the last session because there is no next memory
#                 continue 

# --- [5] Response JSON build ---
#             response_json = {
#                 "dialogue_persona": target_persona,
#                 "memory": target_memory
#             }
# Use ensure_ascii=False to avoid corrupting non-ASCII text
#             response_str = json.dumps(response_json, ensure_ascii=False, indent=4)

# --- Implementation note ---
#             formatted_user_prompt = USER_PROMPT.format(day=day_str, dialogue=dialogue_str)
#             formatted_assistant_prompt = ASSISTANT_PROMPT.format(response=response_str)

# Standard dictionary structure for SFT training
#             instance = {
#                 "id": f"{user_id}_{i+1}",
#                 "messages": [
#                     {"role": "system", "content": SYSTEM_PROMPT.strip()},
#                     {"role": "user", "content": formatted_user_prompt.strip()},
#                     {"role": "assistant", "content": formatted_assistant_prompt.strip()}
#                 ]
#             }
#             dataset.append(instance)

#     return dataset
import json

def make_dialogue_persona_and_memory_extract(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT, data):
    """
    Build an SFT dataset that extracts the newly updated memory (next minus current) 
    and dialogue persona from the current session dialogue.
    """
    dataset = []

    for user_id, sessions in data.items():
        for i in range(len(sessions)):
            curr_session = sessions[i]
            
            # --- Implementation note ---
            date = curr_session.get('date', '')
            weekday = curr_session.get('weekday', '')
            day_str = f"{date}, {weekday}".strip(", ")

            dialogue_list = []
            for turn in curr_session.get('dialogue', []):
                speaker = turn.get('speaker', '')
                text = turn.get('text', '').strip()
                if speaker and text:
                    dialogue_list.append(f"{speaker}: {text}")
            dialogue_str = "\n".join(dialogue_list)

            # --- Implementation note ---
            target_persona = curr_session.get('dialogue_persona', {}).get('extracted', [])

            # --- Implementation note ---
            updated_memory = {
                "episodic_memory": [],
                "recommendation_outcomes": []
            }

            if i + 1 < len(sessions):
                curr_mem = curr_session.get('memory', {})
                next_mem = sessions[i+1].get('memory', {})

                # 1) Implementation note
                curr_episodic = curr_mem.get('episodic_memory', [])
                next_episodic = next_mem.get('episodic_memory', [])
                updated_memory["episodic_memory"] = [
                    m for m in next_episodic if m not in curr_episodic
                ]

                # 2) Implementation note
                curr_outcomes = curr_mem.get('recommendation_outcomes', [])
                next_outcomes = next_mem.get('recommendation_outcomes', [])
                
                # Implementation note.
                curr_item_names = {obj.get('item') for obj in curr_outcomes}
                
                # Implementation note.
                updated_memory["recommendation_outcomes"] = [
                    obj for obj in next_outcomes if obj.get('item') not in curr_item_names
                ]
            else:
                # Implementation note.
                continue 

            # --- [4] Response JSON build ---
            response_json = {
                "dialogue_persona": target_persona,
                "memory": updated_memory
            }
            # response_str = json.dumps(response_json, ensure_ascii=False, indent=4)
            response_str = json.dumps(response_json, ensure_ascii=False, separators=(",", ":"))

            # --- Implementation note ---
            formatted_user_prompt = USER_PROMPT.format(day=day_str, dialogue=dialogue_str)
            formatted_assistant_prompt = ASSISTANT_PROMPT.format(response=response_str)

            instance = {
                "id": f"{user_id}_{i+1}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": formatted_user_prompt.strip()},
                    {"role": "assistant", "content": formatted_assistant_prompt.strip()}
                ]
            }
            dataset.append(instance)

    return dataset

import json

# def make_persona_and_memory_update(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT, data):
#     """
# Implementation note. 
# Implementation note.
#     """
#     dataset = []

#     for user_id, sessions in data.items():
#         for s_id, session in enumerate(sessions):
#             # --- [1] Current Long-term State (Before Update) ---
#             curr_beh = session['behavioral_persona']['before_update']
#             curr_dia = session['dialogue_persona']['before_update']
            
#             raw_curr_mem = session['memory']['before_update']
# Implementation note.
#             current_memory = {
#                 "session_events": raw_curr_mem.get("episodic_memory", []),
#                 "recommendation_history": raw_curr_mem.get("recommendation_outcomes", [])
#             }

#             # --- [2] New Session Data (Current Extraction) ---
#             new_beh = session['behavioral_persona']['current_extraction']
#             new_dia = session['dialogue_persona']['current_extraction']
            
#             raw_new_mem = session['memory']['current_extraction']
#             new_memory = {
#                 "session_events": raw_new_mem.get("episodic_memory", []),
#                 "recommendation_history": raw_new_mem.get("recommendation_outcomes", [])
#             }

#             # --- [3] Target Response (After Update) ---
#             target_beh = session['behavioral_persona']['after_update']
#             target_dia = session['dialogue_persona']['after_update']
            
#             raw_target_mem = session['memory']['after_update']
#             target_memory = {
#                 "episodic_memory": raw_target_mem.get("episodic_memory", []),
#                 "recommendation_history": raw_target_mem.get("recommendation_outcomes", [])
#             }

# Implementation note.
#             response_json = {
#                 "updated_behavioral_persona": target_beh,
#                 "updated_dialogue_persona": target_dia,
#                 "updated_memory": target_memory
#             }


#             formatted_prompt = [{'role' : 'system', "content" : SYSTEM_PROMPT},
#             {'role': 'user' , 'content' : USER_PROMPT.format(
#                 current_behavioral_persona=json.dumps(curr_beh, ensure_ascii=False),
#                 current_dialogue_persona=json.dumps(curr_dia, ensure_ascii=False),
#                 current_memory=json.dumps(current_memory, ensure_ascii=False),
#                 new_behavioral_persona=json.dumps(new_beh, ensure_ascii=False),
#                 new_dialogue_persona=json.dumps(new_dia, ensure_ascii=False),
#                 new_memory=json.dumps(new_memory, ensure_ascii=False))}, 
#             {'role' : 'assistant', 'content' : ASSISTANT_PROMPT.format(response=json.dumps(response_json, ensure_ascii=False, indent=4))}]
                                    
#             # dataset.append(formatted_prompt)
# --- Implementation note ---
#             instance = {
#                 "id": f"{user_id}_{s_id+1}",
#                 "messages": formatted_prompt
#             }
            
#             dataset.append(instance)
            
#     return dataset
import json

def make_persona_and_memory_update(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT, data):
    """
    Use the current session as the persona update target, 
    and use only the next session as the memory update target.
    """
    dataset = []

    for user_id, sessions in data.items():
        # Exclude the last session because next-session memory is required
        for s_id in range(len(sessions) - 1):
            curr_session = sessions[s_id]
            next_session = sessions[s_id + 1] # Next session used to retrieve the memory answer

            # --- Implementation note ---
            curr_beh = curr_session['behavioral_persona']['before_update']
            curr_dia = curr_session['dialogue_persona']['before_update']
            
            # [Current session data]
            raw_curr_mem = next_session['memory']['before_update']
            current_memory = {
                "session_events": raw_curr_mem.get("episodic_memory", []),
                "recommendation_history": raw_curr_mem.get("recommendation_outcomes", [])
            }

            # --- Implementation note ---
            new_beh = curr_session['behavioral_persona']['current_extraction']
            new_dia = curr_session['dialogue_persona']['current_extraction']
            
            # [Current session data]
            raw_new_mem = next_session['memory']['current_extraction']
            new_memory = {
                "session_events": raw_new_mem.get("episodic_memory", []),
                "recommendation_history": raw_new_mem.get("recommendation_outcomes", [])
            }

            # --- [3] Target: update result (answer) ---
            target_beh = curr_session['behavioral_persona']['after_update']
            target_dia = curr_session['dialogue_persona']['after_update']
            
            raw_target_mem = next_session['memory']['after_update'] 
            target_memory = {
                "episodic_memory": raw_target_mem.get("episodic_memory", []),
                "recommendation_history": raw_target_mem.get("recommendation_outcomes", [])
            }

            response_json = {
                "updated_behavioral_persona": target_beh,
                "updated_dialogue_persona": target_dia,
                "updated_memory": target_memory
            }

            # --- Implementation note ---
            formatted_prompt = [
                {'role': 'system', "content": SYSTEM_PROMPT.strip()},
                {'role': 'user', 'content': USER_PROMPT.format(
                    current_behavioral_persona=json.dumps(curr_beh, ensure_ascii=False),
                    current_dialogue_persona=json.dumps(curr_dia, ensure_ascii=False),
                    current_memory=json.dumps(current_memory, ensure_ascii=False),
                    new_behavioral_persona=json.dumps(new_beh, ensure_ascii=False),
                    new_dialogue_persona=json.dumps(new_dia, ensure_ascii=False),
                    new_memory=json.dumps(new_memory, ensure_ascii=False)
                )}, 
                {'role': 'assistant', 'content': ASSISTANT_PROMPT.format(
                    response=json.dumps(response_json, ensure_ascii=False, separators=(",", ":"))
                )}
            ]
                                    
            instance = {
                "id": f"{user_id}_{s_id+1}",
                "messages": formatted_prompt
            }
            
            dataset.append(instance)
            
    return dataset

def make_listwise_ranking(prompt_dict: Dict[str, str],
                                 candidates,
                                 id2_items,
                                 data,
                                 data_split: str,
                                 dialogue_persona_skip_first: int = 0):
   
    candidates_all_users = candidates.get(data_split, {})
    dataset = []
    alphabet_ids = list(string.ascii_uppercase)

    def get_item_profile(item):
        raw_keywords = item.get("keywords", "")
        if raw_keywords:
            kw_list = [k.strip() for k in raw_keywords.split(",")]
            top_5_keywords = ", ".join(kw_list[:5])
        else:
            top_5_keywords = "Unknown"

        return {
            "title": item.get("title", "Unknown"),
            "content_type": item.get("content_type", "Unknown"),
            "genres": item.get("genres", "Unknown"),
            "description": item.get("description", ""),
            "keywords": top_5_keywords,
            "release_year": item.get("release_year", "Unknown"),
            "countries": item.get("countries", "Unknown"),
            "actors": item.get("actors", "Unknown"),
            "directors": item.get("directors", "Unknown")
        }

    for user_id, val in data.items():
        if user_id not in candidates_all_users:
            continue
           
        user_candidate_sessions = candidates_all_users[user_id]

        for session_idx, session in enumerate(val):
            session_id = str(session_idx + 1)
           
            # Find candidate information for the session
            target_candidate_info = next((item for item in user_candidate_sessions if str(item["session_id"]) == session_id), None)
           
            if not target_candidate_info:
                continue
           
            target_id = str(target_candidate_info['target_id'])
            # Use the top50 list instead of the previous top19
            top50_list = [str(mid) for mid in target_candidate_info.get('top50', [])]
           
            # 1. Skip sessions whose 50 candidates do not include the target
            if target_id not in top50_list:
                continue

            # 2. Split into four windows of 20 items each with 10-item overlap
            windows = [
                top50_list[0:20],
                top50_list[10:30],
                top50_list[20:40],
                top50_list[30:50]
            ]

            # Extract persona and dialogue once because they are session-level data
            behavior_persona = session.get('behavioral_persona', {})
            behavior_persona = {key: val for key, val in behavior_persona.items() if key != 'historical_stats'}
            dialogue_persona = drop_initial_items(
                session.get('dialogue_persona', {}).get('input', []),
                dialogue_persona_skip_first,
            )

            dia = ''
            dialogue = session.get('dialogue', [])
            last_turn = 0
            find = 0
            for turn_num, turn in enumerate(dialogue):
                if 'tagged_items' in turn:
                    for item in turn['tagged_items']:
                        if 'target' in item and item['target']:
                            last_turn = turn_num
                            find = 1
                            break
                if find: break

            for turn_num, turn in enumerate(dialogue[:last_turn]):
                speaker = turn['speaker']
                dia += f"{speaker}: {turn['text'].strip()}\n"

            # 3. Iterate over each window
            for w_idx, current_window in enumerate(windows):
                # Copy the list to protect the original
                window_items = current_window.copy()

                # If the window lacks the target, replace the last item with the target to force insertion
                if target_id not in window_items:
                    window_items[-1] = target_id
               
                # Convert item profiles
                current_pool = []
                for mid in window_items:
                    if mid in id2_items:
                        current_pool.append(get_item_profile(id2_items[mid]))
               
                # 4. Shuffle order
                random.shuffle(current_pool)
                final_candidates = current_pool[:20]

                # Build the dictionary and answer mapping for prompt construction
                candidate_dict = {}
                item_to_mid = {}
                for i, item_data in enumerate(final_candidates):
                    char_id = alphabet_ids[i]
                    candidate_dict[char_id] = item_data
                    item_to_mid[(item_data['title'], str(item_data['genres']))] = char_id

                target_info = get_item_profile(id2_items[target_id])
                target_key = (target_info['title'], str(target_info['genres']))
               
                if target_key in item_to_mid:
                    response_str = item_to_mid[target_key]
                    format_args = {"Movie_list": candidate_dict}

                    if 'Dialogue_history' in prompt_dict.get('user', ''):
                        format_args["Dialogue_history"] = dia
                    if 'Behavioral_persona' in prompt_dict.get('user', ''):
                        format_args["Behavioral_persona"] = behavior_persona
                    if 'Dialogue_persona' in prompt_dict.get('user', ''):
                        format_args["Dialogue_persona"] = dialogue_persona

                    try:
                        formatted_prompt = [
                            {"role": "system", "content": prompt_dict['system']},
                            {"role": "user", "content": prompt_dict['user'].format(**format_args)},
                            {"role": "assistant", "content": prompt_dict['agent'].format(response=response_str)}
                        ]
                       
                        dataset.append({
                            # Append the window number to the original session id to ensure uniqueness, e.g. 101187_1_w1
                            "id": f"{user_id}_{session_id}_w{w_idx+1}",
                            "messages": formatted_prompt
                        })
                    except KeyError as e:
                        print(f"Formatting error for {user_id}_{session_id}_w{w_idx+1}: {e}")
                        assert False
                       
    return dataset
