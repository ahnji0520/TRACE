import json
from sklearn.model_selection import train_test_split
import argparse
import os
from pathlib import Path

SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[4]

# Listwise Ranking Chat Template prompts
from prompt import *
from utils import (
    make_behavioral_persona_extract,             
    make_dialogue_persona_and_memory_extract,      
    make_persona_and_memory_update,
    make_listwise_ranking,
)


def split_train_test(data, train_test_valid_name_tag):
    train_ids, test_ids, val_ids = train_test_valid_name_tag['train'],train_test_valid_name_tag['test'], train_test_valid_name_tag['valid'] 

    train_data = {uid: data[uid] for uid in train_ids}
    val_data   = {uid: data[uid] for uid in val_ids}
    test_data  = {uid: data[uid] for uid in test_ids}

    return train_data, val_data, test_data

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_list, filename):
    # 1. Extract only the folder path from the file path
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 2. Save file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(file_list, f, ensure_ascii=False, indent=4)
    print(f"Saved successfully: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run a task with a JSON input file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--task", type=str, required=True, help="Task name to run (e.g., stats, print_keys, etc.)")
    parser.add_argument("--candidate_path", type=str, required=False, 
                            default=str(SUPPLEMENTARY_ROOT / "data/candidate_curator/top-k_candidates/candidates.json"),
                            help="Path to the candidate JSON file; only needed when the task is `ranking`")
    parser.add_argument("--id2item_path", type=str, required=False, default=str(SUPPLEMENTARY_ROOT / "data/trace/mts-kion_processed/id2items_in_dialogue.json"),
                            help="Path to the item's id-info mapped JSON file; only needed when the task is `ranking`")
    parser.add_argument("--memory_path", type=str, required=False, 
                        default=str(SUPPLEMENTARY_ROOT / "data/evocrs/persona_processor/selected_memories.json"),
                        help="Path to the user session memories JSON file")
    parser.add_argument("--cot_path", type=str, default=str(SUPPLEMENTARY_ROOT / "data/qwq_cot/cot_outputs.json"))
    args = parser.parse_args()
    print(args)

    data = load_json(args.json_path)
    train_test_val_name_json = load_json(SUPPLEMENTARY_ROOT / "data/evocrs/user_ids.json")

    task = args.task
    train, val, test = split_train_test(data, train_test_val_name_json)

    if task == 'behavioral_persona_extract':
        train = make_behavioral_persona_extract(BEHAVIORAL_PERSONA_EXTRACTION_SYSTEM_PROMPT,BEHAVIORAL_PERSONA_EXTRACTION_USER_PROMPT, BEHAVIORAL_PERSONA_EXTRACTION_ASSISTANT_PROMPT,  train)
        val = make_behavioral_persona_extract(BEHAVIORAL_PERSONA_EXTRACTION_SYSTEM_PROMPT,BEHAVIORAL_PERSONA_EXTRACTION_USER_PROMPT, BEHAVIORAL_PERSONA_EXTRACTION_ASSISTANT_PROMPT, val)
        test = make_behavioral_persona_extract(BEHAVIORAL_PERSONA_EXTRACTION_SYSTEM_PROMPT,BEHAVIORAL_PERSONA_EXTRACTION_USER_PROMPT, BEHAVIORAL_PERSONA_EXTRACTION_ASSISTANT_PROMPT, test)
        
    elif task == 'dialogue_persona_extract':
        train = make_dialogue_persona_and_memory_extract(DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_SYSTEM_PROMPT, DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_USER_PROMPT, DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_ASSISTANT_PROMPT, train)
        val = make_dialogue_persona_and_memory_extract(DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_SYSTEM_PROMPT, DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_USER_PROMPT, DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_ASSISTANT_PROMPT, val)
        test = make_dialogue_persona_and_memory_extract(DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_SYSTEM_PROMPT, DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_USER_PROMPT, DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_ASSISTANT_PROMPT, test)
        
    elif task == 'persona_update':
        train = make_persona_and_memory_update(PERSONA_AND_MEMORY_UPDATE_SYSTEM_PROMPT,PERSONA_AND_MEMORY_UPDATE_USER_PROMPT,PERSONA_AND_MEMORY_UPDATE_ASSISTANT_PROMPT , train)
        val = make_persona_and_memory_update(PERSONA_AND_MEMORY_UPDATE_SYSTEM_PROMPT,PERSONA_AND_MEMORY_UPDATE_USER_PROMPT,PERSONA_AND_MEMORY_UPDATE_ASSISTANT_PROMPT , val)
        test = make_persona_and_memory_update(PERSONA_AND_MEMORY_UPDATE_SYSTEM_PROMPT,PERSONA_AND_MEMORY_UPDATE_USER_PROMPT,PERSONA_AND_MEMORY_UPDATE_ASSISTANT_PROMPT , test)

    elif 'listwise_ranking' in task.lower():
        candidates = load_json(args.candidate_path)
        id2item = load_json(args.id2item_path)

        prompt_map = {
            "v13": {
                "system": CRS_RANKING_LISTWISE_DEFAULT_SYSTEM,
                "user": CRS_RANKING_LISTWISE_DEFAULT_USER,
                "agent": CRS_RANKING_LISTWISE_DEFAULT_AGENT,
            },
            "v14": {
                "system": CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_SYSTEM,
                "user": CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_USER,
                "agent": CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_AGENT,
            },
            "v15": {
                "system": CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_SYSTEM,
                "user": CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_USER,
                "agent": CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_AGENT,
            },
        }

        prompt_dict = next(
            (prompt for version, prompt in prompt_map.items() if version in task.lower()),
            None,
        )

        if prompt_dict is None and 'v16' in task.lower():
            prompt_dict= {
                "system": CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_SYSTEM,
                "user":   CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_USER,
                "agent":     CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_AGENT,
            }
        elif prompt_dict is None:
            raise ValueError("specify the listwise ranking prompt version: v13, v14, v15, or v16")
            
        dialogue_persona_skip_first = 8 if 'drop_initial_dialogue_persona' in task.lower() else 0

        train = make_listwise_ranking(prompt_dict, candidates, id2item, train, 'train', dialogue_persona_skip_first=dialogue_persona_skip_first)
        val = make_listwise_ranking(prompt_dict, candidates, id2item, val, 'valid', dialogue_persona_skip_first=dialogue_persona_skip_first)
        test = make_listwise_ranking(prompt_dict, candidates, id2item, test, 'test', dialogue_persona_skip_first=dialogue_persona_skip_first)



    else:
        print('task:', task)
        assert False

    save_json(train, args.output_dir+f'/{task}_train.json')        
    save_json(val,args.output_dir+f'/{task}_val.json')        
    save_json(test,args.output_dir+f'/{task}_test.json') if test is not None else None
    
if __name__ == "__main__":
    main()
