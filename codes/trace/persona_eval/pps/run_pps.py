import os
import sys
from pathlib import Path
import json
import concurrent.futures
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv

# Implementation note.
SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENTARY_ROOT = SCRIPT_DIR.parents[3]
PROMPT_DIR = SUPPLEMENTARY_ROOT / "codes" / "trace" / "persona_eval" / "pps"
if str(PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPT_DIR))

from pps_prompt import (
    BEHAVIOR_PERSONA_CREDIBILITY,
    BEHAVIOR_PERSONA_CONSISTENCY,
    BEHAVIOR_PERSONA_COMPLETENESS,
    BEHAVIOR_PERSONA_CLARITY,
    BEHAVIOR_PERSONA_IMMERSION
)

# 1. Set environment variables and API settings
ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(f"API key not found. Check the file: {ENV_PATH}")

genai.configure(api_key=API_KEY)

# Implementation note.
model = genai.GenerativeModel(
    'gemini-2.5-pro',
    generation_config={"response_mime_type": "application/json"}
)

# 2. Implementation note
BASE_DIR = SUPPLEMENTARY_ROOT / "data" / "trace" / "persona_eval" / "pps"
FILE_BEHAVIOR = BASE_DIR / "behavioral_persona_sampled.json"
FILE_PREFERENCE = BASE_DIR / "only_preference_sampled.json"
OUTPUT_FILE = SUPPLEMENTARY_ROOT / "data" / "trace" / "eval" / "persona" / "evaluation_results.json"

# Implementation note.
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 3. Implementation note
PROMPTS = {
    "Credibility": BEHAVIOR_PERSONA_CREDIBILITY,
    "Consistency": BEHAVIOR_PERSONA_CONSISTENCY,
    "Completeness": BEHAVIOR_PERSONA_COMPLETENESS,
    "Clarity": BEHAVIOR_PERSONA_CLARITY,
    "Immersion": BEHAVIOR_PERSONA_IMMERSION
}

# 4. Implementation note
def call_gemini_eval(persona_text, metric_prompt):
    """Call the Gemini API and return evaluation results as JSON."""
    full_prompt = f"{metric_prompt}\n\n[Behavioral Persona Input]\n{persona_text}"
    try:
        response = model.generate_content(full_prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"reasoning": f"API Error: {e}", "score": 0}

def evaluate_single_persona(task):
    """Evaluate all five metrics for a single persona."""
    user_id = task['user_id']
    persona_data = task['persona']
    dataset_type = task['dataset_type']
    
    # Implementation note.
    persona_text = json.dumps(persona_data, indent=2, ensure_ascii=False)
    
    evaluation_result = {
        "user_id": user_id,
        "dataset_type": dataset_type,
        "evaluations": {}
    }
    
    # 5 Implementation note
    for metric_name, prompt in PROMPTS.items():
        result = call_gemini_eval(persona_text, prompt)
        evaluation_result["evaluations"][metric_name] = result
        
    return evaluation_result

def load_data(file_path, dataset_type):
    """Load JSON data and build evaluation tasks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [{"user_id": item["user_id"], "persona": item["persona"], "dataset_type": dataset_type} for item in data]
    except Exception as e:
        print(f"[{dataset_type}] Failed to load file: {e}")
        return []

def main():
    print("Loading data...")
    behavior_tasks = load_data(FILE_BEHAVIOR, "behavioral_persona")
    preference_tasks = load_data(FILE_PREFERENCE, "only_preference")
    all_tasks = behavior_tasks + preference_tasks
    
    if not all_tasks:
        print("No data to evaluate. Check file paths and sampling outputs.")
        return

    print(f"Starting evaluation for {len(all_tasks)} personas. Max workers: 50")
    
    results = []
    
    # 5. Implementation note
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # Implementation note.
        futures = [executor.submit(evaluate_single_persona, task) for task in all_tasks]
        
        # Implementation note.
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            results.append(res)
            print(f"Progress: {i} / {len(all_tasks)} completed (User ID: {res['user_id']})", end='\r')
            
    print("\n\nAll evaluations completed.")
    
    # 6. Implementation note
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Detailed evaluation results saved: {OUTPUT_FILE}")
    
    # 7. Implementation note
    scores_by_metric = defaultdict(list)
    scores_by_type = defaultdict(lambda: defaultdict(list))
    
    for res in results:
        dtype = res["dataset_type"]
        for metric, eval_data in res["evaluations"].items():
            score = eval_data.get("score", 0)
            if isinstance(score, int) and score > 0: # Implementation note.
                scores_by_metric[metric].append(score)
                scores_by_type[dtype][metric].append(score)
                
    # --- Implementation note ---
    print("\n" + "="*50)
    print("Overall Average Scores")
    print("="*50)
    for metric, scores in scores_by_metric.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f" - {metric:<15}: {avg:.2f} / 5.0 (evaluated samples: {len(scores)})")

    print("\n" + "-"*50)
    print("Average Scores by Dataset Type")
    print("-"*50)
    for dtype, metrics in scores_by_type.items():
        print(f"\n[{dtype}]")
        for metric, scores in metrics.items():
            avg = sum(scores) / len(scores) if scores else 0
            print(f"   - {metric:<15}: {avg:.2f}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()