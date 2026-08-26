import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Import prompts
from src.dataset_generation.prompt.dialogue_persona_and_memory_update_prompt import (
    DIALOGUE_PERSONA_AND_MEMORY_EXTRACT,
    DIALOGUE_PERSONA_UPDATE
)

SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = SUPPLEMENTARY_ROOT / ".env"
load_dotenv(ENV_PATH)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm_json(prompt):
    """Call the OpenAI API and return a JSON response"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[Dialogue Updater] API call failed: {e}")
        return {}

def update_dialogue_persona(current_d_persona, dialogue_json, day):
    """
    Returns:
        updated_persona (list): Combined result from the previous persona and current session traits
        next_session_memory (dict): Memory passed to the next session
        new_session_persona (list): Traits newly extracted from the current session only
    """
    
    # --- [Step 1: Extract persona and memory for this session] ---
    extract_prompt = DIALOGUE_PERSONA_AND_MEMORY_EXTRACT.format(
        day=day,
        dialogue_json=json.dumps(dialogue_json, indent=2, ensure_ascii=False)
    )
    extraction_result = get_llm_json(extract_prompt)
    
    # Raw extraction for this session
    new_session_persona = extraction_result.get("dialogue_persona", [])
    next_session_memory = extraction_result.get("memory", {})

    # --- [Step 2: Compose the long-term dialogue persona] ---
    update_prompt = DIALOGUE_PERSONA_UPDATE.format(
        current_dialogue_persona=json.dumps(current_d_persona, indent=2, ensure_ascii=False),
        new_session_persona=json.dumps(new_session_persona, indent=2, ensure_ascii=False)
    )
    
    update_result = get_llm_json(update_prompt)
    
    # Composed final update
    updated_persona = update_result.get("updated_dialogue_persona", current_d_persona)

    # Return exactly three values to avoid unpacking errors
    return updated_persona, next_session_memory, new_session_persona