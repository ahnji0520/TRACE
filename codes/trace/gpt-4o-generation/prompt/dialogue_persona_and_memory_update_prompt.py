DIALOGUE_PERSONA_AND_MEMORY_EXTRACT = """
### [System Role]
You are an expert linguistic analyst specializing in user modeling for Conversational Recommender Systems (CRS). Your task is to extract a "Dialogue Persona" (as a list of descriptive statements) and "Memory" from a given conversation.

### [Definitions]
1. Dialogue Persona (List of Strings): 
A dialogue persona is a holistic representation of user expression formed within the conversational context. It integrates not only linguistically expressed attributes such as identity, attitude, and speaking style, but also user-related information that can be observed through dialogue, including states, preferences, and factual information.

2. Memory:
- Session Event: Specific real-life events or contexts the user mentioned happening to them on this day.
- Recommendation Outcomes (List of Dictionaries): A record of all items recommended by the system during the session. Each entry must include:
    * item: The title of the movie or series recommended.
    * reaction: The user's response type. Choose from:
        - Accept: User shows clear intent to watch.
        - Reject: User clearly dislikes the item or its attributes.
        - Soft Reject: User acknowledges the item is good or fits their taste, but declines for now due to current mood, timing, or situational factors.
        - Already Seen: User has already watched the item.
        - Neutral: Ambiguous response or lack of clear feedback.
    * reason: The specific linguistic or preferential reason given by the user for their reaction.

### [Instructions]
Step 1: Extract Dialogue Persona. Synthesize these into a list of clear, independent descriptive sentences. 
Step 2: Extract Session Event. Identify the real-world context or event the user is experiencing today.
Step 3: Analyze all Recommendation Outcomes. Identify every item the system suggested. For each item, determine the reaction type and provide the specific reasoning, paying close attention to whether a rejection is based on taste (Reject) or situational timing (Soft Reject).
Step 4: Reflection & Refinement(Filter out specific items): Review Dialogue Persona and filter out reaction about specific items(movies or serise)
Step 5: Reflection & Refinement(Filter out daily events): Review Dialogue Persona and filter out one-time occurrences(what happened to them in this day) and unless they demonstrate a persistent pattern or an enduring trait.


### [Constraints]
- Use the language of the provided dialogue for the output.
- Each string in the dialogue_persona list must be a single, concise sentence.
- Never use time-specific adverbs like 'today', 'now' in the Dialogue Persona.
- NO Specific Titles in Dialogue Persona: NEVER include movie or series titles in the Dialogue Persona. Move the specific item data to Memory.
- NO Episodic Events in Dialogue Persona: Do not include "what happened today" in the persona. Move all situational contexts to Memory.


### [Output Format]
Produce a JSON object with the following structure:
{{
  "dialogue_persona": [
    "",
    ""
  ],
  "memory": {{
    "session_event": "[{day}] What happened to the user today?",
    "recommendation_outcomes": [
      {{
        "item": "Movie Title A",
        "reaction": "Accept / Reject / Soft Reject / Already Seen / Neutral",
        "reason": "Detailed reason"
      }}
    ]
  }}
}}

Input Dialogue (Date: {day}):
{dialogue_json}
"""


DIALOGUE_PERSONA_UPDATE = """
### [System Role]
You are a professional data analyst synthesizing and evolving a long-term "Dialogue Persona" list for a recommendation system.

### [Definition]
A dialogue persona is a holistic representation of user expression formed within the conversational context. It integrates not only linguistically expressed attributes such as identity, attitude, and speaking style, but also user-related information that can be observed through dialogue, including states, preferences, and factual information."

### [Instructions]
Step1: You will be given the "Current Persona (List)" and the "New Session Persona (List)." Synthesize them into an updated list of strings based on the following evolution rules:
1. Expand & Add (New Information): If the new persona contains attributes or facts not present in the current persona, add them to the list.
2. Specify & Deepen (Reinforcement): If new info aligns with existing attributes but adds more detail, update the existing string to be more specific and descriptive.
3. Prioritize Recency (Conflict Resolution): If new info directly contradicts the current persona (e.g., change in speaking style or emotional state), replace the old string with the new one to reflect the user's latest state.

Step 2: Reflection & Refinement(Filter out specific items): Review Dialogue Persona and filter out reaction about specific items(movies or serise)
Step 3: Reflection & Refinement(Filter out daily events): Review Dialogue Persona and filter out one-time occurrences(what happened to them in this day) and unless they demonstrate a persistent pattern or an enduring trait.

### [Output Format]
Produce a JSON object with the following structure:
{{
  "updated_dialogue_persona": [
    "string 1",
    "string 2",
    ...
  ]
}}

### [Constraints]
- Immutable Statements (Do NOT Modify/Delete): The following sentences are core traits and must be preserved exactly as they are in the final list:
    1. "The user has high/low clarity of their preferences."
    2. "The user shows high/low initiative and proactiveness in conversation."
    3. "The user provides feedback in a indirect/direct manner."
- Each string must be a single sentence starting directly with the subject.
- Ensure all factual identity information is preserved unless explicitly contradicted.

Input Data:
1. Current Persona: {current_dialogue_persona}
2. New Session Persona: {new_session_persona}
"""