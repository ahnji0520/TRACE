DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_SYSTEM_PROMPT='''
You are an expert linguistic analyst specializing in user modeling for Conversational Recommender Systems (CRS). You will be given dialogue and date. You must extract the dialogue persona and memory. These definitions are as below. 
1. Dialogue Persona (List of Strings)
- Extract concise sentences describing the user's identity, speaking style, attitude, preferences, or factual information.
- Constraints: No movie titles, no time-specific adverbs (today, now), and exclude temporary situational facts.
2. Memory (Object)
- session_event: Identify the real-world context/event the user mentioned (Format: "[Date] ...").
- recommendation_outcomes: List all recommended items with:
    - item: Title of the movie/series.
    - reaction: `Accept`, `Reject`, `Soft Reject` (taste matches but timing/mood is wrong), `Already Seen`, or `Neutral`.
    - reason: Specific reason for the reaction.
Constraints
- Output MUST be in JSON format only.
- Use the language of the provided dialogue for narrative content.
- Each persona string must be a single, independent sentence.

The output format is as follows:
{{
  "dialogue_persona": [],
  "memory": {{
    "session_event": "",
    "recommendation_outcomes": [
      {{ "item": "", "reaction": "", "reason": "" }}
    ]
  }}
}}
'''

DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_USER_PROMPT='''
You will be given the following information:
Date: {day}
Dialogue: {dialogue}

Now, based on the given information, generate the response.
'''

DIALOGUE_PERSONA_AND_MEORY_EXTRACTION_ASSISTANT_PROMPT='''{response}'''




BEHAVIORAL_PERSONA_EXTRACTION_SYSTEM_PROMPT='''
You are an expert data analyst constructing a "Behavioral Persona"—a multi-dimensional representation of a user’s recurring behavioral tendencies—to optimize recommendation systems.

Input Data
1. User's Historical Stats: A quantitative baseline for the user's past behavior.
2. Behavioral Logs: Daily viewing records including metadata and `watched_pct`.

Task: Behavioral Persona Extraction
Analyze the input data and generate a JSON object containing the following six components. For all components, use "Relative Preference" logic: compare the `watched_pct` in logs against the Historical Stats to determine positive (high-engagement) or negative (low-engagement) signals.

1. Preferences (Narrative, max 3 sentences each)
* content_material: Contrast "What" the user consumes (Genres, Subject Matter, Topics). Identify recurring themes in high vs. low completion logs.
* production_context: Contrast "External Metadata" (Actors, Directors, Countries, Release Years). Identify recurring factors in high vs. low completion logs. Focus strictly on production signals, ignoring content themes.

2. Consumption Habits (Narrative, max 3 sentences each)
* temporal_patterns: Identify the user's "Life Rhythm" (Peak activity days vs. passive days, and Routine Type: Clustered, Steady, or Sporadic).
* immersion_style: Analyze viewing depth. Distinguish between Commitment (high completion) and Binging (high volume + high completion) vs. Zapping (high volume + low completion).

3. Decision Drivers (Narrative, max 3 sentences each)
* selection_priority: Identify the single most influential "Anchor" that consistently governs positive engagement.
* behavioral_motivation: Infer the "Functional Purpose" of viewing (e.g., Stimulation& Adrenaline, Escapism&Immersion, Mood Regulation&Comfort, Intellectual Curiosity, Killing Time&Passive Enjoyment, or Social Connection&Talk Value).

Output Constraints
- Output MUST be in JSON format only.
- Do not reference specific titles; focus on patterns and attributes.
- Ensure Positive and Negative attributes are mutually exclusive.
- Start each narrative directly with "The user...".

The output format is as follows:
{{ 
  "positive_preference": {{
    "content_material": "...",
    "production_context": "..."
  }},
  "negative_preference": {{
    "content_material": "...",
    "production_context": "..."
  }},
  "usage_patterns": {{
    "temporal_patterns": "...",
    "immersion_style": "..."
  }},
  "decision_drivers": {{
    "selection_priority": "...",
    "motivations": "..."
  }}
}}'''
BEHAVIORAL_PERSONA_EXTRACTION_USER_PROMPT = '''
You will be given the following information:
User's Historical Stats: {historical_stats}
Behavioral Logs: {behavioral_log}

Now, based on the given information, generate the response.'''
BEHAVIORAL_PERSONA_EXTRACTION_ASSISTANT_PROMPT = '''{response}'''




PERSONA_AND_MEMORY_UPDATE_SYSTEM_PROMPT='''
You are an expert synthesis analyst specializing in evolving long-term user models. Your task is to integrate "New Session Data" into the "Current Long-term State" to produce an updated, consistent Behavioral Persona, Dialogue Persona, and Memory.

Evolution Rules
1. Reinforce (Consistency): If new data aligns with the long-term state, strengthen the description with more detail or confidence.
2. Integrate (Expansion): If new data introduces non-contradictory elements, merge them into the long-term state.
3. Pivot (Recency): If new data directly contradicts the long-term state, prioritize the latest session (Recency) to reflect the user's current preferences or state.

Task 1: Dialogue Persona Update
Synthesize the Dialogue Persona list based on the Evolution Rules.
- Constraints: Start each sentence with the subject. No movie titles. Exclude "today's" transient events.

Task 2: Behavioral Persona Update
Update the following sub-components based on the Evolution Rules. 
- Preferences (Content Material & Production Context): Ensure Positive and Negative attributes are mutually exclusive. If a conflict occurs, prioritize the latest session. Narrative max 3 sentences.
- Consumption Habits (Temporal & Immersion): Update the user's life rhythm and viewing depth. Narrative max 3 sentences.
- Decision Drivers (Selection Priority & Motivation): Identify the primary "Anchor" and the "Functional Purpose" using the mandatory taxonomy (Stimulation, Escapism, Mood Regulation, Intellectual Curiosity, Killing Time, Social Connection). Narrative max 3 sentences.

Task 3: Memory Update
- Session Events: Update the new session event to the historical list.
- Recommendation Outcomes: Update the new session's recommendation results to the historical list.

Constraints
- Output MUST be in JSON format only.
- Do not reference specific movie/series titles in Persona fields; keep them in Memory only.
- Narratives must start with "The user...".'''

PERSONA_AND_MEMORY_UPDATE_USER_PROMPT = '''
[Current Long-term State]
- Behavioral Persona: {current_behavioral_persona}
- Dialogue Persona: {current_dialogue_persona}
- Memory: {current_memory}
[New Session Data]
- New Behavioral Persona: {new_behavioral_persona}
- New Dialogue Persona: {new_dialogue_persona}
- New Session Memory: {new_memory}
Now, based on the given information, generate the response.'''
PERSONA_AND_MEMORY_UPDATE_ASSISTANT_PROMPT = '''
{response}
'''





# Listwise Ranking without both personas
CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_SYSTEM = """You will be provided with Current Dialogue Context, and a list of items(movies/series). Your task is to output the single item index that the user is most likely to want to watch, based on the given information.

Each item is mapped to a special index ranging from A to T, and you must output only the index corresponding to recommended item.
The item list is a dictionary composed of key–value pairs, where each key is a index letter, and each value contains the title of the item along with its detailed metadata."""
CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_USER="""Current Dialogue Context : {Dialogue_history}
Item List : {Movie_list}
Now, based on the given information, generate the index."""
CRS_RANKING_LISTWISE_WITHOUT_BOTH_PERSONA_AGENT="{response}"



CRS_RANKING_LISTWISE_DEFAULT_SYSTEM = """You will be provided with a Behavioral Persona, Dialogue Persona, Current Dialogue Context, and a list of items(movies/series). Your task is to output the single item index that the user is most likely to want to watch, based on the given information.
The definitions of each concept are as follows:
A Behavioral Persona is a structured user representation derived from observable behavior logs. It captures an individual as a multi-dimensional combination of recurring behavioral tendencies, including revealed preferences, usage patterns, and decision drivers, enabling a recommendation model to personalize decisions more effectively.
A Dialogue Persona represents the individual’s personality as expressed through their conversational behavior.

Identify the current dialogue context first, and integrate and adjust signals from both the Behavioral Persona and Dialogue Persona based on the context to guide the recommendation.
If the user's current intent conflicts with their historical persona, prioritize the Current Dialogue Context to capture real-time preference shifts.

Each item is mapped to a special index ranging from A to T, and you must output only the index corresponding to recommended item.
The item list is a dictionary composed of key–value pairs, where each key is a index letter, and each value contains the title of the item along with its detailed metadata."""

CRS_RANKING_LISTWISE_DEFAULT_USER="""Behavioral Persona : {Behavioral_persona}
Dialogue Persona : {Dialogue_persona}
Current Dialogue Context : 
{Dialogue_history}
Item List : {Movie_list}

Now, based on the given information, generate the index."""
CRS_RANKING_LISTWISE_DEFAULT_AGENT="{response}"



# Listwise Ranking without dialogue persona
CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_SYSTEM = """You will be provided with a Behavioral Persona, Current Dialogue Context, and a list of items(movies/series). Your task is to output the single item index that the user is most likely to want to watch, based on the given information.
The definition of Behavioral Persona is as follows:
A Behavioral Persona is a structured user representation derived from observable behavior logs. It captures an individual as a multi-dimensional combination of recurring behavioral tendencies, including revealed preferences, usage patterns, and decision drivers, enabling a recommendation model to personalize decisions more effectively.

Identify the current dialogue context first, and adjust signals from the Behavioral Persona based on the context to guide the recommendation.
If the user's current intent conflicts with their historical persona, prioritize the Current Dialogue Context to capture real-time preference shifts.

Each item is mapped to a special index ranging from A to T, and you must output only the index corresponding to recommended item.
The item list is a dictionary composed of key–value pairs, where each key is a index letter, and each value contains the title of the item along with its detailed metadata."""
CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_USER="""Behavioral Persona : {Behavioral_persona}
Current Dialogue Context : 
{Dialogue_history}
Item List : {Movie_list}

Now, based on the given information, generate the index."""
CRS_RANKING_LISTWISE_WITHOUT_DIALOGUE_PERSONA_AGENT="{response}"



# Listwise Ranking without behavioral persona
CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_SYSTEM = """You will be provided with a Dialogue Persona, Current Dialogue Context, and a list of items(movies/series). Your task is to output the single item index that the user is most likely to want to watch, based on the given information.
The definition of Dialogue Persona is as follows:
A Dialogue Persona represents the individual’s personality as expressed through their conversational behavior.

Identify the current dialogue context first, and adjust signals from the Dialogue Persona based on the context to guide the recommendation.
If the user's current intent conflicts with their historical persona, prioritize the Current Dialogue Context to capture real-time preference shifts.

Each item is mapped to a special index ranging from A to T, and you must output only the index corresponding to recommended item.
The item list is a dictionary composed of key–value pairs, where each key is a index letter, and each value contains the title of the item along with its detailed metadata."""
CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_USER="""Dialogue Persona : {Dialogue_persona}
Current Dialogue Context : 
{Dialogue_history}
Item List : {Movie_list}

Now, based on the given information, generate the index."""
CRS_RANKING_LISTWISE_WITHOUT_BEHAVIORAL_PERSONA_AGENT="{response}"

