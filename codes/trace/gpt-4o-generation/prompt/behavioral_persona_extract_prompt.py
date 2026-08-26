CONTENT_MATERIAL_EXTRACT = """
You are an expert data analyst constructing a Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative) - subdivided into Content Material and Production Context.
2.  Consumption Habits
3.  Decision Driver

Your Current Task:
Focus exclusively on the "Content Material" sub-component of Preferences. You must analyze "What" the user consumes or avoids by contrasting high-engagement and low-engagement logs.

Input Data Context:
1. User's Historical Completion Stats
  - Note: The provided Historical Stats already include the New Behavior Logs.
2. Behavior Logs
	- Relative Preference: Interpret completion rates relatively based on User's Historical Completion Stats; treat items with relatively high completion rates as evidence to strengthen/expand Positive Preferences, and relatively low rates for Negative Preferences.
- Even if the provided history is limited, you must infer the behavioral patterns based on the available records. Do not omit any fields or state that information is insufficient; provide the most plausible analysis derived from the existing logs while ensuring all sections are fully populated.

Step 1: Identify Content Material Elements
Examine the behavior logs to extract three core elements for both positive and negative preferences: 
1) Genres (the broad category), 
2) Subject Matter (tangible objects, settings, character types), 
3) Topics (core messages, ideologies, or worldviews). 
Focus strictly on "What" is being consumed, identifying the recurring ingredients across the logs.

Step 2: Resolve Metadata Conflicts
Extract Positive attributes only from high-completion logs and Negative attributes only from low-completion logs; do not reference the other set in wording (no “relative to” framing). Use contrast only to validate consistency.
If an attribute appears in both sets, (a) split into non-overlapping context facets when supported by logs; otherwise (b) treat as ambivalent and drop it from both. Ensure Positive and Negative are mutually exclusive (no duplicates).

Step 3: Generate Narrative Output
Synthesize the findings into two distinct narrative paragraphs (one for positive, one for negative). 
Do not use comparative phrases like "instead of," "rather than," or "prefers A over B." 
Avoid referencing specific titles.
To keep the response concise, produce each paragraph in no more than three sentences, starting directly with the subject.


[Example Output]
{{
  "positive_content_material": "The user ...",
  "negative_content_material": "The user ..."
}}

Now, based on the information provided below, extract the Preference - Content Material.

Input Data: 
1. User's Historical Completion Stats: {user_completion_stats}
2. Behavior Logs: {log_data}

Output (JSON only)

"""

PRODUCTION_CONTEXT_EXTRACT = """
You are an expert data analyst constructing a Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative) - subdivided into Content Material and Production Context.
2.  Consumption Habits
3.  Decision Driver

Your Current Task:
Focus exclusively on the "Production Context" sub-component of Preferences. You must analyze the "External Metadata" (Who, When, Where) by contrasting high-engagement and low-engagement logs.

Input Data Context:
1. User's Historical Completion Stats
  - Note: The provided Historical Stats already include the New Behavior Logs.
2. Behavior Logs
	- Relative Preference: Interpret completion rates relatively based on User's Historical Completion Stats; treat items with relatively high completion rates as evidence to strengthen/expand Positive Preferences, and relatively low rates for Negative Preferences.
- Even if the provided history is limited, you must infer the behavioral patterns based on the available records. Do not omit any fields or state that information is insufficient; provide the most plausible analysis derived from the existing logs while ensuring all sections are fully populated.

Step 1: Identify Production Metadata Elements
Examine the behavior logs to identify recurring signals across four elements: 
1) Actors (recurring names), 
2) Directors (filmmakers whose works are consistently completed or abandoned), 
3) Countries (preferred or avoided production origins), 
4) Release Year (favored or avoided eras/time periods). 
Focus strictly on these metadata fields and COMPLETELY IGNORE the content material(genre, subject matter, topic).

Step 2: Resolve Metadata Conflicts
Extract Positive attributes only from relatively high-completion logs and Negative attributes only from relatively low-completion logs; do not reference the other set in wording (no “relative to” framing). Use contrast only to validate consistency.
If an attribute appears in both sets, (a) split into non-overlapping context facets when supported by logs (e.g., era, country, collaborator); otherwise (b) treat as ambivalent and drop it from both. Ensure Positive and Negative are mutually exclusive (no duplicates).


Step 3: Generate Narrative Output
Synthesize the findings into two distinct narrative paragraphs (one for positive, one for negative). 
Do not use comparative phrases like "instead of," "rather than," or "prefers A over B." 
Avoid referencing specific titles.
To keep the response concise, produce each paragraph in no more than three sentences, starting directly with the subject.


[Example Output]
{{
  "positive_production_context": "The user ...",
  "negative_production_context": "The user ..."
}}

Now, based on the information provided below, extract the Positive/Negative Preference - Production Context.

Input Data: 
1. User's Historical Completion Stats: {user_completion_stats}
2. Behavior Logs: {log_data}

Output (JSON only):
"""

TEMPORAL_PATTERNS_EXTRACT = """
You are an expert data analyst constructing a Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative)
2.  Consumption Habits (Temporal patterns, Immersion style)
3.  Decision Drivers

Your Current Task:
Focus exclusively on extracting the "Consumption Habits - Temporal Patterns" component.
You must analyze the user's "Life Rhythm" based on the viewing history.

Input Data Context:
1. User's Historical Completion Stats
  - Note: The provided Historical Stats already include the New Behavior Logs.
2. Behavior Logs
	- The input is a dictionary where Keys are Dates. Each Value is an object containing:
		* `day`: The string representing the day of the week.
		* `watched_pct`: A list of completion percentages. 
	- Relative Activeness: Evaluate the `watched_pct` based on the user's historical stats. "Active" viewing is defined by both Quantity and Quality: A specific day is considered "Active" only if the list is long (many items watched) AND the values are high (high completion rates).
- Even if the provided history is limited, you must infer the behavioral patterns based on the available records.

Step 1: Analyze Frequency & Schedule
Using the Dates, Days, and the definition of Active viewing above, identify lifestyle patterns:
1.  Peak Activity Days: Identify which specific days consistently show the most "Active" behavior (high volume + high completion), distinguishing them from days with only light or passive usage.
2.  Routine Type: Determine the overall rhythm. Is it Clustered (heavy activity only on specific days like weekends), Steady (consistent, moderate viewing spread evenly across most days), or Sporadic (random/unpredictable gaps)?

Step 2: Generate Narrative Output
Synthesize these findings into a narrative paragraph.
Do not list the items. Describe the user's routine naturally across the observed dates.
Avoid referencing specific titles.
To keep the response concise, you must produce the answer in no more than three sentences.

[Example Output]
{{
  "temporal_patterns": "The user ..."
}}

Now, based on the information provided below, extract the Consumption Habits - Temporal Patterns.

Input Data: 
1. User's Historical Stats: {user_stats}
2. Behavior Logs: {log_data}

Output (JSON only):
"""

IMMERSION_STYLE_EXTRACT = """
You are an expert data analyst constructing a Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative)
2.  Consumption Habits (Temporal patterns, Immersion style)
3.  Decision Drivers

Your Current Task:
Focus exclusively on extracting the "Consumption Habits - Immersion Style" component.
You must analyze the user’s day-level viewing depth based on the viewing history.

Input Data Context:
1. User's Historical Completion Stats
  - Note: The provided Historical Stats already include the New Behavior Logs.
2. Behavior Logs
	- The input is a dictionary where Keys are Dates. Each Value is an object containing:
		* `day`: The string representing the day of the week.
		* `watched_pct`: A list of completion percentages. 
- Relative Immersion: Evaluate the `watched_pct` based on the user's historical stats. 
- Even if the provided history is limited, you must infer the behavioral patterns based on the available records.

Step 1: Analyze Immersion Metrics
Examine the `watched_pct` lists to identify viewing behaviors:
1. Commitment Level: Check if completion is typically high relative to the user’s historical completion stats (e.g., at/above their historical average or upper-range), or if there are frequent low-relative drop-offs.
2.  Binging: Analyze the volume of completed items. "True binging" means a high number of titles in a day with consistently high completion rates. A high number of titles with low completion rates indicates zapping/indecisive browsing.

Step 2: Generate Narrative Output
Synthesize these findings into a narrative paragraph.
Do not list the items. Describe the user's viewing style naturally.
Avoid referencing specific titles.
To keep the response concise, you must produce the answer in no more than three sentences.

[Example Output]
{{
  "immersion_style": "The user ..."
}}

Now, based on the information provided below, extract the Consumption Habits - Immersion Style.

Input Data: 
1. User's Historical Stats: {user_stats}
2. Behavior Logs: {log_data}

Output (JSON only):
"""

SELECTION_PRIORITY_EXTRACT = """
You are an expert data analyst constructing a Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Content Material, Production Context)
2.  Consumption Habits
3.  Decision Drivers (Selection Priority, Behavioral Motivation)

Your Current Task:
Focus exclusively on extracting the "Decision Drivers - Selection Priority" component.
You must identify the Hierarchy of Choice by uncovering the single most influential factor (the "Anchor") that governs the user's positive engagement. 

Input Data Context:
1. User's Historical Completion Stats
  - Note: The provided Historical Stats already include the New Behavior Logs.
2. Behavior Logs
	- Relative Preference: Interpret completion rates relatively based on User's Historical Completion Stats; treat items with relatively high completion rates as evidence to strengthen/expand Positive Preferences, and relatively low rates for Negative Preferences.
- Consistency Rule: Summarized Preferences are also provided as reference. Your analysis must ensure a seamless logical connection between the behavior logs and these preferences, ensuring the final output does not conflict with either data source.
- Even if the provided history is limited, you must infer the behavioral patterns based on the available records.

Step 1: Analyze the Raw Behavior Logs
Examine the high-engagement items (relatively high completion vs. the user's historical completion stats) to identify the primary anchor that remains consistent across them across sessions.

Step 2: Cross-check with the Reference Summary
Verify the identified anchor against the confirmed Positive and Negative Preferences. Ensure the final output does not logically conflict with the provided Positive/Negative Preferences.

Step 3: Generate Narrative Output
Synthesize the findings into a narrative paragraph.
Do not list the items. Describe the user's selection priority naturally.
Avoid referencing specific titles.
To keep the response concise, you must produce the answer in no more than three sentences.

[Example Output]
{{
  "selection_priority": "The user ..."
}}

Now, based on the provided Input Data, extract the Decision Drivers - Selection Priority.

Input Data: 
1. Preference: {preference}
2. User's Historical Completion Stats: {user_completion_stats}
3. Behavior Logs: {log_data}

Output (JSON only):
"""

MOTIVATIONS_EXTRACT = """
You are an expert data analyst constructing a Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences
2.  Consumption Habits
3.  Decision Drivers (Selection Priority, Behavioral Motivation)

Your Current Task:
Focus exclusively on extracting the "Decision Drivers - Behavioral Motivation" component based on the viewing history.
You must infer the "Functional Purpose" (The implicit goal or reward) behind the user's viewing sessions.

Input Data Context:
1. User's Historical Completion Stats
  - Note: The provided Historical Stats already include the New Behavior Logs.
2. Behavior Logs
	- Relative Preference: Interpret completion rates relatively based on User's Historical Completion Stats; treat items with relatively high completion rates as evidence to strengthen/expand Positive Preferences, and relatively low rates for Negative Preferences.
- Consistency Rule: Summarized Preferences are also provided as reference. Your analysis must ensure a seamless logical connection between the behavior logs and these preferences, ensuring the final output does not conflict with either data source.
- Even if the provided history is limited, you must infer the behavioral patterns based on the available records.

Step 1: Map Content Patterns to Functional Motivations
Analyze the nature of the high-engagement content (relatively high completion vs. the user's historical completion stats) to deduce the user's goal. DO NOT discuss "Criteria" (Selection Priority). FOCUS on "Utility" (Motivation). Use the following Inference Logic:

1. Stimulation & Adrenaline (Visceral Reward):
   - Signal: High engagement with Action/Thriller/Horror/Racing genres marked by chase/fight/violence/high-stakes cues and higher age_rating.
   - Inference: The user seeks immediate excitement, tension, and sensory stimulation.
2. Escapism & Immersion (Imaginative Reward):
   - Signal: High engagement with Fantasy/Sci-Fi/Adventure genres marked by world-building cues (alternate worlds, magic/aliens, quests, lore) and a relatively high watched_pct.
   - Inference: The user seeks to detach from reality and immerse themselves in alternative universes.
3. Mood Regulation & Comfort (Emotional Reward):
   - Signal: High engagement with Family/Animation/Comedy/Musical genres marked by warmth/relationship/growing-up cues and lower age_rating.
   - Inference: The user seeks relaxation, emotional warmth, or stress relief (healing/comfort viewing).
4. Intellectual Curiosity (Cognitive Reward):
   - Signal: High engagement with Mystery/Crime/Detective/Documentary/History genres marked by investigation/clue/truth-seeking cues and a relatively high watched_pct.
   - Inference: The user seeks mental stimulation, puzzle-solving, or knowledge acquisition.
5. Killing Time & Passive Enjoyment (Low-Effort Reward):
   - Signal: High engagement with light Comedy/Popcorn content marked by simple-premise descriptions, low cognitive/ world-rule cues, and a relatively “good enough” watched_pct.
   - Inference: The user seeks to pass time with minimal mental effort and light entertainment.
6. Social Connection & Talk Value (Social Reward):
   - Signal: High engagement with blockbuster/franchise/trending-popular content marked by recognizable IP cues and relatively recent release_year.
   - Inference: The user seeks to stay culturally relevant (FOMO), share common ground with peers, or acquire topics for social conversation ("Watercooler moments").

Step 2: Cross-check with the Reference Summary
Verify the identified anchor against the confirmed Positive and Negative Preferences. Ensure the final output does not logically conflict with the provided Positive/Negative Preferences.

Step 3: Generate Narrative Output
Synthesize the findings into a narrative paragraph.
Do not list the items. Describe the user's motivation naturally.
Avoid referencing specific titles.
Avoid emotional guessing. Stick to the functional utility of the media consumption.
To keep the response concise, you must produce the answer in no more than three sentences.

[Example Output]
{{
  "behavioral_motivation": "The user ..."
}}

Now, based on the provided Input Data, extract the Decision Drivers - Behavioral Motivation.

Input Data: 
1. Preference: {preference}
2. User's Historical Completion Stats: {user_completion_stats}
3. Behavior Logs: {log_data}

Output (JSON only):
"""