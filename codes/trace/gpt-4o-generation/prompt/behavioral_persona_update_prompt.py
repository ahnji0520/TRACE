CONTENT_MATERIAL_UPDATE = """
You are an expert data analyst synthesizing a Long-term Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative) - subdivided into Content Material and Production Context.
2.  Consumption Habits
3.  Decision Driver

Your Task:
Focus exclusively on the "Content Material" sub-component of Preferences. You must update the "Long-term Persona" by integrating the "Session-level Persona" (newly extracted from the latest behavior).

Input Data Context:
1. Long-term Persona (Current): The user's established preferences accumulated from all previous sessions.
2. Session-level Persona (New): The user's preferences extracted specifically from the current session's logs.

Step 1: Analyze Long-term Persona
Review the established Genres, Subject Matter, and Topics in the Long-term Persona to understand the user's historical baseline.

Step 2: Evaluate Session-level Persona
Identify the newly revealed Genre, Subject Matter, and Topic insights from the Session-level Persona.

Step 3: Synthesize and Evolve
- Reinforce (Consistency): If the Session-level insights align with the Long-term Persona, reinforce those specific traits with more detail or higher confidence.
- Integrate (Expansion): If the Session-level insights introduce new elements not present in the Long-term Persona, merge them to reflect a broadening spectrum of interest. Treat these as cumulative additions.
- Pivot (Contradiction): If the Session-level insights directly contradict the Long-term Persona (e.g., a previously liked topic is now explicitly marked as negative in the current session), prioritize the Session-level insights (Recency) to update the core profile.

Step 4: Resolve Metadata Conflicts
- Perform a mandatory check to ensure the updated Positive and Negative sets are mutually exclusive.
- If an attribute appears in both sets:
    (a) Split into non-overlapping context facets if logically supported.
    (b) Otherwise, prioritize the Session-level insight to assign the attribute, or drop it from both if it remains ambivalent.

Step 5: Generate Final Narrative Output
Synthesize the insights into two distinct narrative paragraphs.
- Avoid listing; use overarching concepts to describe the user's evolution.
- Do not use comparative phrases like "instead of," "rather than," or "prefers A over B."
- Avoid referencing specific movie/series titles.
- Keep each paragraph to a maximum of three sentences, starting directly with the subject.

[Example Output]
{{
  "positive_content_material": "",
  "negative_content_material": ""
}}

Now, update the Preference - Content Material using the inputs below.

[Input Data]
1. Long-term Positive Content Material: {current_positive_content}
2. Long-term Negative Content Material: {current_negative_content}
3. Session-level Positive Content Material (New): {new_positive_content}
4. Session-level Negative Content Material (New): {new_negative_content}

Output (JSON only)
"""

PRODUCTION_CONTEXT_UPDATE = """
You are an expert data analyst synthesizing a Long-term Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative) - subdivided into Content Material and Production Context.
2.  Consumption Habits
3.  Decision Driver

Focus exclusively on the "Production Context" sub-component of Preferences. Your task is to synthesize the "Long-term Persona" with the newly extracted "Session-level Persona" to create an evolved representation of the user's external metadata preferences (Who, When, Where).

Input Data Context:
1. Long-term Persona (Current Production Context): The user's established preferences regarding actors, directors, countries, and eras accumulated from previous sessions.
2. Session-level Persona (New Production Context): The user's external metadata preferences extracted specifically from the current session's behavior.

Step 1: Understand Long-term Persona
Analyze the Long-term Persona to understand the established baseline for Actors, Directors, Production Countries, and Release Eras that the user has previously favored or avoided.

Step 2: Evaluate Session-level Persona
Identify the newly revealed Production Context insights (Actors, Directors, Countries, Eras) from the Session-level Persona.

Step 3: Evolve & Synthesize
- Reinforce & Specify (Same Direction): If the Session-level insights align with the Long-term Persona (e.g., same director, country, or era), reinforce these preferences with more detail or higher confidence.
- Expand & Integrate (New Direction): If the Session-level insights introduce different metadata not present in the Long-term Persona, merge them to reflect a broadening spectrum of interest. These are cumulative additions, not contradictions.
- Prioritize Recency (Contradictory Direction): Execute a "preference pivot" ONLY if there is a direct conflict—such as when an element previously marked as positive is now explicitly categorized as negative in the session-level persona. In this case, prioritize the latest evidence (Recency) to reflect the user's current intent.

Step 4: Resolve Metadata Conflicts & Ensure Mutual Exclusivity
- Perform a mandatory conflict check to ensure the updated Positive and Negative sets are mutually exclusive.
- Conflict Resolution: If an attribute appears in both sets:
    (a) Split into non-overlapping context facets (e.g., "[Country] during [Era A]" vs. "[Country] during [Era B]") if logically supported by the personas.
    (b) Otherwise, prioritize the Session-level insight to assign the attribute, or drop it from both if it remains ambivalent.
- Consistency: Ensure the final Positive and Negative narratives contain no overlapping attributes.

Step 5: Generate Updated Narrative Output
Synthesize the insights into two distinct narrative paragraphs (one for positive, one for negative).
- Avoid simple listing; consolidate overlapping elements into higher-level overarching concepts or eras (e.g., mid-2010s, early 2020s).
- Avoid referencing specific movie/series titles.
- Do not use comparative phrases like "instead of," "rather than," or "prefers A over B."
- Keep each paragraph to a maximum of three sentences, starting directly with the subject.

[Example Output]
{{
  "positive_production_context": "",
  "negative_production_context": ""
}}

Now, update the Preference - Production Context using the inputs below.

[Input Data]
1. Long-term Positive Production Context: {current_positive_production}
2. Long-term Negative Production Context: {current_negative_production}
3. Session-level Positive Production Context (New): {new_positive_production}
4. Session-level Negative Production Context (New): {new_negative_production}

Output (JSON only)
"""

TEMPORAL_PATTERNS_UPDATE = """
You are an expert data analyst synthesizing a Long-term Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative)
2.  Consumption Habits (Temporal patterns, Immersion style)
3.  Decision Drivers

Focus exclusively on the "Temporal patterns" sub-component of Consumption Habits. Your task is to synthesize the "Long-term Persona" with the newly extracted "Session-level Persona" to create an evolved representation of the user's life rhythm and viewing schedule.

Input Data Context:
1. Long-term Persona (Current Temporal Patterns): The user's established life rhythm, peak activity days, and routine type (Clustered, Steady, Sporadic) accumulated from previous sessions.
2. Session-level Persona (New Temporal Patterns): The user's engagement rhythm and activeness insights extracted specifically from the current session.

Step 1: Understand Long-term Persona
Analyze the Long-term Persona to understand the established baseline for Peak Activity Days and the Routine Type that has characterized the user's behavior so far.

Step 2: Evaluate Session-level Persona
Identify the newly revealed Temporal insights from the Session-level Persona, focusing on whether the current session's activeness reinforces or shifts the established rhythm.

Step 3: Evolve & Synthesize
- Reinforce (Consistency): If the Session-level insights align with the Long-term Persona (e.g., same peak days or routine type), reinforce the established pattern with higher confidence or more detail.
- Integrate (Expansion): If the Session-level insights indicate minor variations or a widening of the viewing window, merge them to reflect a more flexible or evolving routine.
- Pivot (Contradiction): If the Session-level insights strongly contradict the Long-term Persona (e.g., a shift from weekend clustering to heavy weekday usage), prioritize the latest evidence (Recency) to reflect the user's current life rhythm.

Step 4: Generate Updated Narrative Output
Synthesize the insights into a single narrative paragraph.
- Avoid simple listing; consolidate overlapping elements into higher-level overarching concepts (e.g., "primarily active during mid-week evenings").
- Describe the user's routine naturally, reflecting any evolution from the past to the present.
- Avoid referencing specific movie/series titles.
- Keep the response to a maximum of three sentences, starting directly with the subject.

[Example Output]
{{
  "temporal_patterns": ""
}}

Now, update the Consumption Habits - Temporal Patterns using the inputs below.

[Input Data]
1. Long-term Temporal Patterns: {current_temporal_persona}
2. Session-level Temporal Patterns (New): {new_temporal_persona}

Output (JSON only)
"""

IMMERSION_STYLE_UPDATE = """
You are an expert data analyst synthesizing a Long-term Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Positive/Negative)
2.  Consumption Habits (Temporal patterns, Immersion style)
3.  Decision Drivers

Focus exclusively on the "Immersion style" sub-component of Consumption Habits. Your task is to synthesize the "Long-term Persona" with the newly extracted "Session-level Persona" to create an evolved representation of the user's viewing depth and commitment level.

Input Data Context:
1. Long-term Persona (Current Immersion style): The user's established baseline regarding commitment level (high completion vs. frequent drop-offs) and binging tendencies (True binging vs. Zapping/Browsing) accumulated from previous sessions.
2. Session-level Persona (New Immersion style): The user's current immersion signals and viewing depth insights extracted specifically from the current session.

Step 1: Understand Long-term Persona
Analyze the Long-term Persona to understand the established Commitment Level and Binging Tendency that have characterized the user's immersive behavior so far.

Step 2: Evaluate Session-level Persona
Identify newly revealed Immersion insights from the Session-level Persona, focusing on whether the current session's behavior (focused deep-viewing vs. indecisive browsing) reinforces or shifts the established pattern.

Step 3: Evolve & Synthesize
- Reinforce (Consistency): If the Session-level insights align with the Long-term Persona, reinforce the established pattern with higher confidence or specify the conditions under which the user is most immersive.
- Integrate (Expansion): If the Session-level insights indicate situational shifts or minor variations in viewing depth, merge them to reflect a more multi-layered or flexible immersion style.
- Pivot (Contradiction): If the Session-level insights strongly contradict the Long-term Persona (e.g., a shift from habitual zapping to consistent full-completion), prioritize the latest evidence (Recency) to reflect the user's current immersive behavior.

Step 4: Generate Updated Narrative Output
Synthesize the insights into a single narrative paragraph.
- Avoid simple listing; consolidate overlapping elements into higher-level overarching concepts (e.g., "exhibits a high-depth commitment style").
- Describe the user's viewing depth and commitment style naturally, reflecting any evolution over time.
- Avoid referencing specific movie/series titles.
- To keep the response concise, produce the answer in no more than three sentences, starting directly with the subject.

[Example Output]
{{
   "immersion_style": ""
}}

Now, update the Consumption Habits - Immersion style using the inputs below.

[Input Data]
1. Long-term Immersion style: {current_immersion_persona}
2. Session-level Immersion style (New): {new_immersion_persona}

Output (JSON only)
"""

SELECTION_PRIORITY_UPDATE = """
You are an expert data analyst synthesizing a Long-term Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Content Material, Production Context)
2.  Consumption Habits (Temporal patterns, Immersion style)
3.  Decision Drivers (Selection Priority, Behavioral Motivation)

Focus exclusively on the "Selection Priority" sub-component of Decision Drivers. Your task is to synthesize the "Long-term Persona" with the newly extracted "Session-level Persona" to create an evolved representation of the user's "Hierarchy of Choice" and the primary "Anchor" that governs their engagement.

Input Data Context:
1. Long-term Persona (Current Selection Priority): The user's established primary decision factor (Anchor) accumulated from previous sessions.
2. Session-level Persona (New Selection Priority): The primary decision factor identified specifically from the current session's behavior.
3. Updated Preferences (Reference): The user's latest Positive/Negative preferences for Content Material and Production Context. Your synthesis must align logically with these updated preferences.

Step 1: Understand Long-term Persona
Analyze the Long-term Persona to identify the established "Anchor" (e.g., a specific genre, director style, or era) that has historically driven the user's high-engagement choices.

Step 2: Evaluate Session-level Persona
Identify the "Anchor" signal from the Session-level Persona. Determine if the current session's choices were driven by the same historical factor or if a new primary driver has emerged.

Step 3: Evolve & Synthesize
- Reinforce (Consistency): If the Session-level anchor aligns with the Long-term Persona, reinforce the Anchor with higher confidence or specify sub-conditions.
- Expand (New Direction): If the Session-level persona introduces a new secondary driver that doesn't contradict the baseline, merge them to reflect a more multi-layered hierarchy of choice.
- Pivot (Contradiction): If the Session-level anchor strongly contradicts the Long-term Persona, prioritize the latest evidence (Recency) to reflect a fundamental shift in the user's hierarchy of choice.

Step 4: Cross-check with Preferences
Verify that the synthesized "Selection Priority" is logically consistent with the "Updated Preferences". The primary Anchor must be reflected in the user's strongest positive preferences.

Step 5: Generate Updated Narrative Output
Synthesize the insights into a single narrative paragraph.
- Avoid simple listing; consolidate insights into overarching concepts about the user's decision-making logic.
- Avoid referencing specific movie/series titles.
- Do not use comparative phrases like "instead of," "rather than," or "prefers A over B."
- Keep the response to a maximum of three sentences, starting directly with the subject.

[Example Output]
{{
  "selection_priority": ""
}}

Now, update the Decision Drivers - Selection Priority using the inputs below.

[Input Data]
1. Long-term Selection Priority: {current_selection_priority}
2. Session-level Selection Priority (New): {new_selection_priority}
3. Updated Preferences (Content & Production): {updated_preferences}

Output (JSON only)
"""

MOTIVATIONS_UPDATE = """
You are an expert data analyst synthesizing a Long-term Behavioral Persona for a recommendation system.

Definition of Behavioral Persona:
"A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively."

The Behavioral Persona consists of three main components:
1.  Preferences (Content Material, Production Context)
2.  Consumption Habits (Temporal patterns, Immersion style)
3.  Decision Drivers (Selection Priority, Behavioral Motivation)

Focus exclusively on the "Behavioral Motivation" sub-component of Decision Drivers. Your task is to synthesize the "Long-term Persona" with the newly extracted "Session-level Persona" to create an evolved representation of the user's "Functional Purpose" (the implicit goal or reward).

Input Data Context:
1. Long-term Persona (Current Behavioral Motivation): The user's established psychological goals and rewards accumulated from previous sessions.
2. Session-level Persona (New Behavioral Motivation): The functional reward identified specifically from the current session's behavior.
3. Updated Preferences (Reference): The user's latest Positive/Negative preferences. Your synthesis must align logically with these updated preferences.

Step 1: Understand Long-term Persona
Analyze the Long-term Persona to identify the user's historical functional rewards and purposes.

Step 2: Evaluate Session-level Persona using Mandatory Taxonomy
Identify the reward signal from the Session-level Persona. You MUST interpret and synthesize the motivation based on the following 6 categories:

1. Stimulation & Adrenaline (Visceral Reward):
   - Signal: High engagement with Action/Thriller/Horror/Racing genres marked by chase/fight/violence/high-stakes cues and higher age_rating.
   - Inference: The user seeks immediate excitement, tension, and sensory stimulation.
2. Escapism & Immersion (Imaginative Reward):
   - Signal: High engagement with Fantasy/Sci-Fi/Adventure genres marked by world-building cues (alternate worlds, magic/aliens, quests, lore) and high watched_pct.
   - Inference: The user seeks to detach from reality and immerse themselves in alternative universes.
3. Mood Regulation & Comfort (Emotional Reward):
   - Signal: High engagement with Family/Animation/Comedy/Musical genres marked by warmth/relationship/growing-up cues and lower age_rating.
   - Inference: The user seeks relaxation, emotional warmth, or stress relief (healing/comfort viewing).
4. Intellectual Curiosity (Cognitive Reward):
   - Signal: High engagement with Mystery/Crime/Detective/Documentary/History genres marked by investigation/clue/truth-seeking cues and high watched_pct.
   - Inference: The user seeks mental stimulation, puzzle-solving, or knowledge acquisition.
5. Killing Time & Passive Enjoyment (Low-Effort Reward):
   - Signal: High engagement with light Comedy/Popcorn content marked by simple-premise descriptions, low cognitive/ world-rule cues, and “good enough” watched_pct.
   - Inference: The user seeks to pass time with minimal mental effort and light entertainment.
6. Social Connection & Talk Value (Social Reward):
   - Signal: High engagement with blockbuster/franchise/trending-popular content marked by recognizable IP cues and relatively recent release_year.
   - Inference: The user seeks to stay culturally relevant (FOMO), share common ground with peers, or acquire topics for social conversation ("Watercooler moments").

Step 3: Evolve & Synthesize
- Reinforce (Consistency): If the Session-level reward aligns with the Long-term Persona, reinforce this motivation with higher confidence.
- Expand (New Direction): If the Session-level persona introduces a different category from the taxonomy, interpret this as a multi-layered motivation or a situational expansion.
- Pivot (Contradiction): If the Session-level reward strongly contradicts the Long-term pattern, prioritize the latest evidence (Recency) to reflect the user's current goal.

Step 4: Cross-check with Preferences
Verify that the synthesized "Behavioral Motivation" is logically consistent with the "Updated Preferences".

Step 5: Generate Updated Narrative Output
Synthesize the insights into a single narrative paragraph.
- Avoid simple listing; use overarching concepts from the 6 categories above.
- Avoid referencing specific movie/series titles.
- To keep the response concise, produce the answer in no more than three sentences, starting directly with the subject.

[Example Output]
{{
  "behavioral_motivation": ""
}}

Now, update the Decision Drivers - Behavioral Motivation using the inputs below.

[Input Data]
1. Long-term Behavioral Motivation: {current_motivation_persona}
2. Session-level Behavioral Motivation (New): {new_motivation_persona}
3. Updated Preferences (Content & Production): {updated_preferences}

Output (JSON only)
"""