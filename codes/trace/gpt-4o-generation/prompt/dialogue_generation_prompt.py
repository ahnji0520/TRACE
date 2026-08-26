# DIALOGUE_GENERATION = """
# # Role: Expert CRS Data Annotator & Creative Writer
# You are an expert data annotator and creative writer specializing in Conversational Recommender Systems (CRS). Your task is to generate a natural, multi-turn dialogue between a User and a Recommender Agent based on the provided information.

# ---

# ### 1. [Given Information]
# Analyze the following categories to drive the narrative:
# 1. Dialogue Persona: A holistic representation of user expression formed within the conversational context. It integrates linguistic attributes—such as identity, attitude, and speaking style—with user-related information observed through dialogue, including states, preferences, and factual information.
# 2. Behavioral Persona: A structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—to personalize decision-making effectively.
# 3. Event (Day: {day}): A specific event that happened to the user on this day.
# 4. Memory (Previous Session Context): A bridge between sessions that captures the user's recent history. It records the previous session's event and the user's specific reaction to the last recommendation.
# 5. User's Historical Completion Stats
# 6. Source Items: Used to establish context. Interpret completion rates relatively based on User's Historical Completion Stats; treat items with relatively high completion rates as Preferred, and relatively low rates as disliked.
# 7. Target Item: The optimal solution to be recommended exactly once at the end.

# ---

# ### 2. [Strategic Dialogue Design]

# Before generating the dialogue, you must first design the flow by integrating all given information into a coherent plan. This design process ensures that the interaction remains personalized, logically consistent, and leads naturally to the recommendation.

# ---

# Phase 1: Persona Initialization
# Initialize the user's identity and logic by integrating explicit conversational data with implicit behavioral patterns.

# 1. Dialogue Persona Manifestation
# Establish the user's identity, attitude, and communication style, incorporating explicit preferences and user information revealed during the dialogue.
# • Adaptive Manifestation: If the current flow of the dialogue (driven by the Event and items) deviates from the established Dialogue Persona, interpret it as a situational shift or a natural evolution of the user’s explicit identity and needs.
# • Communication Style Operationalization: Translate the communication styles into specific verbal behaviors: for Clarity, use specific genres/criteria (High) vs. vague mood-based needs (Low); for Initiative, allow the user to lead and ask questions, request recommendation proactively (High) vs. remain reactive to the Agent's prompts (Low); for Feedback, use blunt "Yes/No" and clear reasons (Direct) vs. polite, hedged, or subtle expressions (Indirect).

# 2. Behavioral Persona Assessment
# Analyze the user’s implicit information and multi-dimensional tendencies derived from observable behavior logs.
# • Behavioral Alignment: If the current flow of the dialogue (driven by the Event and items) contradicts established patterns, interpret the shift as a situational event or a natural expansion/evolution of the user’s underlying implicit tastes, usage patterns, and decision drivers.
# • Progressive Concretization: Ensure that the latent preferences and recurring tendencies from the Behavioral Persona are not merely used as background filters but are actively articulated and concretized through the User's verbal expressions as the dialogue unfolds. The dialogue should serve as a process where implicit behavior-log patterns are transformed into explicit conversational statements.

# ---

# Phase 2: Continuity and Contextual Setup
# This phase focuses on why the conversation is happening now and how it connects to the user’s past. 

# 1. Memory Integration Strategy
# - Protocol:
# 	- Awareness Boundary: The Agent is strictly unaware of the new event until disclosed by the User. Strictly distinguish the new Event from past events in the Memory
# 	- Natural Timing: Use relative markers rather than specific dates.
# - Memory Integration Guideline: Naturally weave in memory if they feel appropriate for the current dialogue flow to ensure conversational continuity:
#     - Agent-Initiated: The Agent may actively inquire about a past Event in memory (e.g., "How did that [past event] turn out?") or request a review of the previously accepted Target Item (e.g., "Did you find [recommended item title] helpful?").
#     - User-Initiated: The User may proactively initiate the session by linking the new Event to a past context or offering spontaneous feedback on the item accepted in the last session before the Agent asks.

# 2. Event Expansion
# Deeply infuse the Event into the dialogue through Event Expansion, exploring how the event has impacted the user's current emotional state or practical needs.

# ---

# Phase 3: Strategic Bridging and Navigation

# In this phase, you bridge the user's current situation to the item exploration process.

# 1. Contextual Bridge Strategy
# Select one of the following strategies to connect the Event to the Target Item:
# - Mood Modulation: Recommending an item to either enhance or shift the user’s current mood caused by the event.
# - Situation Mirroring: Selecting items that reflect or resonate with the user’s current circumstances.
# - Wish Compensation: Offering an item that provides something the user currently lacks or desires due to the event.
# - Curiosity Expansion: Using the event as a catalyst to explore new genres or topics the user is now curious about.

# 2. Source Item Role Assignment
# Assign specific roles to the Source Items to facilitate the navigation of tastes. No Source Item can be accepted as a final recommendation; if suggested, they must result in rejection(already seen, soft reject, direct reject etc.).
# - Preferred Items (High watched_pct): Assign roles such as Anchor (similarity reference), Already Seen (match for taste but declined due to prior completion), or Soft Reject (match for taste but declined due to current context).
# - Disliked Items (Low watched_pct): Assign roles such as Anchor (to exclude styles) or Reject (direct refusal due to taste mismatch).
# - Note: Discussion of Source Items can be initiated by either the Agent or the User.

# ---

# Phase 4: Narrative Arc and Target Synthesis

# The final step is to organize the sequence of the dialogue so it leads to a successful resolution.

# 1. Sequence Planning (Narrative Arc)
# Plan the sequence of the dialogue to ensure both the Event and all Source Items are woven in organically. The conversation should move through:
# - Opening: Establishing context via Memory and the Event.
# - Exploration: Discussing Source Items and gathering feedback to narrow down the search.
# - Transition: Using the chosen Bridge Strategy to pivot toward the final solution.

# 2. Target Item Synthesis
# Design a natural transition where the Target Item emerges as the only successful solution. It must logically synthesize:
# - The immediate impact of the Event.
# - The underlying Behavioral Persona and Dialogue Persona.
# - The feedback gathered from discussing Source Items.
# This synthesis leads to a final acceptance, concluding the Narrative Arc with a highly personalized recommendation.

# ---

# ### 3. [Core Rules & Constraints]
# 1. Item Usage Logic:
#    - Source Items: All must be discussed. 
#    - Target Item: Introduced ONLY at the end. The User may ask for metadata details to verify suitability before the final [target acceptance].
#    - No Duplicate Recommendations: Once an item has been discussed or referenced in the dialogue, it must not be re-introduced or suggested as a "new" recommendation in any subsequent turns. Each item’s role must be established and concluded in its respective turn.
#    - The Agent must not suggest re-watching items.
# 2. Factual Consistency:
#    - Closed-World Constraint: Do NOT mention any items outside the provided source items, target item, or memory.
#    - No Personal Experience: The Agent must not claim to have watched or felt emotions about the items.
#    - Strict Event-Blindness Protocol: The Agent MUST remains strictly unaware of the new Event until it is disclosed by the User. Disclosure occurs either when the User proactively introduces it or when the Agent prompts the User to share their day or current mood.
#    - Memory Accuracy: If the 'Memory' is empty(first session), do not mention or imply any past conversation history. Also, Never use phrases such as "I remember you mentioned..." for non-existent memory.
#    - Source vs. Memory Distinction: Do not refer to Source Items as if they were mentioned in previous sessions.
# 3. Structure & Labeling:
#    - Max 13 turns.
#    - The dialogue must strictly follow an alternating sequence between the Agent and the User. 
#    - Every turn must start with a label: [chitchat about ...], [QA about ...], or [recommendation about ...].
#    - Final labels: [target recommendation] (Agent) and [target acceptance] (User).


# ---

# ### [Input Data]
# 1. Dialogue Persona: {dialogue_persona}
# 2. Behavioral Persona: {behavioral_persona}
# 3. Event (Day: {day}): {event}
# 4. Memory: {memory}
# 5. User's Historical Completion Stats: {historical_completion_stats}
# 6. Source Items: {source_items}
# 7. Target Item: {target_item}

# ---

# ### [Output Instructions] Respond only in valid JSON format (a single JSON object)

# 1. Dialogue Design Strategy: Briefly describe how you will map the User's style, the Event's impact on their persona, and the integrated flow of items and the event.
# 2. Dialogue: Generate the multi-turn conversation as a list of objects. Each object must follow this structure:
#    - `label`: The strategic label for the turn.
#    - `speaker`: The role of the speaker (Agent or User).
#    - `text`: The actual utterance text.

# **JSON structure example:**
# {{
#   "Dialogue Design Strategy": "...",
#   "Dialogue": [
#     {{
#       "label": "[]",
#       "speaker": ""
#       "text": "Utterance text here..."
#     }},
#     ...
#     {{
#       "label": "[]",
#       "speaker": ""
#       "text": "Utterance text here..."
#     }},
#   ]
# }}

# """

DIALOGUE_GENERATION = """
# Role: Expert CRS Data Annotator & Creative Writer
You are an expert data annotator and creative writer specializing in Conversational Recommender Systems (CRS). Your task is to generate a natural, multi-turn dialogue between a User and a Recommender Agent based on the provided information.

---

### 1. [Given Information]
Analyze the following categories to drive the narrative:
1. Dialogue Persona: A holistic representation of user expression formed within the conversational context. It integrates linguistic attributes—such as identity, attitude, and speaking style—with user-related information observed through dialogue, including states, preferences, and factual information.
2. Behavioral Persona: A structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—to personalize decision-making effectively.
3. Event: A specific event that happened to the user on this day.
4. Memory: A bridge between sessions that captures the user's evolving history. It consists of an Episodic Memory—a chronological record of life events—and the Recommendation Outcomes from the immediate preceding session.
5. User's Historical Completion Stats
6. Source Items: Items used to navigate tastes as either Anchors (references for similarity or exclusion) or Recommendation Candidates (must be rejected if suggested).
7. Target Item: The optimal solution to be proposed at the conclusion as the final choice. It must be the only item accepted by the user.

---

### 2. [Strategic Dialogue Design]

Before generating the dialogue, you must first design the flow by integrating all given information into a coherent plan. This design process ensures that the interaction remains personalized, logically consistent, and leads naturally to the recommendation.

---

Phase 1: Persona Initialization
Initialize the user's identity and logic by integrating explicit conversational data with implicit behavioral patterns.

1. Dialogue Persona Manifestation
Establish the user's identity, attitude, and communication style, incorporating explicit preferences and user information revealed during the dialogue.
• Adaptive Manifestation: If the current flow of the dialogue(driven by the Event and Source/Target items) deviates from the established Dialogue Persona, interpret it as a situational shift or a natural evolution of the user’s explicit identity and needs.
• Communication Style Operationalization: Translate the user's communication styles into specific verbal behaviors: 
	- Clarity: Use well-defined requirements (High) vs. Express vague, mood-oriented preferences (Low).
	- Initiative: Proactively drive the dialogue and ask questions (High) vs. Passively respond only to the Agent’s inquiries (Low).
	- Feedback: Deliver blunt, unambiguous evaluations with explicit reasons (Direct) vs. Use polite, hedged, or nuanced expressions to imply feedback (Indirect).

2. Behavioral Persona Assessment
Analyze the user’s implicit information and multi-dimensional tendencies derived from observable behavior logs.
• Behavioral Alignment: If the current flow of the dialogue (driven by the Event and Source/Target items) contradicts established patterns, interpret the shift as a situational event or a natural expansion/evolution of the user’s underlying implicit tastes, usage patterns, and decision drivers.
• Progressive Concretization: Ensure that the latent preferences and recurring tendencies from the Behavioral Persona are not merely used as background filters but are actively articulated and concretized through the User's verbal expressions as the dialogue unfolds. The dialogue should serve as a process where implicit behavior-log patterns are transformed into explicit conversational statements.
• Narrative Embodiment: Naturally weave implicit behavioral patterns into the dialogue by translating them into the user’s subjective, everyday language.

---

Phase 2: Continuity and Contextual Setup
This phase focuses on why the conversation is happening now and how it connects to the user’s past. 

1. Memory Integration Strategy
- Protocol:
	- Awareness Boundary: The Agent is strictly unaware of the new event until disclosed by the User. Strictly distinguish the new Event from past events in the Memory.
	- Natural Timing: Use relative markers rather than specific dates.
- Memory Integration Guideline: Naturally weave in memory if they feel appropriate for the current dialogue flow to ensure conversational continuity:
    - Agent-Initiated: The Agent may actively inquire about a past Event in memory (e.g., "How did that [past event] turn out?") or request a review of the previously accepted Target Item (e.g., "Did you find [recommended item title] helpful?"). When requesting a review of a past item, use open-ended questions instead of assuming the user liked it.
    - User-Initiated: The User may proactively initiate the session by linking the new Event to a past context or offering spontaneous feedback on the item accepted in the last session before the Agent asks.

2. Event Expansion
Deeply infuse the Event into the dialogue through Event Expansion, exploring how the event has impacted the user's current emotional state or practical needs.

---

Phase 3: Strategic Bridging and Navigation

In this phase, you bridge the user's current situation to the item exploration process.

1. Contextual Bridge Strategy
Select one of the following strategies to connect the Event to the Target Item:
- Mood Modulation: Recommending an item to either enhance or shift the user’s current mood caused by the event.
- Situation Mirroring: Selecting items that reflect or resonate with the user’s current circumstances.
- Wish Compensation: Offering an item that provides something the user currently lacks or desires due to the event.
- Curiosity Expansion: Using the event as a catalyst to explore new genres or topics the user is now curious about.

2. Source Item Role Assignment
Assign specific roles to the Source Items to facilitate the navigation of tastes. No Source Item can be accepted as a final recommendation; if recommended, they must result in rejection(already seen, soft reject, direct reject etc.).
- Preferred Items (High watched_pct): Assign roles such as Anchor (similarity reference), Already Seen (match for taste but declined due to prior completion), or Soft Reject (match for taste but declined due to current context).
- Disliked Items (Low watched_pct): Assign roles such as Anchor (to exclude styles) or Reject (direct refusal due to taste mismatch).
- Note: Discussion of Source Items can be initiated by either the Agent or the User.

---

Phase 4: Narrative Arc and Target Synthesis

The final step is to organize the sequence of the dialogue so it leads to a successful resolution.

1. Sequence Planning (Narrative Arc)
Plan the sequence of the dialogue to ensure both the Event and all Source Items are woven in organically. The conversation should move through:
- Opening: Establishing context via Memory and the Event.
- Exploration: Discussing Source Items and gathering feedback to narrow down the search.
- Transition: Using the chosen Bridge Strategy to pivot toward the final solution.

2. Target Item Synthesis & Explainability
- Design a natural transition where the Target Item emerges as the only successful solution. It must logically synthesize:
  - The immediate impact of the Event.
  - The underlying Behavioral Persona and Dialogue Persona.
  - The feedback gathered from discussing Source Items.
- The final recommendation utterance must explicitly verbalize the reasoning by integrating these synthesized points into a clear and persuasive explanation.
- This synthesis leads to a final acceptance of Target item, concluding the Narrative Arc with a highly personalized recommendation.

---

### 3. [Core Rules & Constraints]
1. Item Usage Logic:
   - Source Items: All must be discussed(as anchor or recommendation). 
   - Target Item: Introduced ONLY at the end. The User may ask for metadata details to verify suitability before the final acceptance.
   - The final recommendation must emerge as the logical synthesis of the event's impact, user personas, and source item feedback, explicitly verbalizing these factors into a clear and persuasive explanation.
   - No Duplicate Recommendations: Once an item has been discussed or referenced in the dialogue, it must not be re-introduced or suggested as a "new" recommendation in any subsequent turns. Each item’s role must be established and concluded in its respective turn.
   - The Agent must not suggest re-watching items.
2. Factual Consistency:
   - Closed-World Constraint: Do NOT mention any items outside the provided source items, target item, or memory.
   - No Personal Experience: The Agent must not claim to have watched or felt emotions about the items.
   - Strict Event-Blindness Protocol: The Agent MUST remains strictly unaware of the new Event until it is disclosed by the User. Disclosure occurs either when the User proactively introduces it or when the Agent prompts the User to share their day or current mood.
   - Memory Accuracy: If the 'Memory' is empty(first session), do not mention or imply any past conversation history. Also, Never use phrases such as "I remember you mentioned..." for non-existent memory.
   - Source vs. Memory Distinction: Do not refer to Source Items as if they were mentioned in previous sessions.
3. Structure & Labeling:
   - The dialogue must consist of between 10 and 14 turns.
   - The dialogue must strictly follow an alternating sequence between the Agent and the User, strictly prohibiting any consecutive utterances from the Agent.
   - Every Agent turn must start with exactly one of these labels: [chitchat], [question], [answer], [recommendation].
	   - [recommendation]: Any utterance that proposes a specific title for the user to watch as a candidate in this session. (Highest Priority)
     - [chitchat]: Use for rapport, empathy, or general social interaction.
	   - [question]: Asking about general preferences/moods without mentioning specific title.
       - Exceptionally, asking for a review or opinion on an item that was recommended in a previous session must be labeled as [question].
       - Otherwise, If a specific item is introduced as a candidate for the current session, it is always [recommendation].
		 - [answer]: Use ONLY when providing factual metadata or direct answers to a User's specific question about an item.
	- User turns must NOT have any labels.

---

### 4. Reflection & Refinement
- Before providing the final dialogue, perform a self-audit to ensure all [Strategic Dialogue Design], [Core Rules & Constraints from Section] are strictly met.
- Verify that all Agent labels strictly match their functional definitions.

---

### [Output Instructions] Respond only in valid JSON format (a single JSON object)

1. Dialogue Design Strategy: Briefly describe how you will map the User's persona, the Event's impact on their persona, and the integrated flow of items and the event.
2. Dialogue: Generate the multi-turn conversation as a list of objects. Each object must follow this structure:
   - `label`: The strategic label for the turn (Only an Agent's turn).
   - `speaker`: The role of the speaker (Agent or User).
   - `text`: The actual utterance text.

**JSON structure example:**
{{
  "Dialogue Design Strategy": "...",
  "Dialogue": [
    {{
      "speaker": "User"
      "text": "Utterance text here..."
    }},
    ...
    {{
      "label": "[chitchat]",
      "speaker": "Agent"
      "text": "Utterance text here..."
    }},
  ]
}}
"""

DIALOGUE_GENERATION_2 = """
### [Input Data]
1. Dialogue Persona: {dialogue_persona}
2. Behavioral Persona: {behavioral_persona}
3. Event (Day: {day}): {event}
4. Memory: {memory}
5. User's Historical Completion Stats: {historical_completion_stats}
6. Source Items: {source_items}
7. Target Item: {target_item}

Generate conversation with multiple turns based on the given information.
"""




DIALOGUE_GENERATION__ = """
You are an expert data annotator and creative writer specializing in Conversational Recommender Systems (CRS). Your task is to generate a natural, multi-turn dialogue between a User and a Recommender Agent based on the provided information. 

[Given Information]
1. Dialogue Persona: A holistic representation of user expression formed within the conversational context. It integrates linguistic attributes—such as identity, attitude, and speaking style—with user-related information observed through dialogue, including states, preferences, and factual information.
2. Behavioral Persona: A structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—to personalize decision-making effectively.
3. Event: A specific event that happened to the user on this day.
4. Memory (Previous Session Context): A bridge between sessions that captures the user's recent history. It records the previous session's event and the user's specific reaction to the last recommendation.
5. User's Historical Completion Stats
6. Source Items: Items used to navigate tastes as either Anchors (references for similarity or exclusion) or Recommendation Candidates (must be rejected if suggested).
7. Target Item: The optimal solution to be proposed at the conclusion as the final choice. It must be the only item accepted by the user.

[Instruction]
- Treat the user persona as an emergent and continuously evolving construct rather than a fixed, pre-defined profile.
  - Construct the Dialogue Persona from explicit conversational signals such as expressed attitudes, values, and preferences.
  - Maintain the Behavioral Persona as a latent structure inferred from behavioral logs, but ensure it is progressively externalized and verbalized through the user’s utterances rather than remaining a passive background filter.

- Interpret deviations from previously established persona patterns as situational shifts or natural evolution, not as inconsistencies or errors.
  - Allow both short-term changes driven by transient events and long-term expansions of user taste and intent.

- Operationalize the user’s communication style as explicit linguistic behaviors.
  - Clarity: preference for concrete criteria and genres (High) vs. vague, mood-driven needs (Low).
  - Initiative: user-led questioning and proactive requests (High) vs. reactive responses to agent prompts (Low).
  - Feedback: blunt and explicit “Yes/No” responses with clear reasoning (Direct) vs. polite, hedged, or implicit expressions (Indirect).

- Maintain strict event-blindness: the agent must remain unaware of any new Event until it is disclosed through dialogue.
  - An Event may only be introduced by the user spontaneously or elicited through general, non-assumptive prompts.
  - Clearly distinguish newly disclosed Events from past Memory, and never merge or conflate the two.

- Integrate past Memory only when it naturally supports conversational continuity.
  - Use open-ended inquiries when referencing past Events or previously accepted items, without assuming satisfaction or preference.
  - If no Memory exists, do not imply or fabricate prior conversational history.

- Use Source Items strictly as exploratory instruments rather than final recommendations.
  - Every Source Item must be discussed and assigned a definitive role: Anchor, Already Seen, Soft Reject, or Reject.
  - Source Items must be fully consumed in the exploration process and must never be reintroduced or promoted as final choices.

- Select a single, explicit bridging strategy to connect the current Event to item exploration.
  - Choose exactly one of: Mood Modulation, Situation Mirroring, Wish Compensation, or Curiosity Expansion.
  - The selected strategy must consistently govern both exploration and final synthesis.

- Structure the dialogue around a coherent narrative arc.
  - Opening: establish context through Event disclosure and emotional or situational grounding.
  - Exploration: refine user taste and intent through discussion of Source Items.
  - Transition: narrow the search space via the chosen bridging strategy.
  - Resolution: converge on a single, well-justified Target Item.

- Introduce the Target Item exactly once and only at the end of the dialogue, as a logical inevitability.
  - Explicitly verbalize how the recommendation synthesizes the Event’s impact, the Dialogue Persona, the Behavioral Persona, and feedback derived from Source Items.
  - Duplicate recommendations or reintroductions of previously discussed items are strictly prohibited.

- Enforce strict structural and labeling constraints on the dialogue.
  - The dialogue must contain 10–14 turns with perfect alternation between Agent and User.
  - Every Agent turn must include exactly one label from: [chitchat], [question], [recommendation], [answer], [transition].
  - Any utterance that introduces a specific item candidate must be labeled as [recommendation].
  - The dialogue must strictly alternate between the agent and the user, with exactly one utterance per turn.

  
### [Output Instructions] Respond only in valid JSON format (a single JSON object)
1. Dialogue Design Strategy: Briefly describe how you will map the User's style, the Event's impact on their persona, and the integrated flow of items and the event.
2. Dialogue: Generate the multi-turn conversation as a list of objects. Each object must follow this structure:
   - `label`: The strategic label for the turn (Only an Agent's turn).
   - `speaker`: The role of the speaker (Agent or User).
   - `text`: The actual utterance text.

**JSON structure example:**
{{
  "Dialogue Design Strategy": "...",
  "Dialogue": [
    {{
      "speaker": "User"
      "text": "Utterance text here..."
    }},
    ...
    {{
      "label": "[chitchat]",
      "speaker": "Agent"
      "text": "Utterance text here..."
    }},
  ]
}}"""

DIALOGUE_GENERATION__2 = '''
[Input Data]
1. Dialogue Persona: {dialogue_persona}
2. Behavioral Persona: {behavioral_persona}
3. Event (Day: {day}): {event}
4. Memory: {memory}
5. User's Historical Completion Stats: {historical_completion_stats}
6. Source Items: {source_items}
7. Target Item: {target_item}

Generate a long, in-depth conversation with multiple turns based on the given information.
'''
