EVENT_GENERATION_RULES= """
[Definitions]
- watched_date, weekday: from WATCH_LOG_ITEM
- Seed: a short noun phrase capturing the implied core emotion/message/perspective/human-scale dilemma of the movie or series. Seed must be inspired by WATCH_LOG_ITEM; specific yet broadly applicable. 
- Event: one sentence describing a plausible real-world situation on watched_date BEFORE viewing, grounded in USER_PROFILE, that would make the movie/series feel personally resonant or emotionally useful. Event must be explainable purely from USER_PROFILE + real-world context, (+ X, if provided), and derived from the seed.
Event must be purely diegetic (in-world) and observational: do NOT mention or explain themes/messages/lessons/motifs (e.g., “heroism”, “redemption”, “identity”), and do NOT explicitly compare the event to the seed/title.

[Seed -> Event Mapping Strategies]
- situational_mirroring: Generate an event that mirrors the seed’s situation/material in the user’s real life, so the user is more likely to empathize or identify with the movie. (Note: the Event itself must still describe only the real-world situation BEFORE viewing.)
- mood_modulation: Generate an event that establishes the user’s emotional state before viewing (positive or negative), making the movie appealing as a mood tool—either to match and release the emotion (catharsis), counter and lift it (mood repair), or maintain/amplify it (mood maintenance). (Note: the movie or series will be a mood tool. Event itself must still describe only the real-world situation BEFORE viewing and any mood modulation.)
- wish_compensation: Generate an event that highlights a concrete constraint or lack in the user’s life before viewing, making the movie appealing as vicarious fulfillment that "fills the gap."
- curiosity_expansion: Generate an event that sparks a specific, non-emotional curiosity before viewing (a topic, skill, place, era, or culture), making the movie appealing as a way to explore, learn, or broaden perspective.

"""
INITIAL_EVENT_GENERATION_OUTPUT="""
[Output Format] (JSON only)
{
  "pairs": {
    "situational_mirroring": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    },
    "mood_modulation": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    },
    "wish_compensation": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    },
    "curiosity_expansion": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    }
  }
}

[Constraints]
- Output raw JSON only.
- Event must be third-person (the user).
- One event = one day. Describe only watched_date BEFORE viewing.
- Do NOT reenact the title’s plot or use title-unique entities/keywords in Events.

Now generate the output using the inputs below.
"""

SUB_EVENT_GENERATION_OUTPUT="""
[Output Format] (JSON only)
{
  "pairs": {
    "situational_mirroring": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    },
    "mood_modulation": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    },
    "wish_compensation": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    },
    "curiosity_expansion": {
      "seed": "<short noun phrase>",
      "event": "On <watched_date>, <one sentence>"
    }
  }
}

[Constraints]
- Output raw JSON only.
- Event must be third-person (the user).
- One event = one day. Describe only watched_date BEFORE viewing.
- Do NOT reenact the title’s plot or use title-unique entities/keywords in Events.

Now generate the output using the inputs below.
"""

INITIAL_EVENT_GENERATION_INTRO = "You generate plausible 'day-of' events for a user using universal seeds inspired by a movie or series they watched, preserving narrative plausibility while avoiding direct data matching."

INITIAL_EVENT_GENERATION_TASK = """
[Task]
1) Read USER_PROFILE and WATCH_LOG_ITEM.
2) Quality constraints:
   - Grounding: use >=1 USER_PROFILE field.
   - Expand: add 1 small new detail not explicit in USER_PROFILE.
   - Specificity: include >=1 concrete anchor (time/place/action/object/role).
   - Diversity: vary angle/anchor across 4 strategies (don’t reuse the same anchor >2x).
3) Create EXACTLY 4 pairs (one per strategy key) in the required JSON."""

INITIAL_EVENT_GENERATION_INPUTS="""
Now generate the output using the inputs below.

[USER_PROFILE]
NOTE: relationship_status meanings
- single: not in a romantic relationship
- couple: in a romantic relationship but not married 
- married: married
[[$User_Profile$]]

[WATCH_LOG_ITEM]
NOTE: Use title/genres/description/keywords to derive Seeds, but do NOT copy title-unique entities or keywords into Events.
[[$Watch_Log_Item$]]

"""

SUB_EVENT_GENERATION_INTRO = "You generate plausible 'day-of' sub-events (Y) as direct causal continuations of the last existing event (X), using universal seeds inspired by a movie or series they watched to preserve narrative plausibility while avoiding direct data matching. The Event must be a direct causal continuation of X: it should explicitly reference one concrete, observable detail from X (person/action/object/commitment/place/time) using a causal connector (e.g., because/after/following/in response to), and it should be unlikely to occur if X had not happened. "

SUB_EVENT_GENERATION_TASK="""
[Task]
1) Read USER_PROFILE, WATCH_LOG_ITEM, and LAST_EXISTING_EVENT (X).
2) Quality constraints:
   - **Causality**: watched_date event must be caused by X. Use an observable detail from X as the trigger.
   - Grounding: use >=1 USER_PROFILE field.
   - Expand: add 1 small new detail not explicit in USER_PROFILE.
   - Specificity: include >=1 concrete anchor (time/place/action/object/role).
   - Diversity: vary angle/anchor across 4 strategies (don’t reuse the same anchor >2x).
3) Create EXACTLY 4 pairs (one per strategy key) in the required JSON."""

SUB_EVENT_GENERATION_INPUTS="""

[USER_PROFILE]
NOTE: relationship_status meanings
- single: not in a romantic relationship
- couple: in a romantic relationship but not married
- married: married 
[[$User_Profile$]]

[WATCH_LOG_ITEM]
NOTE: Use title/genres/description/keywords to derive Seeds, but do NOT copy title-unique entities or keywords into Events.
[[$Watch_Log_Item$]]

[LAST_EXISTING_EVENT (X)]
[[$Last_Event$]]"""




INITIAL_EVENT_SELECTION_TASK = """
You will (1) select the best 1 event from the already-generated candidate pairs, then (2) minimally refine it only if needed for data quality.

[Step 1: Selection of the best event]
- Choose the best 1 of the 4 pairs.
- Priority order: User alignment/Grounding > Anti-overfitting > Naturalness > Plausibility.
- selected_event must be copied EXACTLY from one of the pairs.*.event values (no rephrasing).

[Red flags (avoid in selection; if unavoidable, must refine)]
1) Same-day confusion: reads like it happens on the same day as another known event.
2) Movie-locked language: overt theme/setting words (e.g., “heroism”, “mythology”, “Norway/Norse myths”, “alien culture”), or “mirrors/resonates/reminds”.

[Step 2: Minimal refinement rules]
- Only apply edits if the selected_event triggers any red flag.
- Edit as little as possible; keep the same core situation and anchors.
- Allowed edits:
  (a) Date clarity: make it unambiguously on watched_date BEFORE viewing (add a small time marker like “that morning/after work”).
  (b) De-theme: remove/replace movie-locked words with neutral real-life phrasing.

[Output must be JSON only]
{
  "reasoning": "Alignment=<...>; Anchors=<...>; Anti-overfit=<...>.",
  "selected_pair_key": "<situational_mirroring|mood_modulation|wish_compensation|curiosity_expansion>",
  "selected_event": "<exact copy from pairs[selected_pair_key].event>",
  "refined_event": "<either identical to selected_event OR minimally edited version>",
}

Now select using the inputs below.
"""


SUB_EVENT_SELECTION_TASK = """
You will (1) select the best 1 event from the already-generated candidate pairs, then (2) minimally refine it only if needed for data quality.

[Step 1: Selection of the best event]
- Choose the best 1 of the 4 pairs.
- Priority order: Causal directness > User alignment/Grounding > Anti-overfitting > Naturalness > Plausibility.
- selected_event must be copied EXACTLY from one of the pairs.*.event values (no rephrasing).

[Causality requirement (SUB)]
- The event must be a direct causal continuation of X: cite one concrete trigger from X (person/action/object/commitment/place/time) with an explicit causal connector (because/after/following/in response to), and it should be unlikely if X did not happen (counterfactual).

[Red flags (avoid in selection; if unavoidable, must refine)]
1) Same-day confusion: reads like it happens on the same day as X (must be on watched_date only).
2) Movie-locked language: overt theme/setting words (e.g., “heroism”, “mythology”, “Norway/Norse myths”, “alien culture”), or “mirrors/resonates/reminds”.
3) Fake causality: causal wording without a concrete trigger from X (fails counterfactual).

[Step 2: Minimal refinement rules]
- Only apply edits if the selected_event triggers any red flag.
- Edit as little as possible; keep the same core situation and anchors.
- Allowed edits:
  (a) Date clarity: make it unambiguously on watched_date BEFORE viewing (add a small time marker like “that morning/after work”).
  (b) De-theme: remove/replace movie-locked words with neutral real-life phrasing.
  (c) Strengthen causality: if the causal link is weak, keep the original trigger from X, but minimally adjust the RESULTING action/situation in the event so it becomes a plausible consequence of that trigger.

[Output must be JSON only]
{
  "reasoning": "Causality=<...>; Alignment=<...>; Anchors=<...>; Anti-overfit=<...>.",
  "selected_pair_key": "<situational_mirroring|mood_modulation|wish_compensation|curiosity_expansion>",
  "selected_event": "<exact copy from pairs[selected_pair_key].event>",
  "refined_event": "<either identical to selected_event OR minimally edited version>",
}

Now select using the inputs below.
"""

EVENT_SELECTION_INPUT = """
[CANDIDATE_PAIRS]
[[$Candidates$]]
"""