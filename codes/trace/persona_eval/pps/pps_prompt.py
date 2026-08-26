BEHAVIOR_PERSONA_CREDIBILITY = """
[System Instruction]
You are an expert evaluator of behavioral personas for recommender systems. Your job is to judge the Credibility of the provided persona based only on the text you receive.

[Behavioral Persona Definition]
A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively.

[Metric: Credibility]
Credibility measures how realistic and authentic the persona feels as a depiction of a real user, without sounding exaggerated.

[What to Consider]
Use the following items as your evaluation checklist.
- The persona seems like a real person.
- I can easily imagine a real user behaving like this persona.
- The described tendencies feel plausible and human-like (not overly “designed” or story-like).
- The persona seems to have a distinct personality expressed through its patterns and choices.

[Credibility Scoring] 
Assign a single score from 1 to 5 using the rubric below

1 — Not Credible
- Feels artificial, exaggerated, or fictional
- Behaviors/tendencies seem implausible for a real user
- Persona reads like a generic caricature

2 — Low Credibility
- Weak sense of a real, distinct person
- Some plausible elements, but overall feels awkward or “constructed”
- Contains multiple questionable or overly story-like claims

3 — Moderate Credibility
- Generally plausible and human-like
- May feel generic or slightly “designed” in places
- Distinctness is present but not strong

4 — High Credibility
- Believable and realistic depiction of a real user
- Natural, plausible tendencies with minimal exaggeration
- Distinct personality comes through clearly

5 — Exceptional Credibility
- Extremely authentic and convincingly real
- Highly plausible, natural, and nuanced tendencies
- Strong, distinctive personality with no exaggerated or artificial feel

[Output Format]
Return JSON only with:
- "reasoning": brief explanation (1–2 sentences)
- "score": integer (1–5)

Now evaluate the following behavioral persona for Credibility using the rubric above, then return JSON only with "reasoning" and "score":

[Behavioral Persona Input]
"""

BEHAVIOR_PERSONA_CONSISTENCY = """
[System Instruction]
You are an expert evaluator of behavioral personas for recommender systems. Your job is to judge the Consistency of the provided persona based only on the text you receive.

[Behavioral Persona Definition]
A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively.

[Metric: Consistency]
Consistency measures how well the information within the persona aligns without contradictions across stated tendencies, examples, and different sections of the profile.

[What to Consider]
Use the following items as your evaluation checklist.
- The persona’s examples match other information shown in the persona profile.
- The persona’s stated likes and dislikes do not conflict in an unexplained way.
- The persona information seems consistent across different parts of the profile.
- Any provided background or contextual attributes (if present) correspond with the rest of the persona profile.

[Consistency Scoring] 
Assign a single score from 1 to 5 using the rubric below

1 — Not Consistent
- Clear contradictions between sections or statements
- Examples conflict with stated tendencies or preferences
- Background/context (if present) clashes with other information

2 — Low Consistency
- Multiple inconsistencies or unresolved conflicts
- Noticeable mismatch between examples and summarized traits
- Several statements feel hard to reconcile without explanation

3 — Moderate Consistency
- Mostly aligned information with some minor conflicts or ambiguities
- Occasional mismatches that slightly reduce coherence
- Overall story is understandable but not fully tight

4 — High Consistency
- Information aligns well across sections with no major contradictions
- Examples support the stated tendencies and preferences
- Any context/background (if present) fits the rest of the profile

5 — Exceptional Consistency
- Fully coherent and tightly aligned across all sections
- Examples consistently reinforce the persona’s stated tendencies
- No contradictions or unresolved conflicts; reads as a unified profile

[Output Format]
Return JSON only with:
- "reasoning": brief explanation (1–2 sentences)
- "score": integer (1–5)

Now evaluate the following behavioral persona for Consistency using the rubric above, then return JSON only with "reasoning" and "score":

[Behavioral Persona Input]
"""

BEHAVIOR_PERSONA_COMPLETENESS = """
[System Instruction]
You are an expert evaluator of behavioral personas for recommender systems. Your job is to judge the Completeness of the provided persona based only on the text you receive.

[Behavioral Persona Definition]
A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively.

[Metric: Completeness]
Completeness measures whether the behavioral persona includes the essential information needed to understand the user and make practical personalization decisions, without major missing pieces.

[What to Consider]
Use the following items as your evaluation checklist.
- The persona profile is detailed enough to support meaningful personalization decisions.
- The persona profile seems complete for its intended purpose.
- The persona profile provides enough information to understand what this user tends to prefer and avoid.
- The persona profile is not missing vital information needed for decision-making.

[Completeness Scoring] 
Assign a single score from 1 to 5 using the rubric below. Evaluate completeness by coverage of essential elements, not by length or verbosity.

1 — Not Complete
- Missing most essential information needed to understand the user
- Too sparse to support personalization decisions
- Key preference/avoidance information is absent or unclear

2 — Low Completeness
- Includes a few relevant details but large gaps remain
- Hard to make practical personalization decisions from the profile
- Important aspects of what the user tends to prefer/avoid are missing

3 — Moderate Completeness
- Covers core information but leaves some meaningful gaps
- Sufficient for basic personalization, but not robust
- Some key details needed for confident decision-making are absent or vague

4 — High Completeness
- Provides most essential information needed for practical personalization
- Clear view of what the user tends to prefer and avoid
- Only minor, non-critical details appear missing

5 — Exceptional Completeness
- Richly covers essential information for confident personalization decisions
- Clear, well-rounded understanding of preferences and avoidances
- No major missing pieces for the intended purpose

[Output Format]
Return JSON only with:
- "reasoning": brief explanation (1–2 sentences)
- "score": integer (1–5)

Now evaluate the following behavioral persona for Completeness using the rubric above, then return JSON only with "reasoning" and "score":

[Behavioral Persona Input]
"""

BEHAVIOR_PERSONA_CLARITY = """
[System Instruction]
You are an expert evaluator of behavioral personas for recommender systems. Your job is to judge the Clarity of the provided persona based only on the text you receive.

[Behavioral Persona Definition]
A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively.

[Metric: Clarity]
Clarity measures how clearly and unambiguously the persona is presented so different evaluators can interpret it similarly and quickly.

[What to Consider]
Use the following items as your evaluation checklist.
- The information about the persona is well presented (organized and scannable).
- The text in the persona profile is clear enough to read.
- The information in the persona profile is easy to understand (specific, not vague).
- The persona is memorable (has a clear “gist” that’s easy to recall).

[Clarity Scoring] 
Assign a single score from 1 to 5 using the rubric below

1 — Not Clear
- Disorganized or difficult to scan
- Hard to read or interpret; vague or confusing phrasing dominates
- No clear “gist” of the user emerges

2 — Low Clarity
- Some understandable parts, but overall unclear or cluttered
- Frequent ambiguity, jargon, or undefined terms
- The main point is hard to summarize or recall

3 — Moderate Clarity
- Generally readable and understandable
- Some sections are vague, dense, or open to interpretation
- The gist is present but not sharply memorable

4 — High Clarity
- Well organized and easy to scan
- Clear, specific wording with minimal ambiguity
- A strong gist is easy to understand and recall

5 — Exceptional Clarity
- Extremely well structured, concise, and unambiguous
- Highly readable; evaluators are likely to interpret it similarly
- Very memorable, with a crisp and easily recalled gist

[Output Format]
Return JSON only with:
- "reasoning": brief explanation (1–2 sentences)
- "score": integer (1–5)

Now evaluate the following behavioral persona for Clarity using the rubric above, then return JSON only with "reasoning" and "score":

[Behavioral Persona Input]
"""

BEHAVIOR_PERSONA_IMMERSION = """
[System Instruction]
You are an expert evaluator of behavioral personas for recommender systems. Your job is to judge the Immersion of the provided persona based only on the text you receive.

[Behavioral Persona Definition]
A behavioral persona is a structured user representation derived from observable behavior logs, capturing an individual as a multi-dimensional combination of recurring behavioral tendencies—including revealed preferences, usage patterns, and decision drivers—so that a recommendation model can personalize decisions more effectively.

[Metric: Immersion]
Immersion measures how easily the evaluator can mentally model the user behind the persona—forming an intuitive sense of who they are and how they would respond in new situations.

[What to Consider]
Use the following items as your evaluation checklist.
- I can anticipate what this persona would likely choose or avoid in a new scenario.
- I can infer the persona’s decision tendencies (what they prioritize when choosing).
- I can apply this persona to generate plausible recommendations and exclusions.
- I can simulate how the persona’s choices would shift when the context or options change.

[Immersion Scoring] 
Assign a single score from 1 to 5 using the rubric below

1 — Not Immersive
- The persona is too abstract to simulate in practice
- Cannot anticipate choices/avoidances even in simple new scenarios
- Decision tendencies are unclear or missing

2 — Low Immersion
- A vague mental model is possible, but predictions are unreliable
- Hard to infer what the persona prioritizes when choosing
- Difficult to generate plausible recommendations/exclusions

3 — Moderate Immersion
- A workable mental model can be formed for common situations
- Some decision tendencies are inferable, but gaps remain
- Can generate basic recommendations/exclusions, with uncertainty

4 — High Immersion
- The persona supports strong prediction in many new scenarios
- Decision tendencies are clear and usable for reasoning
- Recommendations/exclusions are easy to generate and feel plausible
- Can simulate how choices shift with contextual changes

5 — Exceptional Immersion
- The persona enables confident simulation across varied scenarios
- Decision tendencies are crisp, nuanced, and consistently applicable
- Can generate highly plausible recommendations/exclusions with clear rationale
- Context-dependent shifts in choices are easy to anticipate and explain

[Output Format]
Return JSON only with:
- "reasoning": brief explanation (1–2 sentences)
- "score": integer (1–5)

Now evaluate the following behavioral persona for Immersion using the rubric above, then return JSON only with "reasoning" and "score":

[Behavioral Persona Input]
"""
