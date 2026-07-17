# LinkedIn Post — Blog 10: Socratic Alignment

## Schedule

- **Date:** Tuesday, August 25, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

I ran seven alignment recipes on the same data, same model, same budget, and the most respected one came last. DPO produced the best leakage number in the whole study and the worst tutor.

Blog 10 is live: "Socratic Alignment: Teaching a Small Model to Withhold the Answer."

The task inverts everything alignment usually does. Instead of training a model to say more of the useful thing, train it to hold the answer back and ask a guiding question instead. That inversion has a trap: "never reveal the answer" is a one-sided constraint, and the cheapest way to obey it is to say nothing useful at all.

What the numbers showed (Qwen2.5-0.5B, 3 seeds, 71 GPU configs):

- SFT, KTO, ORPO, and SimPO all threaded the needle: judge score up 15 to 19 points, leakage cut 4 to 6 times.
- DPO hit the lowest leakage in the study (0.056) and scored 8 points below the untrained base on teaching. Its winning reply, in full: "Sure, I can help you with that!" Purely contrastive losses can win by crushing the bad reply instead of learning the good one.
- GRPO and PPO, the online RL heavyweights, barely moved. Sparse one-scalar rewards lost to dense token-level supervision at this budget.

The split was predictable from one property: does the objective keep an anchor toward producing the good behavior, or does it only care about a contrast?

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #LLMs #AI #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/10-socratic-alignment/

It builds the whole study from scratch: a simulated classroom where a weak student model gets tutored, a three-layer scorecard (a ten-line auditable leakage rule, a five-axis LLM judge, and a real learning-gain measured in the student's head), a generated preference dataset with a held-out benchmark, and all seven recipes under one fixed LoRA budget. Every number regenerates for a few dollars on Modal.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients
6. TRPO and PPO
7. RLHF
8. GRPO
9. DPO and Agentic RL
10. Socratic Alignment (this one, the production case study)

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog10/blog10-social-cover.png` — series-style diagram cover: dark navy with a faint grid, neon nodes (LEAKAGE RULE, LLM JUDGE, LEARNING GAIN) converging into a glowing "SOCRATIC TUTOR" node, title and subtitle below (recommended hero)
2. **Guide vs evade scatter**: `marketing/blog10/fig-guide-evade.png` — judge score against leakage rate; anchored recipes cluster high-and-left, DPO sits alone at the evade corner, GRPO and PPO hug the base lines (the headline figure)
3. **Delta judge**: `marketing/blog10/fig-delta-judge.png` — judge-score change per recipe, four gainers and three losers
4. **Judge axes**: `marketing/blog10/fig-judge-axes.png` — the five-axis rubric; an evader maxes only "withholds" and caps near 37 of 100
5. **SimPO across sizes**: `marketing/blog10/fig-simpo-sizes.png` — SimPO delivers at 360M where other recipes stall
6. **Blog hero (fallback)**: `blogs/10-socratic-alignment/images/ai-hero.png` — a tightrope walker between the revealed answer and the hollow non-answer

Recommended: lead with `blog10-social-cover.png`, or use `fig-guide-evade.png` if you want the three-cluster result front and center. A carousel works well: slide 1 the "best leakage, worst tutor" hook, slide 2 the one-sided constraint trap, slide 3 the +1.8 tie between a real question and a hollow one, slide 4 the results scatter, final slide the anchored-vs-contrastive lesson plus link.
