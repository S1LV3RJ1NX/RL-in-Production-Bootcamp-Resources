# SWE-RL — all resources

## Learn
- **Slides (lecture deck):** https://rl-bootcamp-decks.vercel.app/lecture-p2-teaching-machines-to-code/
- **Companion book — "Teaching Machines to Code":** https://teaching-machines-to-code.vercel.app
  (16 chapters, from the basics; every figure in the deck comes from it)

## Project sites
- **Project 2 — research site:** https://swe-rl-ipr.vercel.app (paper + code + results)
- **Project 2 — beginner before/after demo:** https://rl-teaches-code.vercel.app

## Code
- **Project 1 — Mini-SWE-RL:** https://github.com/RajatDandekar/Mini-SWE-RL
- **Project 2 — agentic RL (swe-rl-ipr):** https://github.com/RajatDandekar/Mini-SWE-RL/tree/main/swe-rl-ipr
- **This folder (deck + book source):** `phase-2/swe-rl/` in this repo

## Paper
- **Project 2 research paper (PDF):** `paper/swe-rl-ipr-paper.pdf` · also live at
  https://swe-rl-ipr.vercel.app/assets/ipr-paper.pdf

## Key facts (grounded — see `book/FACTS.md`)
- **Mini-SWE-RL:** Qwen2.5-Coder-1.5B, Apple M4 Pro, ~30 min; 66.7% → 73.3%; 7 new bugs.
- **Agentic RL:** MBPP+ (378 tasks), Qwen2.5-Coder 0.5B–7B on Modal H100s; 0.5B 44 → 51 solved, 14 fixed.
- **ECHO:** replication of arXiv 2605.24517; `L_ECHO = L_GRPO + 0.05·L_env`; TerminalBench-2.0 (89 tasks);
  ECHO ≈ 2× GRPO in a controlled A/B (full runs in progress).
- **Algorithm throughout:** GRPO (Group Relative Policy Optimization) — the same recipe behind DeepSWE.
