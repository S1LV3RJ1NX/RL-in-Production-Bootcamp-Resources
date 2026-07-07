# Phase 2 — Research Projects, Built End to End

Phase 1 taught the algorithms. **Phase 2** is where we build complete research projects — each taken from an idea to a paper — following the same skeleton every time: **environment → reward → data → method → harness → an honest result.**

Every project ships with a lecture deck, a hand-illustrated companion book, a paper, and the full reproducible code in its folder here.

| # | Project | What you build | Code |
|---|---|---|---|
| 01 | **Socratic Alignment of Small Language Models** | Align a small LM to *withhold* the answer and guide with a question, scored by a verifiable leakage reward, an LLM judge, and real learning-gain. Nine recipes (SFT · DPO · KTO · ORPO · SimPO · GRPO · PPO) — the crossroads where Lecture 06's GRPO meets Lecture 05's PPO. | [`socratic-alignment/`](./socratic-alignment/) |
| 02 | Dream to Catch — an IRIS World Model | *(companion book + paper on the course site)* | — |

## Project 01 · Socratic Alignment

- 📊 **Slides:** <https://rl-bootcamp-decks.vercel.app/lecture-p2-socratic/>
- 📖 **Companion book (14 illustrated chapters):** <https://rl-bootcamp-decks.vercel.app/book-socratic/>
- 📄 **Paper (PDF):** <https://rl-bootcamp-decks.vercel.app/pdfs/socratic-alignment-paper.pdf>
- 💻 **Reproduce it:** [`socratic-alignment/`](./socratic-alignment/) — see its README for the Modal run commands.

**The finding:** a clean, verifiable reward (GRPO) can still *lose* to a cheap offline preference method on a one-sided "don't reveal" constraint. Pure-contrast DPO learns to evade; the anchored offline methods (SFT/KTO/ORPO/SimPO) are the ones that actually learn to guide.

---

© Vizuara AI Labs · 2026 · RL in Production, Cohort 2026.
