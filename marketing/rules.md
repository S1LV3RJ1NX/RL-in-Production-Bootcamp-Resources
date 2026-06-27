# Marketing Rules — RL Blog Series

Reusable guidelines for promoting each blog post on LinkedIn and X.

---

## Brand Positioning

**Niche**: RL foundations explained for the LLM engineer.
**Tagline**: "The math behind RLHF/GRPO — intuition first, one derivation, a worked example, then code."
**Differentiator**: Most content is either too academic (Sutton & Barto) or too hand-wavy (Medium posts). We sit in the middle — equations with typed Python, readable by working ML engineers.

---

## Content Formula (per blog)

### LinkedIn (1 long-form post)

Structure:
1. **Hook** (1-2 lines): controversial take, surprising fact, or relatable frustration
2. **The problem** (2-3 lines): what's missing in existing resources
3. **What the post covers** (bullet list with → arrows): 3-5 key concepts
4. **Who it's for** (2-3 lines): target audience explicitly named
5. **CTA**: "Link in comments" (LinkedIn suppresses posts with links in the body)
6. **Comment**: post the link + series progress as the first comment immediately

Formatting rules:
- Max 1300 characters in body (before "see more" cutoff is ~210 chars — front-load the hook)
- Use line breaks generously (mobile readability)
- No more than 5 hashtags, placed at the end
- Always tag @VizuraAI and any relevant people/bootcamp instructors

### X / Twitter — Use Articles (not threads) for blog drops

**Why Articles over Threads (with X Premium):**
- Articles maximize dwell time — the #1 algorithm signal in 2026
- Premium gives 2-4x visibility multiplier (documented in algorithm source code)
- Articles support code blocks, images, bold/italic, and subheadings
- Articles are searchable and bookmarkable long-term (threads decay in search)
- The algorithm rewards native content that keeps people on-platform

**When to still use threads:** live commentary, listicles, story arcs with cliffhangers.

Article structure (aim for 1,000–1,500 words):
1. **Hook paragraph** (first 280 chars show in feed — make them count)
2. **The problem** this blog solves
3. **3-4 sections** with subheadings covering key ideas
4. **Who it's for** (explicit audience callout)
5. **CTA**: link to full blog + "follow for the series"

Formatting rules:
- Keep paragraphs to 1-3 sentences (mobile readability)
- Use **bold** and *italics* liberally (Premium formatting features)
- Embed at least 1 image inline (figures from the blog)
- Always self-reply within 5 minutes with a question to spark engagement
- Quote-repost your own article 4-6 hours later with a different hook
- Tag @VizuraAI in the article body

**The first 30-60 minutes are critical:**
- Replies are the strongest positive signal (heavier than likes or reposts)
- Be ready to engage the second you hit publish
- DM 3-5 relevant people about the article
- Do NOT cluster posts — space them hours apart (Author Diversity penalty)

---

## Visual Strategy

Priority order for images per post:
1. **Hero/cover image** (from the blog's `images/` folder — usually the AI-generated one)
2. **A key figure** from the post (SVG — screenshot or export as PNG for social)
3. **Code screenshot** (use a tool like ray.so or carbon.sh for styled code blocks)
4. **AI-generated image** (only if no good blog figure exists)

### Cover and banner template (per blog)

Every blog gets two AI-generated lead images, kept in that blog's `marketing/blogN/` folder:

- **`blogN-social-cover.png`** for LinkedIn (landscape, ~3:2): diagram on top, title centered below.
- **`blogN-x-banner.png`** for the X article header (wide, ~5:2): diagram on the left, title and subtitle on the right.

Keep them visually consistent across the whole series. The established style is a **flat 2D vector infographic** (not photographic, not painterly, no 3D, no people, no scenery):

- Very dark navy background (around `#0a0e1a`) with a faint, subtle grid pattern.
- A small node diagram of the blog's **one core concept**, built from neon-outlined rounded-rectangle nodes with bold uppercase labels (rotate blue, orange, green), connected by thin glowing arrows into a single glowing cyan-white target node. Examples so far: blog 2 = backup tree (s to a1/a2/a3 to s'), blog 3 = DP/MC/TD converging to V\*, blog 4 = SARSA/Q-learning/DQN converging to Q\*.
- Crisp bold white title (the blog title) with a smaller lighter-gray subtitle beneath it. Modern image models render short titles cleanly, so put the real title in the prompt.
- Generous negative space and soft neon glow.

How to generate: use the image tool and **pass a previous blog's banner (e.g. `marketing/blog3/blog3-x-banner.png`) as a style reference** so the new one matches. Generate the X banner and the social cover as a pair, then drop both into `marketing/blogN/`.

**X does not support SVG.** Most blog figures are SVG, so export a PNG before embedding in an X article. Convert with:

```
rsvg-convert -z 2 path/to/fig.svg -o marketing/blogN/fig.png
```

(`-z 2` doubles the resolution so it stays sharp.) LinkedIn handles PNG fine too, so PNG is the safe default for both platforms.

**X Articles render LaTeX, but only use it for display (block) equations.** On X, LaTeX renders on its own line, so reserve it for standalone equations with `$$...$$` (e.g. `$$V(s) = \sum_a \pi(a \mid s)\, Q(s,a)$$`). Do **not** use inline `$...$` math mid-sentence: it breaks the line awkwardly. Keep inline references as plain text instead (`V(s)`, `p(s'|s,a)`, `π`, `γ`). Use `\mid` for the conditioning bar so it doesn't read as a table pipe, and spell out `\pi`, `\gamma`, `\sum`, `\max`, `\arg\max`, `V^*`.
- X Article: LaTeX for block equations only; plain text inline.
- X plain tweets / self-replies / quote-reposts: no LaTeX, keep equations as plain text (`V*(s) = max_a Q*(s,a)`).
- LinkedIn: no LaTeX either; keep equations as readable plain text or embed an image of the equation.

**X Articles support fenced code blocks.** For code-heavy posts (most of this series), drop a short, self-contained snippet in the Article body with a ```` ```python ```` fence. Keep it tight (10-15 lines), keep the type hints, and add a one-line comment mapping the code to the equation it implements. This reinforces the "typed Python, line by line" angle. LinkedIn does not render code blocks well, so for LinkedIn either screenshot the code (use ray.so / carbon.sh) or describe it in prose.

For carousels (LinkedIn):
- 4-6 slides max
- Slide 1: hook question
- Slides 2-5: one concept per slide (title + 1 sentence + visual)
- Final slide: CTA + link

---

## Hashtags

### LinkedIn
Always include: #ReinforcementLearning #MachineLearning #LearningInPublic
Rotate 2-3 from: #RLHF #LLMs #AI #DeepLearning #PPO #GRPO #DataScience

### X
Skip hashtags in threads (they hurt engagement on X). Use them only in standalone tweets.

---

## Posting Schedule (data-backed for 2026)

**The cadence: a new blog every Tuesday, a follow-up on the same blog every other day.**
Tuesday is launch day (the strongest day for reach). The remaining six days each run one
follow-up post about that same blog, on both LinkedIn and X, until the next Tuesday's launch.
Post around **10:30 AM IST** daily.

**Key findings from 4.8M+ posts analyzed:**
- LinkedIn peak engagement: Tuesday–Thursday, 10 AM–2 PM or 3–5 PM local time
- LinkedIn worst days: Monday (lowest), weekends (slow), so keep those days light follow-ups
- Wednesday is consistently the #1 day across all studies
- X algorithm rewards dwell time above all else: long-form Articles beat threads for tutorials

| Day | Platform | Content |
|-----|----------|---------|
| Tuesday | LinkedIn + X | **Launch the new blog**: long-form LinkedIn post + X Article (10:30 AM IST) |
| Wednesday | LinkedIn + X | Follow-up 1 on the same blog |
| Thursday | LinkedIn + X | Follow-up 2 |
| Friday | LinkedIn + X | Follow-up 3 (also quote-repost Tuesday's X article with a new hook) |
| Saturday | LinkedIn + X | Follow-up 4 (lighter angle) |
| Sunday | LinkedIn + X | Follow-up 5 (lighter angle) |
| Monday | LinkedIn + X | Follow-up 6 (last teaser before Tuesday's new blog) |

Each blog's follow-up posts (Wed–Mon) are pre-written in that blog's
`marketing/blogN/follow-ups.md`, one numbered angle per day, ready to copy-paste.

Best posting times (IST):
- LinkedIn: 10:00 AM–12:00 PM or 3:00–5:00 PM (the Tuesday launch and Wed–Thu follow-ups hit hardest)
- X: 9:00–10:30 AM or 5:00–6:00 PM
- Weekend/Monday follow-ups still go out, just treat them as lighter touches, not the big swings.

---

## Engagement Rules

1. **Reply to every comment** within 30 minutes on X (first hour is critical), within 4 hours on LinkedIn
2. **Self-reply with a question** immediately after posting (sparks the engagement loop)
3. **Engage on others' RL/LLM posts** with genuine insights (not spammy links)
4. **Quote-tweet relevant papers** with "Here's the foundation behind this → [blog link]"
5. **Never post bare links** on LinkedIn body — always "link in comments"
6. **Tag people** who would genuinely find it useful (bootcamp peers, instructors)
7. **Answer with a question** — treat every reply as a chance to keep the dialogue going (signals session depth to algorithm)
8. **Never engagement-bait** — a single block/mute does measurable damage to your post score on X

---

## Series Progress Tracker

Cadence: **one blog per week, launched every Tuesday** (LinkedIn post + X Article the same
day). The other six days (Wed–Mon) run follow-ups on that same blog from its `follow-ups.md`.

| Blog | Title | LinkedIn (launch) | X Article (launch) |
|------|-------|------|------|
| 01 | RL from First Principles | ☐ Thu Jun 25 (~12:45 PM) | ☐ Thu Jun 25 (~12:45 PM) |
| 02 | MDPs & Bellman Equations | ☐ Tue Jun 30 (10:30 AM) | ☐ Tue Jun 30 (10:30 AM) |
| 03 | DP, Monte Carlo & TD | ☐ Tue Jul 7 | ☐ Tue Jul 7 |
| 04 | SARSA, Q-learning & DQN | ☐ Tue Jul 14 | ☐ Tue Jul 14 |
| 05 | Policy Gradients | ☐ Tue Jul 21 | ☐ Tue Jul 21 |
| 06 | TRPO & PPO | ☐ Tue Jul 28 | ☐ Tue Jul 28 |

> Blog 01 was already live (it is the prerequisites post), so it launched on a Thursday and
> gets a short follow-up run; from Blog 02 onward, every launch lands on a Tuesday.

---

## Humanize every post before publishing

All marketing copy (LinkedIn posts, X articles, comments) must be run through the **humanizer skill** before it ships, so the writing doesn't read as AI-generated. The skill lives at `~/.cursor/skills/humanizer/SKILL.md` (source: https://github.com/blader/humanizer) and is based on Wikipedia's "Signs of AI writing" guide.

Apply this to the published copy of every blog (1 through 9). The meta sections (Schedule, Image Suggestions, Formatting Tips) don't need it.

The tells it removes, worth scanning for by hand:
- **Em and en dashes** (`—`, `–`): hard cut. Replace with a period, comma, colon, or parentheses.
- **Arrow-formatted lists** (`→`) and decorative emojis (`🔗`, `✅`, `🚀`): use plain bullets and plain text.
- **Rule of three**: "real math, real code, and zero hand-waving" becomes two items or prose.
- **Em-dash asides and inline `→` chains**: "intuition → derivation → example → code" becomes a sentence.
- **Significance inflation / promo words**: pivotal, vibrant, testament, landscape, showcase, delve, crucial.
- **Copula avoidance**: "serves as / stands as" becomes "is".
- **Negative parallelism**: "It's not just X, it's Y" becomes the direct claim.
- **Filler and hedging**: "in order to" becomes "to"; "could potentially possibly" becomes "may".

After rewriting, scan once more for any remaining `—`/`–` and emojis. A hit means it isn't done.

How to apply: read `~/.cursor/skills/humanizer/SKILL.md`, then rewrite each post following the draft -> audit -> final loop. Keep the author's voice (varied sentence length, opinions, concrete detail); the goal is natural, not sterile.

---

## Virality Playbook (2026, data-backed)

Both platforms now rank primarily on **dwell time** and **early engagement velocity**, not likes. The mechanics below come from LinkedIn's LiRank papers and xAI's open-sourced X algorithm.

### The single most important rule
**Reply to every comment in the first 60 minutes.** On X, an author-engaged reply is weighted **~150x a like** (+75 vs +0.5). On LinkedIn, substantive comments (15+ words) plus the first-hour velocity decide whether you escape your own network. Be online and present the moment you post.

### LinkedIn: stack 4+ hook tactics in the first 200 characters
Viral posts average 4–6 stacked tactics. The highest-leverage combo:
1. **Pattern interrupt** — an unexpected opening line
2. **Quantified proof** — a specific number ("only 1 in 100 episodes reaches the goal")
3. **Open loop** — tease a payoff you resolve later ("here's the one sentence the whole field hangs on")
4. **Memorable quote** — a line worth screenshotting

Rules:
- First line must earn the "see more" tap (140 chars mobile / 210 desktop). This tap *is* the dwell-time signal.
- NO hedging ("I think", "maybe", "kind of") — it kills the hook.
- 150–300 words, line breaks every 1–2 sentences.
- **Optimize for saves** — saves are a lasting quality signal, higher-weighted than likes. End with something worth saving (a cheat-sheet, a formula).
- Links in comments, never the body (external links suppress reach).
- CTA = a specific question that invites thoughtful replies.

### X: engineer "cognitive friction" + the Show More click
- First 280 chars = the whole game. Use a **shocking question** ("Is the math behind RLHF actually simple?") or a **terrifying metric** ("90% of people skip the one equation all of RL is built on").
- Make the Article long enough to trigger truncation; put a cliffhanger right at the cut so people click "Show More" (registered as high-intent engagement).
- Reply substantively to 5–10 larger AI/ML accounts (2–10x your followers) daily — earn a like/reply back. This is the #1 growth lever under 1K followers.
- Bookmarks and quote-posts > likes (they signal "I'll come back to this").
- A single new **follow from a post** outweighs dozens of likes — pin your strongest post.
- Keep links out of the main post; drop in a reply.
- Don't burst-post: space posts hours apart (author-diversity decay).

### Keywords / hashtags that surface in RL-for-LLMs discovery
Use these naturally in copy (not stuffed):
- **Topic terms** (help LinkedIn's topic-relevance retrieval): `Reinforcement Learning`, `RLHF`, `GRPO`, `PPO`, `DPO`, `LLM alignment`, `reward modeling`, `policy gradients`, `Bellman equation`, `value function`
- **LinkedIn hashtags** (3–5 max): `#ReinforcementLearning` `#MachineLearning` `#LLMs` `#RLHF` `#AI` `#DeepLearning` `#LearningInPublic`
- **X**: skip hashtags (they hurt reach); instead name the concepts in plain text and tag relevant accounts/papers.

### Quote-repost your own article the same day

A few hours after the X article goes live (aim for ~5:30-6:00 PM, roughly 4-6 hours after publishing), quote-repost it with a fresh one-line hook. This gives the article a second pass through the feed and catches people who missed the morning slot.

- Hit repost on your own article, choose "Quote," and type a new hook above the embedded post.
- Use a *different* hook than the article's opener (a new angle, not a repeat).
- Stack the usual tactics: open loop, a number, or a pattern interrupt. No emojis, no em dashes.
- Reply to anyone who engages with the quote-repost, same as the first hour of the original.
- Keep per-blog hook options saved in that blog's `x-article.md` under "Quote-repost hooks".

### Ride trending waves
When a new RLHF/GRPO/DPO paper or model drops, quote-post it within hours with: "Here's the foundational math behind this →" + your blog. Attaching expertise to an active discussion beats standalone posts.

---

## Template Variables

Replace these in each blog's marketing copy:
- `[YOUR_PORTFOLIO_URL]` → your portfolio blog base URL
- `[BLOG_SLUG]` → e.g., `01-rl-intro-and-prerequisites`
- `@VizuraAI` → actual handle (verify before posting)
