# BLOG_RULES.md — how to write a blog in this series

This is the reusable recipe for every RL blog in `blogs/`. Drop the next lecture's
notes beside it, follow this spec, and the output will match the rest of the series
in structure, math style, figures, and code. The goal of each post: take a reader
from **intuition → math → a worked example → code**, with Gymnasium as the end target.

> **The series throughline (repeat it in every post):**
> *The value of where I am is the reward I just got, plus a discounted value of where I'll land next.*
> Tie each new idea back to this sentence.

---

## 1. The 4-part spine (every blog, in this order)

1. **Intuition** — explain the idea in plain language first, with one AI hero illustration
   and a Mermaid diagram where a flow/loop helps. No equations before the reader has a
   mental picture. Use one concrete, lived example and stick with it across the post
   (the series reuses the Pune commute, tic-tac-toe, and a mobile-game level).
2. **The math you need — with code next to each idea.** The key equations in good LaTeX,
   **one clean derivation** (not a wall of proofs); define every symbol on first use.
   **Show the code for a concept immediately after its explanation**, not batched at the
   end — a few lines of runnable Python right under the equation it implements, with the
   line that maps to the math annotated. Code follows the explanation; never make the
   reader scroll to a code dump to connect a formula to its implementation.
3. **Worked example(s) by hand, placed inline.** Concretise every important formula with
   real numbers, solved step by step, and add a code-generated figure when the numbers tell
   a story. **Put each worked example directly in the section of the concept it illustrates**
   (right after that concept's explanation and code), not collected in a separate section at
   the end. A reader should meet the numbers while the idea is still fresh.
4. **Putting it all together — a recap table, not a capstone code dump.** Close the math with
   a short `Concept → Math → Code` table that summarises the pieces already shown inline. Do
   **not** paste a large end-of-post program. The full end-to-end **runnable Gymnasium**
   implementation lives in the post's GitHub assignment notebook; link to it from this section
   (and from "Practice"). Every per-concept snippet has already appeared inline next to its
   explanation, so the table plus the assignment link is enough.

Close with a short **"Where this goes next"** that sets up the following post, states the next
equation the reader will derive when natural, and **ends with a markdown link to the next post
by its `shortName`** (e.g. "the [TRPO & PPO](../06-trpo-ppo/README.md) post"), mirroring how
each post links forward to the one after it.

### 1a. Depth rule — expand the new, recall-and-link the old

Source slides are usually terse one-liners. **Expand every one-liner about a _new_ idea into a
full explanation:** state it, give the intuition, work the *why*. No new term is used before it
is defined; no new equation appears without the one line of algebra a reader could get stuck on.
Fold companion notes/PDFs into the prose as depth rather than citing them.

**But never re-teach a concept an earlier post already covered.** For anything established in a
previous blog (e.g. the Bellman equation, discounting, the TD update, bootstrapping, the Markov
property, GPI), give a **one-line recall plus a markdown link** to that post and immediately use
it; do not re-derive it. The depth budget belongs to what is genuinely new in *this* post. This
keeps each post focused instead of ballooning into a re-run of the series so far.

**Reference other posts by their `shortName`, never by number.** Write "the [MDPs & Bellman](...)
post" or "as we saw in [DP, MC & TD](...)", not "Blog 2" or "the previous blog". Numbers go stale
when posts are reordered, and a name tells the reader what the link is about before they click.
The link text is the target post's `shortName` from its frontmatter.

### 1b. Understanding checks — include every Q&A from the source

If the source material (slides, companion PDFs, or the matching assignment) contains quiz or
self-check Q&A, **include all of it.** Render each as a hidden-answer check so the reader can
self-test first:

```markdown
<details>
<summary><strong>Check:</strong> one-sentence question?</summary>

**Answer.** The answer, 1–3 sentences, in this post's own wording.
</details>
```

Two formatting rules for checks:

- **Summary stays clean:** the visible line is just `Check:` plus the question. Do **not** tag it
  with the source (no "(from the assignment)", "(B4 exercise)", "(ablation A)", etc.); the reader
  doesn't care where it came from.
- **Revealed answer leads with a bold keyword:** the hidden text starts with **`Answer.`** (or
  **`Explanation.`** when it's more of a walkthrough than a single answer), so the resolution is
  obvious the moment the block expands.

**No `$…$` math in the `<summary>` line.** The summary is raw HTML, so KaTeX never processes it
and `$\gamma$` renders as the literal characters `$\gamma$`. Write the question with plain text
and Unicode instead (e.g. `γ < 1`, `V(s)`, `Q(s, a)`, `max_a Q`, `10^170`, `s′`). Math `$…$` is
fine in the **answer body** (it sits after a blank line, so it's normal markdown and renders).
Likewise, keep any single display equation narrow enough to fit: split a long one across lines
with `\begin{aligned} … \\ … \end{aligned}`, since display math does not wrap and overflows
get clipped on the right.

Place each check **immediately after the section it tests** (mirroring how lectures interleave
them), and collect any that don't map cleanly into a single **"Check your understanding"**
appendix just before "Where this goes next." Reword answers to match the post's prose and
notation; pull in answers we wrote in the assignment (e.g. ablation analyses) as checks too.

---

## 2. Folder & file layout

```
blogs/
  BLOG_RULES.md                     <- this file
  README.md                         <- index of all posts
  assets/figures.py                 <- ONE script regenerates every fig-*.svg
  NN-slug/
    README.md                       <- the post (markdown, source of truth)
    images/
      ai-*.png                      <- AI-generated conceptual illustrations
      fig-*.svg                     <- code-generated diagrams (matplotlib/seaborn)
```

- One folder per post, numbered: `01-...`, `02-...`. The post body is `README.md`
  so it renders on GitHub as the folder's landing page.
- The **RL repo is the source of truth.** Publishing = sync a copy of the markdown
  and images into the portfolio (see §7).

---

## 3. Frontmatter (required)

Start every post with YAML frontmatter — it drives the portfolio's content collection:

```yaml
---
title: "..."          # sentence case, specific, no clickbait
shortName: "..."      # 2-4 word handle other posts use to link here (e.g. "MDPs & Bellman")
date: "YYYY-MM-DD"    # the actual publish date (today when you ship). Set it fresh, never copy the previous post's date
summary: "1-3 sentences. What the reader will be able to do after reading."
tags: ["reinforcement-learning", "..."]
order: N               # matches the NN- folder prefix
---
```

`shortName` is how the series cross-references itself (see §1a): every post links to others
by this handle, never by "Blog 2".

`date` must be the **current date on the day you publish** (the day you run the sync command),
in `YYYY-MM-DD`. Do not inherit it from the copy-paste skeleton or a sibling post: each post
carries its own real publish date, so the series never shows every entry on the same day.

---

## 4. Math / LaTeX conventions

- Inline math with `$...$`, display math with `$$...$$` (GitHub + KaTeX both render these).
- **Define every symbol the first time it appears.** Never assume notation.
- Prefer one **clean derivation** over formal rigor. Show the algebra step the reader
  would otherwise get stuck on; push full proofs to a footnote or a linked companion.
- Use standard RL notation consistently across the series:
  $\pi$ policy, $r$/$R$ reward, $G_t$ return, $V$/$Q$ value, $\gamma$ discount,
  $\alpha$ step size, $\mathbb{E}$ expectation, $s,a,s'$ state/action/next-state.
- Keep equations small and frequent rather than one giant block.
- **Every display equation gets a two-step explanation: read it, then interpret it.** Right
  after `$$…$$`, first **read the equation aloud symbol by symbol** (name each term: "the left
  side is …; on the right, $r$ is …, $\gamma$ is …"), then give the **plain-English meaning**:
  what the result *means* and what goes up or down when something changes. Teach the reader to
  decode the notation first, then the takeaway. The reader should never see a formula without
  this read-then-interpret pair immediately below it. (Intermediate algebra steps in a single
  derivation chain can share one read-then-interpret pass at the end of the chain.)

---

## 5. Image conventions

Two kinds of images, two clear jobs. **Default to SVG; reserve interactivity.**

- **`ai-*.png` — conceptual/intuitive illustrations** (AI-generated). Use for the hero
  and for "feel" images with **no numbers or precise labels** (AI tools garble text/math).
  - Prompt for the house palette: paper `#FAFAFA` background, near-black `#0A0A0A` line
    work, terracotta accent `#C8421A`, soft gray `#E5E5E5`. Flat/editorial, lots of
    negative space, **explicitly say "no text or letters."**
- **`fig-*.svg` — anything with numbers, axes, or math** (code-generated). Always produced
  by `blogs/assets/figures.py` so they are reproducible and consistent. SVG = vector,
  tiny, renders on GitHub and in the static site with zero JavaScript.
- **Mermaid** for flows/loops/decision trees — write it as a ```mermaid code block.
- **Plotly / interactive** — only for a single "hero" figure where interaction genuinely
  teaches something (e.g. a convergence slider). It does **not** render on GitHub markdown
  and needs MDX + JS in the site, so use sparingly and never as the default.
- Reference images with **relative paths**: `![alt text](./images/fig-foo.svg)`.
- Every image needs **descriptive alt text** (a sentence, not "figure 1").
- **Every diagram or figure gets an explanatory paragraph** right after it (this includes
  Mermaid diagrams). Say what the picture shows and, explicitly, **how it ties back to the
  concept just taught**. Never drop an image and move on; a figure with no prose around it
  leaves the reader to guess why it is there.

### `figures.py` rules
- Headless (`matplotlib.use("Agg")`), seaborn `whitegrid` theme, the shared `house_style()`.
- Palette constants: `INK #0A0A0A`, `ACCENT #C8421A`, `MUTED #525252`, `DIVIDER #E5E5E5`,
  `CANVAS #FAFAFA`. `svg.fonttype = "none"` so text stays selectable/light.
- Save with `save(fig, blog_folder, name)` → `blogs/NN-slug/images/fig-name.svg`.
- Register each post's builder in `BUILDERS`; support `--blog NN` to rebuild one post.
- Run: `uv run python blogs/assets/figures.py [--blog NN]`.

---

## 6. Code conventions

- **Python only.** Every snippet must be **runnable** as written (test it before publishing).
- **Code follows the explanation.** Put each snippet *immediately after* the concept it
  implements (usually inside the relevant Section 2 subsection), so the reader
  never has to scroll away to connect a formula to its code. Do **not** collect all code
  into one section at the end.
- Keep the snippets small (a few lines per idea).
- **Comments go on the line above the code, never beside it.** Do not put trailing inline
  comments after a statement; place the comment on its own line directly above. This includes
  the comment that maps a line to its equation, e.g.:

```python
# G_t = Σ γ^k r  (accumulate the discounted return)
G += discount * reward
```

- **Make output reproducible: seed everything.** Before any randomness, set the seeds
  (`np.random.seed(...)`, `torch.manual_seed(...)`, and the env via `env.reset(seed=...)` /
  `env.action_space.seed(...)`), so the captured output matches on a re-run. Note that
  `env.action_space.sample()` uses its own RNG, so seed it explicitly when you use it.
- **Show real output.** After every runnable snippet add a ` ```text title="Output" ` block
  containing the **actual stdout captured by running the code** (never hand-typed or guessed),
  followed by a **one-sentence interpretation** of the numbers. Result-value comments inside
  the code (`# 3.5`, `# ≈ 3.5`) move into the Output block.
- **No end-of-post capstone code block.** The integrative end-to-end **Gymnasium** program
  lives in the post's GitHub assignment notebook, not in the post. "Putting it all together"
  contains only the `Concept → Math → Code` **recap** table (a summary of code already shown
  inline) plus a link to that assignment for the full runnable program.
- Reuse the repo's runnable code where it exists: [`code/mars-rover/`](../code/mars-rover/),
  [`code/tic-tac-toe/`](../code/tic-tac-toe/). Link to it rather than duplicating.
- Run snippets with `uv run python ...` (the repo is a `uv` project).

---

## 7. Publishing to the portfolio (Astro → Cloudflare Pages)

The portfolio at `/Users/prathamesh/portfolio/site` is Astro 4 + Tailwind, deployed to
Cloudflare Pages. Markdown bodies render via Astro's `<Content />` with Shiki code
highlighting. To host a math+figures post, the site needs (one-time):

1. A `blog` content collection in `src/content/config.ts` (schema = the frontmatter above).
2. Markdown integrations: `@astrojs/mdx`, `remark-math`, `rehype-katex` (+ KaTeX CSS),
   and a Mermaid integration; wired in `astro.config.mjs`.
3. A render page `src/pages/blog/[slug].astro` (clone `case-studies/[slug].astro`) and a
   `/blog` index.

**Per-post publish steps:**
1. Copy `blogs/NN-slug/README.md` → `site/src/content/blog/NN-slug.md`.
2. Copy `blogs/NN-slug/images/*` → `site/public/blogs/NN-slug/` and rewrite image paths
   from `./images/...` to `/blogs/NN-slug/...`.
3. `npm run build` to verify, then commit & push (Cloudflare Pages auto-deploys).
4. Subdomain `blogs.prathameshsaraf.com`: add it as a custom domain on the Pages project
   (or a dedicated Pages project) and add the `blogs` CNAME in Cloudflare DNS.

> Note the **plural** `public/blogs/<slug>/` directory and `/blogs/<slug>/...` URL prefix —
> that is what the existing posts (01–03) and the `[slug]` route use. Content markdown lives
> under the singular `src/content/blog/` collection.

**Concrete sync command (steps 1–2, copy-paste).** Run from the RL repo root; set `SLUG`
to the post folder. It copies the images, copies the markdown, and rewrites the relative
`./images/...` paths to the site's absolute `/blogs/<slug>/...` paths in one go:

```bash
SLUG="04-sarsa-qlearning-dqn"                                  # <- the only thing to change
SITE="/Users/prathamesh/portfolio/site"

mkdir -p "$SITE/public/blogs/$SLUG" "$SITE/src/content/blog"
cp blogs/$SLUG/images/* "$SITE/public/blogs/$SLUG/"            # figures, hero, gifs
sed -e "s#\./images/#/blogs/$SLUG/#g" \
    -e 's#\.\./\([0-9][0-9]-[^/]*\)/README\.md#/blogs/\1#g' \
    blogs/$SLUG/README.md \
    > "$SITE/src/content/blog/$SLUG.md"                        # markdown + path rewrite

cd "$SITE" && npm run build                                    # step 3: verify before pushing
```

Re-running it is idempotent (it overwrites), so it doubles as the "update a published post"
command.

Keep the palette/fonts aligned: canvas `#FAFAFA`, ink `#0A0A0A`, accent `#C8421A`,
Geist / Geist Mono — so figures and site share one visual language.

---

## 8. Tone & style

- Intuition-first, conversational but precise. Short paragraphs. One idea per paragraph.
- **Keep sentences simple.** Prefer short, plain sentences over long, multi-clause ones. If a
  sentence runs long or stacks several ideas, split it into two or three. Simple English beats
  clever phrasing; the reader is here for the concept, not the prose.
- **Flow between sections.** Each section should lead into the next: end a section by motivating
  what is coming, or open one by recalling what just came, so the post reads as one thread of
  thought rather than disconnected blocks.
- Lead with the "why" before the "what." Use a vivid concrete example, then generalize.
- Bold the one takeaway sentence per section. Don't pad.
- No emojis unless asked. American spelling. Define jargon on first use.
- **No em dashes (`—`).** They read as AI-generated. Use the punctuation the sentence actually
  wants instead: a comma for an aside, a colon to introduce, parentheses for a true parenthetical,
  or a period to split into two sentences. (This applies to prose only; the minus sign `−` in math
  and the en dash `–` in numeric ranges like `B1–B5` are fine.)

---

## 9. Pre-publish checklist

- [ ] 4-part spine present and in order; throughline sentence referenced.
- [ ] Every new one-liner expanded; prior-blog concepts recalled in one line + linked, not re-derived (§1a).
- [ ] Other posts referenced by `shortName` + link, never "Blog N" (§1a, §3).
- [ ] No em dashes (`—`) anywhere in prose; proper punctuation used instead (§8).
- [ ] All source Q&A included as hidden-answer checks, placed after the section each tests (§1b).
- [ ] Check summaries are untagged (`Check:` + question only); revealed answers lead with a bold `Answer.`/`Explanation.` (§1b).
- [ ] Frontmatter complete (`title`, `shortName`, `date`, `summary`, `tags`, `order`); `date` is today's real publish date, not copied from the skeleton or a sibling post (§3).
- [ ] Every symbol defined on first use; one clean derivation, no proof dumps.
- [ ] Every display equation is read aloud symbol by symbol, then interpreted in plain English right after it (§4).
- [ ] Long/complex sentences split into simple ones; sections flow into each other (§8).
- [ ] At least one hand-worked numeric example per key formula, placed inline in that concept's section, not in a separate end section (§1).
- [ ] Figures: `ai-*` for concepts, `fig-*.svg` for numbers; all have alt text and relative paths.
- [ ] Every diagram and figure (including Mermaid) has an explanatory paragraph after it tying it to the concept (§5).
- [ ] `figures.py` regenerates all `fig-*.svg` for the post with no errors.
- [ ] Every code snippet runs (`uv run python ...`); equation-to-code mapping shown.
- [ ] Code comments are on the line above, never trailing inline (§6).
- [ ] Randomness is seeded so captured output is reproducible (§6).
- [ ] No end-of-post capstone code; "Putting it all together" is the recap table only, with the runnable program linked from the GitHub assignment (§1, §6).
- [ ] "Where this goes next" ends with a `shortName` link to the next post (§1).
- [ ] Every snippet has a `text title="Output"` block with real, captured stdout + a one-line interpretation.
- [ ] Code is placed inline next to the concept it implements; no end-of-post code dump.
- [ ] "Where this goes next" sets up the following post and links to it by `shortName`.
- [ ] (If publishing) synced to the portfolio, image paths rewritten, `npm run build` passes.

---

## 10. Copy-paste skeleton

```markdown
---
title: "..."
shortName: "..."
date: "YYYY-MM-DD"   # today's date when you publish; do not copy this line's value forward
summary: "..."
tags: ["reinforcement-learning", "..."]
order: N
---

# Title

![hero alt text](./images/ai-hero.png)

> **The throughline:** *The value of where I am is the reward I just got, plus a discounted value of where I'll land next.*

## 1. The intuition
... plain-language explanation, one concrete example, a ```mermaid``` diagram ...

## 2. The math you need
### 2.1 ...
$$ ... $$
... read the equation symbol by symbol, then interpret it; define symbols;
one clean derivation; link back to the throughline ...
![fig alt](./images/fig-foo.svg)
... a paragraph explaining the figure and how it ties to this concept ...

​```python
# comment on the line above maps this line to the equation (G_t = Σ γ^k r)
# seed all randomness so the output below is reproducible
... small runnable snippet for THIS concept, right under its explanation ...
​```

​```text title="Output"
<actual stdout captured by running the snippet>
​```

One-sentence interpretation of the output.

... then the hand-worked numeric example for THIS concept, inline ...

### 2.2 ...
... and so on: every concept gets its code AND its worked example inline ...

## 3. Putting it all together
| Concept | Math | In code |
|---|---|---|
| ... | $...$ | `...` |   <- recap of code already shown inline above

The full end-to-end runnable Gymnasium program lives in the assignment:
> **[Assignment — ...](https://github.com/.../assignments/...ipynb)**

## Where this goes next
... one paragraph + the next equation, ending with a link to the next post
by its shortName, e.g. the [TRPO & PPO](../06-trpo-ppo/README.md) post ...
```
