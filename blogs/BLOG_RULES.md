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
3. **Worked example(s) by hand** — concretise every important formula with real numbers,
   solved step by step. Add a code-generated figure when the numbers tell a story.
4. **A capstone in code** — one end-to-end **runnable Gymnasium program** that combines
   the pieces introduced above, preceded by a short `Concept → Math → Code` recap table.
   This is the only code that lives near the end, because it is *integrative* by nature;
   every per-concept snippet has already appeared inline next to its explanation.

Close with a short **"Where this goes next"** that sets up the following post (and, if
natural, states the next equation the reader will derive).

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
date: "YYYY-MM-DD"
summary: "1–3 sentences. What the reader will be able to do after reading."
tags: ["reinforcement-learning", "..."]
order: N               # matches the NN- folder prefix
---
```

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
  implements (usually inside the relevant Section 2 / Section 3 subsection), so the reader
  never has to scroll away to connect a formula to its code. Do **not** collect all code
  into one section at the end.
- Keep the snippets small — a few lines per idea — and **annotate the line that maps to the
  equation** (e.g. `G += discount * reward  # G_t = Σ γ^k r`).
- **Show real output.** After every runnable snippet add a ` ```text title="Output" ` block
  containing the **actual stdout captured by running the code** (never hand-typed or guessed),
  followed by a **one-sentence interpretation** of the numbers. Result-value comments inside
  the code (`# 3.5`, `# ≈ 3.5`) move into the Output block; equation-mapping comments stay.
- The **one** allowed end-of-post code block is the **capstone**: a single runnable
  **Gymnasium** program that combines the pieces, with the `reset()` / `step()` loop shown
  explicitly. Precede it with a `Concept → Math → Code` **recap** table (the table is a
  summary of code already shown inline, not a substitute for it).
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
2. Copy `blogs/NN-slug/images/*` → `site/public/blog/NN-slug/` and rewrite image paths
   from `./images/...` to `/blog/NN-slug/...`.
3. `npm run build` to verify, then commit & push (Cloudflare Pages auto-deploys).
4. Subdomain `blogs.prathameshsaraf.com`: add it as a custom domain on the Pages project
   (or a dedicated Pages project) and add the `blogs` CNAME in Cloudflare DNS.

Keep the palette/fonts aligned: canvas `#FAFAFA`, ink `#0A0A0A`, accent `#C8421A`,
Geist / Geist Mono — so figures and site share one visual language.

---

## 8. Tone & style

- Intuition-first, conversational but precise. Short paragraphs. One idea per paragraph.
- Lead with the "why" before the "what." Use a vivid concrete example, then generalize.
- Bold the one takeaway sentence per section. Don't pad.
- No emojis unless asked. American spelling. Define jargon on first use.

---

## 9. Pre-publish checklist

- [ ] 4-part spine present and in order; throughline sentence referenced.
- [ ] Frontmatter complete (`title`, `date`, `summary`, `tags`, `order`).
- [ ] Every symbol defined on first use; one clean derivation, no proof dumps.
- [ ] At least one hand-worked numeric example per key formula.
- [ ] Figures: `ai-*` for concepts, `fig-*.svg` for numbers; all have alt text and relative paths.
- [ ] `figures.py` regenerates all `fig-*.svg` for the post with no errors.
- [ ] Every code snippet runs (`uv run python ...`); equation-to-code mapping shown.
- [ ] Every snippet has a `text title="Output"` block with real, captured stdout + a one-line interpretation.
- [ ] Code is placed inline next to the concept it implements; only the integrative Gymnasium capstone sits near the end.
- [ ] "Where this goes next" sets up the following post.
- [ ] (If publishing) synced to the portfolio, image paths rewritten, `npm run build` passes.

---

## 10. Copy-paste skeleton

```markdown
---
title: "..."
date: "YYYY-MM-DD"
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
... define symbols; one clean derivation; link back to the throughline ...
![fig alt](./images/fig-foo.svg)

​```python
# small runnable snippet for THIS concept, placed right under its explanation,
# with the line that maps to the equation annotated
​```

​```text title="Output"
<actual stdout captured by running the snippet>
​```

One-sentence interpretation of the output.

### 2.2 ...
... and so on: every concept gets its code inline, next to the idea ...

## 3. Worked example(s) by hand
... step-by-step numbers ...

## 4. Putting it all together (capstone)
| Concept | Math | In code |
|---|---|---|
| ... | $...$ | `...` |   <- recap of code already shown inline above

​```python
# ONE end-to-end runnable Gymnasium program that combines the pieces,
# with reset()/step() shown explicitly and lines annotated to the equations
​```

​```text title="Output"
<actual stdout captured by running the capstone>
​```

One-sentence interpretation of the capstone output.

## Where this goes next
... one paragraph + the next equation ...
```
