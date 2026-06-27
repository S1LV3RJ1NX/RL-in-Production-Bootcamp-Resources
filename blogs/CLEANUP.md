# CLEANUP.md — the pre-publish cleanup pass for a blog

A draft is "written" when the ideas are down. It is "done" only after this pass. Run the
steps **in order** on `blogs/NN-slug/README.md`. Each step maps to a rule in
[`BLOG_RULES.md`](./BLOG_RULES.md); the section numbers in parentheses point there. Do not
skip steps: later steps assume earlier ones are finished (for example, the worked-example
moves in step 6 assume the code outputs in step 1 are already locked).

> Mindset: you are turning a correct draft into a teaching artifact. Every formula gets
> decoded, every figure gets a reason to exist, every snippet runs and prints exactly what
> the post claims, and the reader can follow one thread from intuition to code.

---

## 1. Lock the code and its output (do this first) — §6

Outputs anchor the prose, so make them real and reproducible before editing words.

1. Copy every runnable snippet into a scratchpad (e.g. `self-notes/script.py`) and run it
   with `uv run python ...` (or `python` if the env is already set up).
2. **Seed everything** so the run is deterministic: `np.random.seed(...)`,
   `torch.manual_seed(...)`, and each env via `env.reset(seed=...)` /
   `env.action_space.seed(...)`. Remember `env.action_space.sample()` has its own RNG.
3. If a stochastic run lands on a misleading result (a seed that converges to the wrong
   answer), **sweep a handful of seeds** and pick one whose behavior matches the narrative,
   then hard-code that seed.
4. Run the snippet **twice** and confirm identical output.
5. Paste the **actual captured stdout** into the ` ```text title="Output" ` block (never
   hand-type it), and add a one-sentence interpretation under it.
6. Delete the scratchpad changes from any committed file once done.

## 2. Move every comment above its line — §6

No trailing inline comments. For each ` ```python ` block, move any `code  # comment` so the
comment sits on its own line directly above the code. Keep full-line section separators
(`# ── Policy network ──`) as they are. After this step, a search for trailing comments
(code followed by `#`) should return nothing.

## 3. Decode every display equation: read, then interpret — §4

For each `$$…$$`, ensure the lines right below it do two things, in order:

1. **Read it aloud, symbol by symbol** ("the left side is …; on the right, $r$ is …,
   $\gamma$ is …").
2. **Plain-English meaning**: what the result means, and what goes up or down when something
   changes.

A chain of algebra steps can share **one** read-then-interpret pass at the end of the chain.
No formula should sit bare with no decoding beneath it.

## 4. Give every figure a paragraph — §5

Walk every image and the Mermaid diagram top to bottom. Each one needs an explanatory
paragraph **immediately after it** that says what the picture shows and **how it ties back to
the concept just taught**. Two stacked figures need two paragraphs, one each. A figure that
flows straight into a code lead-in or a new topic is missing its paragraph; add it. Confirm
alt text is a descriptive sentence and paths are relative (`./images/...`).

## 5. Distribute worked examples inline; delete the standalone section — §1, §3

The old "Worked examples by hand" section is gone in this series. Move each hand-worked
numeric example into the Section 2 subsection of the concept it illustrates, right after that
concept's explanation and code, so the reader meets the numbers while the idea is fresh.

- Fold a one-update / one-step hand calculation into the subsection whose formula it
  concretizes (e.g. the per-logit update example belongs with the backward-pass section).
- A multi-method empirical comparison (a "variance ladder", an ablation table) becomes its
  own `### 2.x` subsection near the method it compares, not a separate top-level section.
- After moving everything, **delete the `## 3. Worked examples by hand` header**, renumber
  the promoted subsections into the `2.x` sequence, and fix any "Section 3.x" cross-references.

## 6. Replace the capstone with a recap table — §1, §6

There is **no end-of-post capstone code dump**. "Putting it all together" must contain only:

1. The `Concept → Math → Code` recap table (a summary of snippets already shown inline).
2. One or two sentences noting the full runnable program lives in the assignment, with a
   markdown link to the assignment notebook.

Steps when removing an existing capstone:
- Confirm the inline sections already contain the complete, runnable program (usually the
  main algorithm's subsection does). If they do not, promote the missing piece inline first.
- Delete the capstone ` ```python ` block and its output block.
- Re-home any understanding checks that referenced "the capstone" (reword "in the capstone"
  to "in the <section> loop" and keep them).

## 7. Simplify prose and smooth the flow — §8

- Split long, multi-clause sentences into two or three short ones. Simple English wins.
- **No em dashes (`—`)** in prose. Use a comma, colon, parentheses, or a period instead.
  (The math minus `−` and en-dash numeric ranges like `B1–B5` are fine.)
- Watch for formatting artifacts a prettier pass can introduce (e.g. a broken `*word\*`
  italic); fix them to `_word_`.
- Add a one-line **bridge** between sections: end a section by motivating the next, or open
  one by recalling the last, so the post reads as a single thread.

## 8. Forward link and cross-references — §1, §1a

- "Where this goes next" ends with a markdown link to the next post **by its `shortName`**
  (e.g. "the [TRPO & PPO](../06-trpo-ppo/README.md) post").
- Every reference to another post uses its `shortName` + link, never "Blog N".
- Re-check `date` in the frontmatter is the real publish date, not copied from a sibling.

## 9. Final lint against the checklist — §9

Run the **Pre-publish checklist** in `BLOG_RULES.md §9` line by line. Then regenerate figures
if any changed: `uv run python blogs/assets/figures.py --blog NN`.

## 10. Publish (sync to the portfolio) — §7

Only when the post is going live, run the sync command from `BLOG_RULES.md §7` (set `SLUG`),
verify `npm run build` passes in the portfolio, then commit and push both repos.

---

### One-line summary of the order

lock outputs → comments above lines → decode equations → figure paragraphs → worked examples
inline → recap table (no capstone) → simplify prose + flow → forward links → checklist →
sync.
