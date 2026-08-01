# STYLE.md — *Teaching Machines to Code* (house style)

Every chapter and every figure follows this spec. Writer agents: do not improvise outside it.
This is a **beginner** book. The reader may never have opened a terminal or trained a model. Warmth and
clarity beat cleverness. Ground everything in real, concrete examples.

---

## PART A — The page (a warm, calm "notebook" page)

- **Two surfaces.** The site shell (home, sidebar) is a dark terminal green. The article pages are the
  opposite: warm off-white paper, serif type, calm and roomy. Terminal outside, notebook inside.
- **Body:** serif (EB Garamond), generous line spacing. A wide right margin holds numbered **sidenotes**.
- **Inline code** in backticks renders **crimson red** — use it for anything a computer would read literally:
  `range(2, n+1, 2)`, `ls`, `assert add(2,2)==4`, `reward = 1.0`, file names, function names, `git`.
- **Bold** a new term on first mention *with a plain-words expansion*: "a **reward** — a single number that
  says how good an attempt was." Then use it plainly.
- **Real numbers live in the prose, in bold:** "the model went from **66.7%** to **73.3%**". Use a small
  table only for a genuine side-by-side (before/after, a config).

## PART B — Voice (patient, from-foundations, honest)

1. **Explain like the reader is smart but brand new.** No assumed background. Every acronym is expanded the
   first time (GRPO = **Group Relative Policy Optimization**), then used plainly. If a sentence needs a term
   we have not defined, define it first.
2. **Lead with a real-life analogy, then the technical version.** Riding a bike leads to the RL loop; a
   hunch before flipping a switch leads to a world model; a recipe that is half a box leads to a
   volume-formula bug. The analogy is scaffolding; always land back on the concrete computing thing.
3. **One idea per section.** Short `##` headings that read like questions or claims ("Why the tests are a
   perfect teacher"). Build up; never dump.
4. **Concrete over abstract, always.** Show the actual buggy line and the actual fixed line. Name the real
   puzzle (`hard_scope_bug`). Quote the real number (`λ = 0.05`). Vague hand-waving is the failure mode.
5. **Be honest about results.** Our real numbers are modest and true — say so plainly ("this is a real but
   modest jump, and that honesty is the point"). Never inflate. Prefer "of the tasks it was failing, RL
   fixed 14" over a shiny aggregate.
6. **Directed figures.** The prose says "here is the loop, drawn out —", the figure follows immediately on
   its own line, and the prose refers back to it. Never drop a figure with no lead-in.
7. **Length:** 1,400–2,400 words per chapter. 4–7 figures. 2–5 sidenotes. Tight and complete beats long.
8. **Open with the stakes in one or two plain sentences**; close with a one-line bridge to the next chapter.

## PART C — Teaching callouts (use them; they are the book's warmth)

Use these sparingly (1–3 per chapter) on their own line. Syntax: `[[note: TYPE || text ]]`. Types:
- `metaphor` — a real-life analogy that unlocks the idea.
- `example` — a tiny worked-by-hand example (numbers, a 2-line code trace).
- `production` — "in the real world" — how the grown-up systems (DeepSWE, real repos) do it.
- `confusion` — a common beginner misconception, named and corrected.
- `aha` — the one-sentence click; the thing to remember.

## PART D — Figure grammar (hand-drawn, like a smart friend at a whiteboard)

Figures are rendered as **hand-drawn Excalidraw-style** ink diagrams with handwritten labels. Write each
`[[fig: … || caption]]` as a SELF-CONTAINED scene someone could draw without reading the chapter — name the
boxes, the arrows, the handwritten labels, and the colors. Syntax, always on its OWN line:

`[[fig: <detailed hand-drawn scene, naming boxes/arrows/labels and their colors> || <one-line caption>]]`

**Semantic colors (keep consistent across the whole book):**
- **black** = structure, titles, the main boxes and labels.
- **blue** = how things move / mechanism / the flow of control or data (arrows, "the model reads this").
- **green** = numbers, results, rewards, scores, sizes ("reward = 1.0", "73.3%", "lambda=0.05", "89 tasks").
- **red** = the bug, the failure, the "before", warnings ("tests fail", "IndexError", "returns a box!").
- **purple** = literal code / commands in a snippet (`range(2, 2*n+1, 2)`, `ls`, `git commit`).
- **orange** = emphasis, the "look here" callout, the single key takeaway.
- **green fill / pale hatch** = the "after", the passing state, success (a green-tinted card).
- **red fill / pale hatch** = the "before", the failing state.

**Recurring archetypes (reuse these compositions):**
1. **The RL loop** — a clean cycle of rounded cards: MODEL (writes something) then ENVIRONMENT / TESTS
   (runs it) then REWARD (a green number) then LEARN (nudge the model) then back to MODEL. Curved blue
   arrows, the reward in green, a dashed takeaway box.
2. **Before / after code cards** — two side-by-side cards, LEFT red-tinted "BEFORE" with a red cross and the
   buggy line in purple, RIGHT green-tinted "AFTER" with a green check and the fixed line in purple; a bold
   labeled arrow between them ("reinforcement learning").
3. **A group of tries (GRPO)** — one problem box at left, a fan-out of eight little attempt cards, each with
   a green reward (1 or 0), a dashed line showing the group average, and up/down arrows (orange) marking
   which beat the average.
4. **A terminal window** — a dark rounded rectangle with a few handwritten monospace command lines in purple
   (`$ ls`, `$ python test.py`) and the computer's reply below; use it to show what an agent sees and types.
5. **A learning curve** — hand-drawn axes (x = training steps, y = fraction solved), a wobbly green line
   rising then flattening, a red dashed "before" level and a green dashed "after" level, orange note on the
   rise.
6. **System / scale map** — boxes for laptop vs cloud GPU (H100), or the ECHO extra-loss wiring (the shared
   forward pass with two arrows out: one to "act", one to "predict the reply"), with green spec numbers.

Rules: render ONLY what the prompt describes; never invent numbers or constants not given; write colors as
*ink*, not as the word (draw a green "73.3%", do not write the word "green"). Wide composition, lots of white
space, flat (no shadows/gradients/photos), clean legible handwriting. Handwritten labels ARE required here.

## PART E — Custom markdown (exact)
- `[[fig: <scene> || <caption>]]` — a figure. MUST start on its own line. A mid-paragraph `[[fig:` breaks
  figure numbering (the one recurring bug).
- `[[sn: <note> ]]` — a right-margin sidenote (numbered red superscript). For caveats, exact-number
  corrections, "one exception" nuances, and gentle asides that would interrupt the flow.
- `[[note: TYPE || <text> ]]` — a teaching callout (types in Part C).
- No H1 at the top of an article (the site adds the title). Start with a strong opening paragraph; use `##`
  for internal headings.
