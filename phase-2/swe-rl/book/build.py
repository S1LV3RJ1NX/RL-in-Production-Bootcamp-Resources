#!/usr/bin/env python3
"""Compact two-surface reader for a single chapter/mini-book.

Descendant of the book-forge `build.py`, trimmed to one book: dark terminal SHELL
(home + chapter index + sidebar tree) and warm-white "Tufte" article pages. Keeps the
custom markdown verbatim:
  [[fig: <excalidraw prompt> || <caption>]]  -> figures/<slug>-<n>.png
  [[sn: <note text>]]                         -> right-margin sidenote (red superscript)
  [[note: TYPE || content]]                   -> teaching callout (metaphor/example/say/aha/...)
Run:  python3 build.py
"""
import json, os, re, html, shutil, pathlib

ROOT = pathlib.Path(__file__).parent
DOCS = ROOT / "docs"
ART = ROOT / "articles"
MAN = json.loads((ROOT / "manifest.json").read_text())
SITE = MAN["site"]

FLAT = []
for sec in MAN["sections"]:
    for a in sec["articles"]:
        FLAT.append({**a, "section_id": sec["id"], "section_num": sec["num"], "section_title": sec["title"]})
SLUG2IDX = {a["slug"]: i for i, a in enumerate(FLAT)}

# ============================================================ markdown
def esc(s): return html.escape(s, quote=False)

def inline(text, ctx):
    stash = []
    def sn(m):
        ctx["sn"] += 1; n = ctx["sn"]
        note = inline_basic(m.group(1).strip())
        stash.append(
            f'<label for="sn-{ctx["slug"]}-{n}" class="sn-ref">{n}</label>'
            f'<input type="checkbox" id="sn-{ctx["slug"]}-{n}" class="sn-toggle">'
            f'<span class="sidenote"><sup>{n}</sup> {note}</span>')
        return f"\x01{len(stash)-1}\x01"
    text = re.sub(r"\[\[sn:\s*(.+?)\]\]", sn, text, flags=re.S)
    text = inline_basic(text)
    text = re.sub(r"\x01(\d+)\x01", lambda m: stash[int(m.group(1))], text)
    return text

def inline_basic(text):
    codes = []
    def stash(m):
        codes.append(m.group(1)); return f"\x00{len(codes)-1}\x00"
    text = re.sub(r"`([^`]+)`", stash, text)
    text = esc(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: f'<a href="{esc(m.group(2))}" target="_blank" rel="noopener">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{esc(codes[int(m.group(1))])}</code>", text)
    return text

def md_to_html(md, slug):
    ctx = {"slug": slug, "sn": 0, "fig": 0}
    lines = md.replace("\r\n", "\n").split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("```"):
            lang = line.strip()[3:].strip(); i += 1; buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(f'<pre class="code" data-lang="{esc(lang)}"><code>{esc(chr(10).join(buf))}</code></pre>')
            continue
        if line.strip().startswith("[[note:"):
            buf = [line]
            while "]]" not in buf[-1] and i + 1 < n:
                i += 1; buf.append(lines[i])
            i += 1
            raw = " ".join(buf).strip()
            mm = re.match(r"\[\[note:\s*(\w+)\s*\|\|\s*(.+?)\]\]\s*$", raw, flags=re.S)
            if mm:
                typ = mm.group(1).lower(); content = inline(mm.group(2).strip(), ctx)
            else:
                typ = "teach"; content = inline(raw[7:].strip().rstrip("]").strip(), ctx)
            cmeta = {"metaphor": ("🧠", "A real-life analogy"), "example": ("🔢", "By hand"),
                     "production": ("🤖", "In the real world"), "teach": ("🎓", "Teaching note"),
                     "say": ("🎤", "Worth saying plainly"), "demo": ("▶️", "Try it"),
                     "confusion": ("⚠️", "A common mix-up"), "aha": ("✨", "The click")}
            icon, label = cmeta.get(typ, ("🎓", "Note"))
            out.append(f'<div class="cal cal-{typ}"><div class="cal-h"><span class="cal-i">{icon}</span> {label}</div>'
                       f'<div class="cal-b">{content}</div></div>')
            continue
        if line.strip().startswith("[[fig:"):
            buf = [line]
            while "]]" not in buf[-1] and i + 1 < n:
                i += 1; buf.append(lines[i])
            i += 1
            raw = " ".join(buf).strip()
            m = re.match(r"\[\[fig:\s*(.+?)\]\]\s*$", raw, flags=re.S)
            body = m.group(1) if m else raw[6:]
            if "||" in body:
                prompt, cap = body.split("||", 1)
            else:
                prompt, cap = body, ""
            ctx["fig"] += 1
            fname = f"{slug}-{ctx['fig']}.png"
            caph = inline_basic(cap.strip())
            out.append(
                f'<figure class="fig"><div class="fig-frame">'
                f'<img src="../figures/{fname}" alt="{esc(cap.strip()[:120])}" decoding="async" '
                f'onerror="if(!this.dataset.retry){{this.dataset.retry=1;var s=this.src.split(&#39;?&#39;)[0];setTimeout(function(t){{return function(){{t.src=s+&#39;?r=&#39;+Date.now();}};}}(this),400);}}else{{this.parentNode.classList.add(&#39;fig-missing&#39;);this.style.display=&#39;none&#39;;}}" '
                f'data-fig="{fname}">'
                f'<span class="fig-ph">figure rendering &middot; {esc(cap.strip()[:70])}</span>'
                f'</div>' + (f'<figcaption>{caph}</figcaption>' if cap.strip() else '') + '</figure>')
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1)); txt = inline(m.group(2).strip(), ctx)
            hid = re.sub(r"[^a-z0-9]+", "-", m.group(2).strip().lower()).strip("-")
            out.append(f'<h{lvl} id="{hid}">{txt}</h{lvl}>')
            i += 1; continue
        if line.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            out.append(f'<blockquote>{inline(" ".join(buf), ctx)}</blockquote>')
            continue
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$", lines[i+1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2; rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f"<th>{inline(c, ctx)}</th>" for c in header)
            trs = "".join("<tr>" + "".join(f"<td>{inline(c, ctx)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            continue
        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*[-*]\s+", "", lines[i]), ctx)); i += 1
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in buf) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(inline(re.sub(r"^\s*\d+\.\s+", "", lines[i]), ctx)); i += 1
            out.append("<ol>" + "".join(f"<li>{x}</li>" for x in buf) + "</ol>")
            continue
        if not line.strip():
            i += 1; continue
        buf = [line]; i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|>|\s*[-*]\s|\s*\d+\.\s|```|\[\[fig:|\[\[note:)", lines[i]):
            buf.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(buf), ctx)}</p>")
    return "\n".join(out)

# ============================================================ shell
def sidebar(active_slug, rel):
    rows = [f'<a class="sb-home" href="{rel}index.html">‹ {esc(SITE["title"].lower())}</a>']
    for sec in MAN["sections"]:
        open_sec = any(a["slug"] == active_slug for a in sec["articles"])
        rows.append(f'<details class="sb-sec"{" open" if open_sec else ""}>')
        rows.append(f'<summary><span class="sb-num">{sec["num"]}</span> {esc(sec["title"])}</summary>')
        for a in sec["articles"]:
            act = " active" if a["slug"] == active_slug else ""
            chip = f'<span class="chip">{esc(a["chip"])}</span>' if a["chip"] else ""
            rows.append(f'<a class="sb-item{act}" href="{rel}a/{a["slug"]}.html">{esc(a["title"])}{chip}</a>')
        rows.append("</details>")
    return '<nav class="sidebar" id="sidebar">' + "".join(rows) + "</nav>"

def shell(title, main_html, active_slug=None, rel="", canvas="dark", with_sidebar=False, active_nav=""):
    def nl(href, label, key):
        return f'<a class="tn{" active" if key==active_nav else ""}" href="{rel}{href}">{label}</a>'
    topnav = nl("index.html", "The Book", "home") + nl("quiz.html", "Self-check", "quiz")
    sb = sidebar(active_slug, rel) if with_sidebar else ""
    layout_cls = "layout" if with_sidebar else "layout nosb"
    return f"""<!doctype html><html lang="en" data-theme="terminal"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(SITE['tagline'])}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{rel}assets/app.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head><body class="canvas-{canvas}{' has-sb' if with_sidebar else ''}">
<div class="topbar">
  <a class="brand" href="{rel}index.html"><span class="brand-txt">Vizuara <b>&nbsp;Teaching&nbsp;Machines&nbsp;to&nbsp;Code</b></span></a>
  <nav class="topnav">{topnav}</nav>
  {'<button class="menu-btn" onclick="document.body.classList.toggle(&#39;sb-open&#39;)">☰</button>' if with_sidebar else ''}
  <div class="top-links">
    <button class="tbtn icon" id="search-open" title="Search (⌘K)">⌘K</button>
    <button class="tbtn" id="theme-btn">Terminal</button>
  </div>
</div>
<div class="{layout_cls}">
{sb}
<main class="content">{main_html}</main>
</div>
<div class="search-modal" id="search-modal"><div class="search-box">
  <input id="search-input" placeholder="Search the chapter…" autocomplete="off">
  <div id="search-results"></div>
  <div class="search-hint">↑↓ to navigate · ↵ to open · esc to close</div>
</div></div>
<script>window.SEARCH_BASE="{rel}";</script>
<script src="{rel}assets/app.js"></script>
</body></html>"""

# ============================================================ pages
def build_article(a, idx):
    slug = a["slug"]
    md_path = ART / f"{slug}.md"
    if md_path.exists():
        body = md_to_html(md_path.read_text(), slug); stub = ""
    else:
        body = (f'<p class="lead">{esc(a["blurb"])}</p>'
                f'<div class="stub">This section is being written.</div>')
        stub = " stub"
    prev_a = FLAT[idx-1] if idx > 0 else None
    next_a = FLAT[idx+1] if idx < len(FLAT)-1 else None
    nav = '<div class="prevnext">'
    nav += (f'<a class="pn prev" href="{prev_a["slug"]}.html"><span>‹ previous</span>{esc(prev_a["title"])}</a>'
            if prev_a else '<span></span>')
    nav += (f'<a class="pn next" href="{next_a["slug"]}.html"><span>next ›</span>{esc(next_a["title"])}</a>'
            if next_a else '<span></span>')
    nav += "</div>"
    chip = f'<span class="chip lg">{esc(a["chip"])}</span>' if a["chip"] else ""
    art = f"""<article class="worklog{stub}">
<div class="art-kicker"><span class="art-sec">{a['section_num']} · {esc(a['section_title'])}</span></div>
<h1 class="art-title">{esc(a['title'])} {chip}</h1>
<div class="art-body">{body}</div>
{nav}
</article>"""
    html_out = shell(f"{a['title']} · {SITE['title']}", art, active_slug=slug, rel="../",
                     canvas="paper", with_sidebar=True, active_nav="home")
    (DOCS / "a" / f"{slug}.html").write_text(html_out)

def build_index():
    parts = []
    for sec in MAN["sections"]:
        arts = "".join(
            f'<a href="a/{a["slug"]}.html" class="ch-art">{esc(a["title"])}'
            + (f'<span class="chip">{esc(a["chip"])}</span>' if a["chip"] else "") + '</a>'
            for a in sec["articles"])
        parts.append(
            f'<div class="chapter"><div class="ch-side"><div class="ch-num">{sec["num"]}</div>'
            f'<div class="ch-title">{esc(sec["title"])}</div>'
            f'<div class="ch-count">{len(sec["articles"])} sections</div></div>'
            f'<div class="ch-body"><p class="ch-blurb">{esc(sec["blurb"])}</p>'
            f'<div class="ch-arts">{arts}</div></div></div>')
    first = FLAT[0]["slug"]
    main = f"""<div class="home">
<section class="hero2">
  <div class="eyebrow">{esc(SITE['org'])} · {esc(SITE['byline'])}</div>
  <h1>Teaching Machines to <span class="hl">Code</span></h1>
  <p class="sub">{esc(SITE['tagline'])} A figure-first book that starts from the very beginning — what a software task is, what a terminal is, what reinforcement learning does — and then builds three real projects that teach a language model to fix bugs, solve real programming tasks, and even learn a model of the computer it works on.</p>
  <div class="hero-cta">
    <a class="btn solid" href="a/{first}.html">Start reading →</a>
    <a class="btn" href="quiz.html">Self-check quiz</a>
  </div>
  <div class="hero-arc">the arc &nbsp;·&nbsp; what is a SWE task &nbsp;→&nbsp; the terminal &nbsp;→&nbsp; RL &amp; GRPO &nbsp;→&nbsp; RL on your laptop &nbsp;→&nbsp; agentic RL on real code &nbsp;→&nbsp; a world model for free &nbsp;→&nbsp; the big picture</div>
</section>
<section class="book-page" style="max-width:980px;margin:0 auto">
<div class="chapters">{''.join(parts)}</div>
</section>
<footer class="home-credit">© {esc(SITE['org'])} · a lecture companion · read it inside-out: terminal shell, notebook pages.</footer>
</div>"""
    (DOCS / "index.html").write_text(shell(f"{SITE['title']} · {SITE['tagline']}", main, rel="", canvas="dark", active_nav="home"))

QUIZ = [
    ("In this book, what makes something a 'software engineering task' we can train on?",
     ["Any request phrased as a question to a chatbot", "A concrete code change with an automatic test that is currently failing and must pass", "A design document with no code"], 1,
     "The whole book rests on tasks with a definition of done you can check automatically: some code, a failing test, and 'done' = make the test pass without breaking the others. That checkable goal is what lets a machine grade its own attempts."),
    ("Why does giving a coding AI a terminal matter so much?",
     ["Terminals make the AI run faster", "A terminal lets the AI actually DO things — run tests, change files, see errors — instead of only talking about code", "It is only a cosmetic, retro look"], 1,
     "The terminal is the agent's hands. Without it, a model can describe a fix; with it, the model can run the tests, read the real error, edit the file, and check that it worked. Doing beats describing."),
    ("Across all three projects, what is the reward the model learns from?",
     ["A human rating each answer from 1 to 10", "Whether the tests pass — 1.0 if they all pass, 0.0 otherwise", "The number of lines of code written"], 1,
     "The recurring big idea: the tests are the teacher. Nobody hand-labels the correct code. A real environment runs the model's attempt and hands back a 1 or a 0, and reinforcement learning does the rest."),
    ("What is the core recipe of GRPO (Group Relative Policy Optimization)?",
     ["Train a separate 'critic' network to score every state", "Let the model try one problem several times, then reinforce the tries that beat the group's average", "Copy the answers from a bigger model"], 1,
     "GRPO gives the model, say, 8 attempts at one problem, scores each, and computes each attempt's advantage = its reward minus the group's average. Above-average tries are made more likely, below-average ones less. The group's own average is the baseline — so no separate critic network is needed."),
    ("Why must the training puzzles be 'just hard enough' — the sweet spot?",
     ["Hard puzzles look more impressive", "If every attempt in a group gets the same score (all pass or all fail), every advantage is zero and there is nothing to learn", "The model runs out of memory on easy puzzles"], 1,
     "GRPO needs variance: some tries that succeed and some that fail on the SAME problem. All-pass and all-fail groups both give a flat, zero learning signal. In Mini-SWE-RL, only the 18 'mixed' puzzles (25–75% solve rate) actually taught the model anything."),
    ("What did Mini-SWE-RL achieve, and on what hardware?",
     ["99% on real GitHub bugs, on 64 GPUs", "66.7% → 73.3% solve rate (7 new bugs solved) in ~30 minutes on a MacBook", "No improvement at all"], 1,
     "Using Qwen2.5-Coder-1.5B on an Apple M4 Pro laptop for ~30 minutes, the solve rate rose from 66.7% (20/30) to 73.3% (22/30), with 7 puzzles solved that had never been solved before — the same GRPO algorithm as the big systems, just miniaturized."),
    ("In project two, each task has a few VISIBLE tests and many HIDDEN tests. Why keep hidden tests?",
     ["To make training slower", "To honestly check whether the model really solved the problem, versus just satisfying the few visible checks", "Because visible tests are always wrong"], 1,
     "The reward can only see the visible tests, so the model optimizes those. The hidden tests — which the model never trains on — are the honest ground truth: they tell you whether an 'after' is a genuine fix or just something shaped to the visible checks."),
    ("Why was the 0.5-billion-parameter model chosen to show the before/after examples, not the 7B?",
     ["The 0.5B model is smarter", "The 7B model already solved ~84% before any training, so it had almost no room to improve; the smaller model has more headroom for a clear story", "The 7B model would not fit on any GPU"], 1,
     "A model near the ceiling can't show much learning. The 0.5B model started far lower (44% solved), so RL had room to help — and 14 tasks flipped from failing to passing, giving honest, visible before/after examples."),
    ("What is a 'world model', in plain terms?",
     ["A 3-D map of the Earth", "A model's ability to predict what will happen next in its environment — like expecting the light before you flip the switch", "A very large language model"], 1,
     "A world model is an internal predictor of what comes next. For a terminal agent, it means predicting what the terminal will print after a command. That anticipation is exactly what ECHO teaches the agent to build."),
    ("What is ECHO's one extra idea on top of ordinary GRPO?",
     ["Use ten times more GPUs", "Also train the agent to predict the terminal's response (the environment's tokens), reusing the same forward pass — a world model 'for free'", "Replace tests with human labels"], 1,
     "ECHO adds a small second loss term, L_ECHO = L_GRPO + λ·L_env with λ=0.05, that trains the model to predict the terminal's output. Because it reuses the same forward pass, it costs almost nothing — and learning to predict the computer's replies teaches the agent how the computer behaves."),
    ("What is the honest way to read ECHO's benchmark numbers?",
     ["The exact scores (like 10.79) are guaranteed and easy to reproduce", "The controlled, head-to-head result — ECHO roughly doubling GRPO with only the extra loss added — is the real claim; the exact absolute numbers are only a few of 89 tasks and depend on data we can't fully reproduce", "ECHO always solves every task"], 1,
     "The scientific claim is the relative A/B: same everything, add the one extra loss, and the score roughly doubles. The absolute pass@1 numbers on TerminalBench-2.0's 89 tasks are small and depend on private training data, so a replication reports 'near the ballpark, with a gap' — not an exact match."),
    ("What single idea ties all three projects together?",
     ["Bigger models are always better", "The environment is the teacher — tests or a terminal grade the model's own attempts, and RL turns that signal into skill; no hand-labeled answers", "You must always train in the cloud"], 1,
     "Laptop puzzles, cloud coding tasks, terminal world models — underneath, the same move. A real environment scores the model's attempts, and GRPO nudges it toward the better ones. That is why 'the tests are the teacher' is the through-line of the whole book."),
]

def build_quiz():
    qs = ""
    for i, (q, opts, correct, exp) in enumerate(QUIZ):
        obtns = "".join(f'<button class="q-opt" data-i="{j}">{esc(o)}</button>' for j, o in enumerate(opts))
        qs += (f'<div class="quiz-q" data-correct="{correct}"><div class="q-num">Q{i+1}</div>'
               f'<div class="q-text">{esc(q)}</div><div class="q-opts">{obtns}</div>'
               f'<div class="q-exp">{esc(exp)}</div></div>')
    main = f"""<div class="section-page interactive-page">
<div class="crumb">/ self-check</div>
<h1 class="sec-h1">Self-check</h1>
<p class="sec-blurb">Twelve questions across the whole arc — what a software task and a terminal are, the RL loop and GRPO, the three projects (bug-fixing on a laptop, agentic RL on real code, and a world model for free), and the one idea underneath them all. Pick an answer to see the explanation. A good warm-up, and a good check that the ideas landed.</p>
<div class="int-panel">
  <div class="int-head"><h2 class="ws-h2">Quiz yourself</h2><span class="quiz-score" id="quiz-score">0 / {len(QUIZ)}</span></div>
  <div class="quiz" data-total="{len(QUIZ)}">{qs}</div>
</div>
</div>"""
    (DOCS / "quiz.html").write_text(shell(f"Self-check · {SITE['title']}", main, rel="", canvas="dark", active_nav="quiz"))

def build_search_index():
    idx = []
    for a in FLAT:
        idx.append({"t": a["title"], "s": a["slug"], "sec": a["section_title"],
                    "chip": a["chip"], "b": a["blurb"], "u": f"a/{a['slug']}.html"})
    (DOCS / "search.json").write_text(json.dumps(idx, ensure_ascii=False))

def main():
    (DOCS / "a").mkdir(parents=True, exist_ok=True)
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "assets").glob("*"):
        shutil.copy(src, DOCS / "assets" / src.name)
    (DOCS / ".nojekyll").write_text("")
    for i, a in enumerate(FLAT):
        build_article(a, i)
    build_index(); build_quiz(); build_search_index()
    have = sum(1 for a in FLAT if (ART / f"{a['slug']}.md").exists())
    print(f"built {len(FLAT)+2} pages · {len(FLAT)} sections ({have} written)")

if __name__ == "__main__":
    main()
