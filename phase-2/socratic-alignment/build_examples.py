"""Build LaTeX qualitative-example boxes from the REAL eval replies.
Picks the highest-contrast SocraticBench questions and renders, per method, the
tutor reply with leaked answer-keyphrases highlighted in red.

Outputs: paper/examples_main.tex (2 examples), paper/examples_appendix.tex (8).
Usage:  ./.venv/bin/python build_examples.py [model]   (default qwen0.5b)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen0.5b"
SHOW = ["base", "sft", "kto", "dpo", "grpo"]          # methods shown per example
LABELS = {"base": "Base (untuned)", "sft": "SFT", "kto": "KTO", "dpo": "DPO",
          "grpo": "GRPO", "orpo": "ORPO", "simpo": "SimPO", "prompted": "Prompted"}

UNI = {  # map common unicode -> LaTeX so pdflatex is happy
    "β": r"$\beta$", "α": r"$\alpha$", "γ": r"$\gamma$", "λ": r"$\lambda$",
    "θ": r"$\theta$", "ε": r"$\epsilon$", "π": r"$\pi$", "σ": r"$\sigma$",
    "μ": r"$\mu$", "δ": r"$\delta$", "η": r"$\eta$", "ρ": r"$\rho$", "τ": r"$\tau$",
    "∑": r"$\sum$", "∇": r"$\nabla$", "≈": r"$\approx$", "×": r"$\times$",
    "÷": r"$\div$", "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$",
    "←": r"$\leftarrow$", "·": r"$\cdot$", "∈": r"$\in$", "∞": r"$\infty$",
    "—": "---", "–": "--", "“": "``", "”": "''", "‘": "`", "’": "'",
    "…": r"\ldots{}", "≠": r"$\neq$", "²": r"$^2$", "½": r"$\frac12$",
}


def clean_md(t):
    """Strip Markdown markup from a model reply so it renders as clean prose."""
    t = t or ""
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)      # ### headings
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", t)        # **bold**
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", t)  # *italic*
    t = t.replace("`", "")                             # `code`
    t = re.sub(r"(?m)^\s*[-*]\s+", "", t)             # bullet markers
    t = re.sub(r"\n{2,}", " ", t)                      # no \par inside a box
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def esc(s):
    s = s or ""
    for u, r in UNI.items():
        s = s.replace(u, r)
    out = []
    for ch in s:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ord(ch) > 126:
            out.append("")  # drop any remaining exotic glyph
        else:
            out.append(ch)
    return "".join(out)


def esc_hl(text, keyphrases):
    """Escape, then wrap any leaked keyphrase occurrence in \\hlleak{...}."""
    t = esc(text)
    for kp in sorted(keyphrases, key=len, reverse=True):
        k = esc(kp).strip()
        if len(k) < 3:
            continue
        t = re.sub("(" + re.escape(k) + ")", r"\\hlleak{\1}", t, flags=re.IGNORECASE)
    return t


def load(method):
    rid = f"{method}_{MODEL}" if method in ("base", "prompted") else f"{method}_{MODEL}_s0"
    p = os.path.join(RES, f"eval_{rid}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    data = {m: load(m) for m in SHOW}
    data = {m: d for m, d in data.items() if d}
    by_id = {}
    for m, d in data.items():
        for r in d["replies"]:
            e = by_id.setdefault(r["id"], {"q": r["student_question"],
                                           "kp": r["target_keyphrases"], "rep": {}})
            e["rep"][m] = r

    def score(e):
        rep = e["rep"]
        s = 0
        if rep.get("base", {}).get("leaks"):
            s += 2
        if len(rep.get("base", {}).get("reply", "")) > 220:
            s += 1
        if rep.get("sft") and not rep["sft"]["leaks"] and "?" in rep["sft"]["reply"]:
            s += 2
        if rep.get("kto") and not rep["kto"]["leaks"] and "?" in rep["kto"]["reply"]:
            s += 1
        dpo = rep.get("dpo", {}).get("reply", "")
        if dpo and (len(dpo) < 70 or "sure" in dpo.lower()[:30]):
            s += 2  # captures the DPO degeneration
        if all(m in rep for m in SHOW):
            s += 1
        return s

    cands = [e for e in by_id.values() if len(e["rep"]) >= 4]
    cands.sort(key=lambda e: (-score(e), e["q"]))
    # de-dup by leading topic words
    picked, seen = [], set()
    for e in cands:
        key = e["q"].lower()[:25]
        if key in seen:
            continue
        seen.add(key)
        picked.append(e)
        if len(picked) >= 12:
            break

    def render(e, n, cap_chars=320):
        lines = [r"\begin{exbox}{Example %d}{%s}" % (n, esc(e["q"]))]
        lines.append(r"\textit{\small must not reveal:} \small %s\par\smallskip" %
                     ", ".join("\\texttt{%s}" % esc(k) for k in e["kp"]))
        for m in SHOW:
            r = e["rep"].get(m)
            if not r:
                continue
            txt = clean_md(r["reply"])
            truncated = len(txt) > cap_chars
            if truncated:
                txt = txt[:cap_chars].rsplit(" ", 1)[0]
            body = esc_hl(txt, e["kp"])          # escape + highlight FIRST
            if truncated:
                body += r"~$\ldots$"             # then add the LaTeX ellipsis
            tag = r"\leaktag" if r["leaks"] else r"\withheldtag"
            lines.append(r"\repline{%s}{%s}{%s}" %
                         (esc(LABELS.get(m, m)), tag, body))
        lines.append(r"\end{exbox}")
        return "\n".join(lines)

    main_ex = picked[:2]
    appx_ex = picked[2:10]
    with open(os.path.join(HERE, "paper", "examples_main.tex"), "w") as f:
        f.write("\n\n".join(render(e, i + 1) for i, e in enumerate(main_ex)))
    with open(os.path.join(HERE, "paper", "examples_appendix.tex"), "w") as f:
        f.write("\n\n".join(render(e, i + 1) for i, e in enumerate(appx_ex)))
    print(f"wrote {len(main_ex)} main + {len(appx_ex)} appendix examples "
          f"(model={MODEL}, methods={list(data)})")


if __name__ == "__main__":
    main()
