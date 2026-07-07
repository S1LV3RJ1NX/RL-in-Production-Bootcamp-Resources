"""Build a single-file workshop demo (demo/index.html) from the real eval
results: flip through SocraticBench questions and see, side by side, how each
recipe replies to the same student — base answer-dump vs aligned Socratic vs the
DPO/GRPO degeneration. Keyphrase leaks are highlighted in red.

Usage:  ./.venv/bin/python build_demo.py [model]   (default qwen0.5b)
"""
import glob
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen0.5b"
METHODS = ["base", "prompted", "sft", "dpo", "kto", "orpo", "simpo", "grpo"]


def load(method):
    rid = f"{method}_{MODEL}" if method in ("base", "prompted") else f"{method}_{MODEL}_s0"
    p = os.path.join(RES, f"eval_{rid}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def clean_md(t):
    t = t or ""
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", t)
    t = t.replace("`", "")
    t = re.sub(r"(?m)^\s*[-*]\s+", "", t)
    t = re.sub(r"\n{2,}", "\n", t).strip()
    return t


def main():
    data = {m: load(m) for m in METHODS}
    data = {m: d for m, d in data.items() if d}
    methods = list(data.keys())

    # index replies by question id
    by_id = {}
    for m, d in data.items():
        for r in d["replies"]:
            e = by_id.setdefault(r["id"], {"q": r["student_question"],
                                           "kp": r["target_keyphrases"], "rep": {}})
            e["rep"][m] = {"text": clean_md(r["reply"]), "leaks": r["leaks"]}
    questions = [v for v in by_id.values() if len(v["rep"]) >= 2]

    stats = {m: {"leakage": d["verifiable"]["leakage_rate"],
                 "judge": d["judge"]["judge_overall"],
                 "asksq": d["verifiable"]["asks_question_rate"]}
             for m, d in data.items()}

    payload = json.dumps({"model": MODEL, "methods": methods,
                          "questions": questions, "stats": stats})

    html_doc = TEMPLATE.replace("/*DATA*/", payload)
    outdir = os.path.join(HERE, "demo")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "index.html")
    open(out, "w").write(html_doc)
    print(f"wrote {out}  ({len(questions)} questions, methods={methods})")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Socratic Alignment — before vs after</title>
<style>
 :root{--bg:#f6f1e7;--card:#fffdf8;--ink:#2b2722;--muted:#7c7464;--line:#e6dcc8;
   --good:#1f7a4d;--bad:#b23b3b;--accent:#8a5a2b}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
 header{padding:22px 26px;border-bottom:1px solid var(--line);background:var(--card)}
 h1{margin:0 0 4px;font-size:21px}
 .sub{color:var(--muted);font-size:14px}
 .wrap{max-width:1100px;margin:0 auto;padding:20px 26px}
 .controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
 select,button{font:inherit;padding:8px 12px;border:1px solid var(--line);
   border-radius:9px;background:var(--card);color:var(--ink);cursor:pointer}
 button:hover{border-color:var(--accent)}
 .q{background:var(--card);border:1px solid var(--line);border-radius:12px;
   padding:16px 18px;margin-bottom:16px}
 .q .label{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
 .q .text{font-size:18px;margin-top:4px}
 .kp{margin-top:8px;font-size:13px;color:var(--muted)}
 .kp b{color:var(--accent)}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
   padding:14px 16px;display:flex;flex-direction:column}
 .m{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
 .m .name{font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:13px}
 .badge{font-size:12px;padding:2px 8px;border-radius:20px;border:1px solid var(--line)}
 .leak{color:var(--bad);border-color:var(--bad)}
 .ok{color:var(--good);border-color:var(--good)}
 .reply{white-space:pre-wrap}
 .reply mark{background:#f7d9d4;color:var(--bad);padding:0 2px;border-radius:3px}
 .stat{font-size:12px;color:var(--muted);margin-top:8px;border-top:1px dashed var(--line);
   padding-top:6px}
 .tag{font-size:11px;padding:1px 7px;border-radius:6px;margin-left:6px}
 .t-good{background:#dcefe2;color:var(--good)} .t-bad{background:#f4dcdc;color:var(--bad)}
</style></head><body>
<header><div class="wrap" style="padding:0">
  <h1>Socratic Alignment of Small Language Models — <span id="mdl"></span></h1>
  <div class="sub">Same student question, every recipe. <b>Red highlight</b> = the
   tutor leaked an answer keyphrase. Watch SFT/KTO/ORPO/SimPO <i>guide</i> while
   DPO/GRPO <i>evade or dump</i>. Real outputs from the sweep.</div>
</div></header>
<div class="wrap">
  <div class="controls">
    <button id="prev">&larr; Prev</button>
    <select id="qsel"></select>
    <button id="next">Next &rarr;</button>
    <button id="rand">Random</button>
  </div>
  <div class="q"><div class="label">Student asks</div>
    <div class="text" id="qtext"></div>
    <div class="kp" id="qkp"></div></div>
  <div class="grid" id="grid"></div>
</div>
<script>
const D = /*DATA*/;
document.getElementById('mdl').textContent = D.model;
const sel=document.getElementById('qsel');
D.questions.forEach((q,i)=>{const o=document.createElement('option');o.value=i;
  o.textContent=(i+1)+'. '+q.q.slice(0,70);sel.appendChild(o);});
let cur=0;
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function hl(text,kps){let t=esc(text);
  kps.forEach(k=>{if(k.length<3)return;
    const re=new RegExp('('+k.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','ig');
    t=t.replace(re,'<mark>$1</mark>');});
  return t;}
function render(){
  const q=D.questions[cur];sel.value=cur;
  document.getElementById('qtext').textContent=q.q;
  document.getElementById('qkp').innerHTML='must NOT reveal: <b>'+
    q.kp.map(esc).join('</b>, <b>')+'</b>';
  const order=D.methods;
  document.getElementById('grid').innerHTML=order.filter(m=>q.rep[m]).map(m=>{
    const r=q.rep[m];const st=D.stats[m]||{};
    const leak=r.leaks;
    return `<div class="card"><div class="m"><span class="name">${m}</span>
      <span class="badge ${leak?'leak':'ok'}">${leak?'leaked':'withheld'}</span></div>
      <div class="reply">${hl(r.text,q.kp)}</div>
      <div class="stat">method avg — leak ${(st.leakage??0).toFixed(2)} ·
        judge ${(st.judge??0).toFixed(0)}/100 · asks-Q ${(st.asksq??0).toFixed(2)}
        ${st.judge>=70?'<span class="tag t-good">Socratic</span>':
          (st.judge<D.stats.base.judge?'<span class="tag t-bad">worse than base</span>':'')}
      </div></div>`;}).join('');
}
document.getElementById('prev').onclick=()=>{cur=(cur-1+D.questions.length)%D.questions.length;render();};
document.getElementById('next').onclick=()=>{cur=(cur+1)%D.questions.length;render();};
document.getElementById('rand').onclick=()=>{cur=Math.floor(Math.random()*D.questions.length);render();};
sel.onchange=()=>{cur=+sel.value;render();};
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
