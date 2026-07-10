"""Build demo/index.html — a self-contained page telling the IRIS story with the
REAL generated frames (reconstructions, dreams, world-model-vs-real) and the
headline result (policy vs baselines).  Everything is inlined as base64 so the
page is a single portable file.

Run with the plotting venv:  ./.plotvenv/bin/python build_demo.py
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
DEMO = os.path.join(HERE, "demo")
os.makedirs(DEMO, exist_ok=True)


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def gif_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(b64, cls="frame"):
    return f'<img class="{cls}" src="data:image/png;base64,{b64}">'


ev = load("eval_catch.json") or {}
dr = load("dreams_catch.json") or {}
summ = load("run_all_catch.json") or {}

pol = ev.get("policy", {})
rnd = ev.get("random_baseline", {})
orc = ev.get("oracle_baseline", {})

# --- headline numbers (REAL) ---
pol_ret = pol.get("return_mean")
rnd_ret = rnd.get("return_mean")
orc_ret = orc.get("return_mean")
pol_cr = pol.get("catch_rate")
rnd_cr = rnd.get("catch_rate")
fidelity = ev.get("wm_fidelity_mse")


def fmt(v, spec="+.2f"):
    return f"{v:{spec}}" if isinstance(v, (int, float)) else "—"


# --- reconstructions strip ---
recon_html = ""
for r in (dr.get("reconstructions") or [])[:8]:
    recon_html += (f'<div class="pair"><div class="lbl">real</div>{img(r["real"])}'
                   f'<div class="lbl">decoded</div>{img(r["recon"])}</div>')

# --- dream GIFs / filmstrips ---
dream_html = ""
for i, d in enumerate((dr.get("dreams") or [])[:4]):
    g = gif_b64(os.path.join(FIG, "dreams", f"dream_catch_{i}.gif"))
    strip = "".join(img(f) for f in d["frames"])
    acts = d.get("actions", [])
    rews = d.get("rewards", [])
    caption = f"actions: {['L' if a==0 else '·' if a==1 else 'R' for a in acts]}"
    tot = sum(rews)
    caption += f" &nbsp; dreamed return: <b>{tot:+.0f}</b>"
    gif_tag = (f'<img class="gif" src="data:image/gif;base64,{g}">' if g else "")
    dream_html += (f'<div class="dream"><div class="dhead">Dream {i} {gif_tag}</div>'
                   f'<div class="strip">{strip}</div><div class="cap">{caption}</div></div>')

# --- wm vs real ---
wmreal_html = ""
for cmp in (dr.get("wm_vs_real") or [])[:2]:
    real = "".join(img(f) for f in cmp["real"])
    dream = "".join(img(f) for f in cmp["dream"])
    mse = cmp.get("mse_per_step", [])
    mm = (sum(mse) / len(mse)) if mse else None
    wmreal_html += (f'<div class="cmp"><div class="row"><span class="rl">real</span>{real}</div>'
                    f'<div class="row"><span class="rl">world&nbsp;model</span>{dream}</div>'
                    f'<div class="cap">mean per-step MSE: <b>{mm:.4f}</b></div></div>' if mm is not None else "")

# --- verdict text ---
beats = (pol_ret is not None and rnd_ret is not None and pol_ret > rnd_ret + 0.05)
verdict_cls = "good" if beats else "warn"
if beats:
    verdict = (f"Yes — the policy trained <b>entirely inside the world model's dreams</b> "
               f"reaches a real-environment return of <b>{fmt(pol_ret)}</b> "
               f"(catch rate {pol_cr:.0%}), beating the random baseline "
               f"({fmt(rnd_ret)}, {rnd_cr:.0%}) and approaching the optimal oracle "
               f"({fmt(orc_ret)}).")
else:
    verdict = (f"The imagination-trained policy reaches return <b>{fmt(pol_ret)}</b> "
               f"vs random <b>{fmt(rnd_ret)}</b> and oracle <b>{fmt(orc_ret)}</b> — "
               f"see the honest discussion in the report.")

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRIS from scratch — learning to Catch in a dream</title>
<style>
 :root{{--bg:#F7F2E8;--card:#fffdf8;--ink:#2b2722;--muted:#7c7464;--line:#e6dcc8;
   --good:#1f7a4d;--warn:#b23b3b;--accent:#8a5a2b;--blue:#2f6fb0}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
   font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
 header{{padding:26px 30px;border-bottom:1px solid var(--line);background:var(--card)}}
 h1{{margin:0 0 6px;font-size:26px}}
 .sub{{color:var(--muted);font-size:15px;max-width:760px}}
 .wrap{{max-width:1080px;margin:0 auto;padding:24px 30px}}
 section{{background:var(--card);border:1px solid var(--line);border-radius:14px;
   padding:20px 24px;margin-bottom:22px}}
 h2{{margin:0 0 6px;font-size:19px;color:var(--accent)}}
 .note{{color:var(--muted);font-size:14px;margin-bottom:14px}}
 .hero{{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0 4px}}
 .stat{{flex:1;min-width:150px;background:var(--bg);border:1px solid var(--line);
   border-radius:12px;padding:14px 16px;text-align:center}}
 .stat .n{{font-size:30px;font-weight:800}}
 .stat .k{{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
 .n.good{{color:var(--good)}} .n.blue{{color:var(--blue)}} .n.mut{{color:var(--muted)}}
 .verdict{{font-size:17px;padding:14px 16px;border-radius:12px;margin-top:6px}}
 .verdict.good{{background:#eaf5ee;border:1px solid var(--good)}}
 .verdict.warn{{background:#f8ecec;border:1px solid var(--warn)}}
 img.frame{{width:52px;height:52px;image-rendering:pixelated;border-radius:5px;
   border:1px solid var(--line);margin:1px}}
 img.gif{{height:64px;image-rendering:pixelated;border-radius:6px;border:1px solid var(--line);
   vertical-align:middle;margin-left:10px}}
 .pair{{display:inline-flex;flex-direction:column;align-items:center;margin:6px 10px}}
 .pair .lbl,.lbl{{font-size:11px;color:var(--muted);margin:2px 0}}
 .dream{{margin:12px 0;padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:10px}}
 .dhead{{font-weight:700;margin-bottom:6px}}
 .strip img{{margin:1px}}
 .cap{{font-size:13px;color:var(--muted);margin-top:6px}}
 .cmp{{margin:12px 0;padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:10px}}
 .cmp .row{{display:flex;align-items:center;gap:8px;margin:2px 0}}
 .rl{{width:92px;font-size:12px;color:var(--muted);text-align:right}}
 .story{{max-width:820px}}
 code{{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:13px}}
 footer{{color:var(--muted);font-size:13px;padding:20px 30px;text-align:center}}
</style></head><body>
<header>
  <h1>IRIS from scratch — learning to <em>Catch</em> in a dream</h1>
  <div class="sub">A small, faithful re-implementation of IRIS (Micheli et&nbsp;al., 2023):
  a discrete-token <b>world model</b> learned from pixels, and a policy trained
  <b>entirely inside that world model's imagination</b> — never touching the real
  game during learning. Every number and every frame below is from a real Modal
  A10G run. Vizuara RL-in-Production bootcamp.</div>
</header>
<div class="wrap">

  <section>
    <h2>The headline</h2>
    <div class="hero">
      <div class="stat"><div class="n good">{fmt(pol_ret)}</div><div class="k">IRIS policy return</div></div>
      <div class="stat"><div class="n mut">{fmt(rnd_ret)}</div><div class="k">random baseline</div></div>
      <div class="stat"><div class="n blue">{fmt(orc_ret)}</div><div class="k">oracle (optimal)</div></div>
      <div class="stat"><div class="n good">{pol_cr:.0%}</div><div class="k">catch rate (policy)</div></div>
    </div>
    <div class="verdict {verdict_cls}">{verdict}</div>
  </section>

  <section class="story">
    <h2>What is happening here</h2>
    <div class="note">The three IRIS pieces, all trained from scratch on 64&times;64 RGB frames.</div>
    <p><b>1. A tokenizer</b> (a VQ-VAE) compresses every frame into just 16 discrete
    tokens from a learned codebook, and decodes them back. <b>2. A GPT-style
    Transformer world model</b> reads the token+action history and autoregressively
    predicts the next frame's tokens, the reward, and whether the episode ended —
    it <em>is</em> a learned simulator of the game. <b>3. An actor-critic policy</b>
    is trained purely on rollouts <em>hallucinated</em> by that Transformer
    (lambda-returns, REINFORCE with a value baseline), then deployed once in the
    real game to measure it.</p>
  </section>

  <section>
    <h2>1 &middot; Tokenizer reconstructions</h2>
    <div class="note">Top row: a real frame. Bottom row: the same frame after being
    crushed to 16 tokens and decoded. If these match, the world model has a faithful
    vocabulary to speak in.</div>
    {recon_html or "<i>reconstruction frames unavailable</i>"}
  </section>

  <section>
    <h2>2 &middot; Dreams — imagined policy rollouts</h2>
    <div class="note">The first frame is real; <b>every frame after it is generated by
    the world model</b> as the policy acts. Does the dreamed ball fall straight and
    the dreamed paddle follow the action?</div>
    {dream_html or "<i>dream frames unavailable</i>"}
  </section>

  <section>
    <h2>3 &middot; World model vs. the real game</h2>
    <div class="note">Same starting frame, same (oracle) action sequence, rolled in
    the real environment (top) and imagined by the world model (bottom). Lower MSE =
    a more pixel-faithful dream.</div>
    {wmreal_html or "<i>comparison frames unavailable</i>"}
    <div class="cap" style="margin-top:10px">Aggregate world-model fidelity (mean next-frame MSE over 10 seeds):
    <b>{fmt(fidelity, '.4f')}</b></div>
  </section>

  <footer>IRIS-from-scratch &middot; Vizuara RL-in-Production bootcamp &middot;
  all results from a real Modal A10G run &middot; env: Catch (8&times;8, 64&times;64 px)</footer>
</div></body></html>"""

with open(os.path.join(DEMO, "index.html"), "w") as f:
    f.write(HTML)
print(f"wrote demo/index.html ({len(HTML)//1024} KB)  policy={pol_ret} random={rnd_ret} oracle={orc_ret}")
