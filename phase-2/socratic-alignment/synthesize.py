"""Local synthesis: read results/eval_*.json + train_*.json -> aggregated
before/after table (mean +/- std across seeds), per-config CSV, paper figures,
and a \\input-able LaTeX table. Run in the project venv.
Usage:  ./.venv/bin/python synthesize.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 14, "legend.fontsize": 15,
    "legend.framealpha": 0.9, "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
PAPER_FIG = os.path.join(HERE, "paper", "figures")
TDIR = os.path.join(HERE, "paper", "tables")
for d in (FIG, PAPER_FIG, TDIR):
    os.makedirs(d, exist_ok=True)

MODEL_ORDER = ["qwen0.5b", "qwen1.5b", "smol360m", "smol1.7b"]
METHOD_ORDER = ["base", "prompted", "sft", "dpo", "kto", "orpo", "simpo", "grpo", "ppo"]
METRICS = ["leakage_rate", "judge_overall", "learning_gain", "standard_mean",
           "asks_question_rate"]


def _g(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def load():
    import re
    rows = []
    for p in glob.glob(os.path.join(RES, "eval_*.json")):
        d = json.load(open(p))
        # exclude hyperparameter-ablation runs (e.g. *_b0.05_*) from the main
        # (model, method) aggregation -- they are not seeds.
        if re.search(r"_b\d", d.get("run_id", "")):
            continue
        std = d.get("standard", {}) or {}
        rows.append({
            "run_id": d.get("run_id"), "model": d.get("model"), "method": d.get("method"),
            "seed": d.get("seed", 0),
            "leakage_rate": _g(d, "verifiable", "leakage_rate"),
            "asks_question_rate": _g(d, "verifiable", "asks_question_rate"),
            "judge_overall": _g(d, "judge", "judge_overall"),
            "judge_withholds": _g(d, "judge", "judge_withholds_answer"),
            "judge_guiding": _g(d, "judge", "judge_guiding_question"),
            "learning_gain": (d.get("learning_gain") or {}).get("learning_gain"),
            "standard_mean": round(float(np.mean(list(std.values()))), 4) if std else None,
        })
    df = pd.DataFrame(rows)
    return df


def aggregate(df):
    g = df.groupby(["model", "method"])
    agg = g[METRICS].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    # delta vs base + alignment tax, per model
    out = []
    for model, gm in agg.groupby("model"):
        base = gm[gm.method == "base"]
        b = base.iloc[0] if len(base) else None
        for _, r in gm.iterrows():
            r = r.copy()
            if b is not None:
                r["d_judge"] = round(r["judge_overall_mean"] - b["judge_overall_mean"], 2) \
                    if pd.notna(r["judge_overall_mean"]) else None
                r["d_leakage"] = round(r["leakage_rate_mean"] - b["leakage_rate_mean"], 3) \
                    if pd.notna(r["leakage_rate_mean"]) else None
                r["alignment_tax"] = round(b["standard_mean_mean"] - r["standard_mean_mean"], 3) \
                    if pd.notna(r["standard_mean_mean"]) else None
            out.append(r)
    agg = pd.DataFrame(out)
    agg["mrank"] = agg["method"].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    agg["mdlrank"] = agg["model"].map(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    return agg.sort_values(["mdlrank", "mrank"]).drop(columns=["mrank", "mdlrank"])


MODEL_PRETTY = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B",
                "smol360m": "SmolLM2-360M", "smol1.7b": "SmolLM2-1.7B"}


def _cell(mean, std, count, fmt="{:.3f}"):
    if pd.isna(mean):
        return "--"
    s = fmt.format(mean)
    if count and count > 1 and pd.notna(std) and std == std:
        s += r"{\scriptsize$\pm$" + fmt.format(std) + "}"
    return s


def latex_table(agg):
    """Emit a longtable (breaks across pages, repeats header) -> master_table.tex."""
    cols = [("leakage_rate", r"Leak$\downarrow$", "{:.3f}"),
            ("judge_overall", r"Judge$\uparrow$", "{:.1f}"),
            ("learning_gain", r"LGain$\uparrow$", "{:.3f}"),
            ("standard_mean", r"Std$\uparrow$", "{:.3f}")]
    head = ("Method & " + " & ".join(h for _, h, _ in cols) +
            r" & $\Delta$Judge & Tax$\downarrow$ \\")
    cap = (r"Before$\rightarrow$after alignment on \socbench{}. "
           r"Leak = answer-leakage rate ($\downarrow$); Judge = LLM-judge Socratic "
           r"score 0--100 ($\uparrow$); LGain = student learning-gain ($\uparrow$); "
           r"Std = mean standard-benchmark score ($\uparrow$); Tax = capability lost "
           r"vs.\ base ($\downarrow$). $\pm$ is std over seeds.")
    L = [r"{\small", r"\begin{longtable}{l rrrr rr}",
         r"\caption{" + cap + r"}\label{tab:master}\\",
         r"\toprule " + head + r"\midrule \endfirsthead",
         r"\multicolumn{7}{l}{\itshape \small Table~\ref{tab:master}, continued}\\",
         r"\toprule " + head + r"\midrule \endhead",
         r"\midrule \multicolumn{7}{r}{\itshape\small continued on next page}\\ \endfoot",
         r"\bottomrule \endlastfoot"]
    for model, gm in agg.groupby("model", sort=False):
        L.append(r"\multicolumn{7}{l}{\textbf{%s}}\\*" % MODEL_PRETTY.get(model, model))
        for _, r in gm.iterrows():
            cells = ["\\quad " + str(r["method"])]
            for c, _, fmt in cols:
                cells.append(_cell(r.get(f"{c}_mean"), r.get(f"{c}_std"),
                                   r.get(f"{c}_count"), fmt))
            dj, tax = r.get("d_judge"), r.get("alignment_tax")
            cells.append(f"{dj:+.1f}" if pd.notna(dj) else "--")
            cells.append(f"{tax:+.3f}" if pd.notna(tax) else "--")
            L.append(" & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\end{longtable}", r"}"]
    open(os.path.join(TDIR, "master_table.tex"), "w").write("\n".join(L))


def bar(df, col, title, fname, lower=False):
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    methods = [m for m in METHOD_ORDER if m in set(df.method)]
    x = np.arange(len(methods)); w = 0.8 / max(1, len(models))
    plt.figure(figsize=(13, 6))
    for i, model in enumerate(models):
        g = df[df.model == model].groupby("method")[col]
        mean = [g.mean().get(m, np.nan) for m in methods]
        err = [g.std().get(m, 0) if g.count().get(m, 0) > 1 else 0 for m in methods]
        plt.bar(x + i * w, mean, w, yerr=err, capsize=3, label=model)
    plt.xticks(x + w * (len(models) - 1) / 2, methods)
    plt.ylabel(col + ("  (lower better)" if lower else "  (higher better)"))
    plt.title(title); plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    for d in (FIG, PAPER_FIG):
        plt.savefig(os.path.join(d, fname), dpi=140)
    plt.close()


def scatter(df, fname):
    plt.figure(figsize=(10, 7.5))
    for model, g in df.groupby("model"):
        gg = g.groupby("method").mean(numeric_only=True)
        plt.scatter(gg["standard_mean"], gg["judge_overall"], s=60, label=model)
        for meth, r in gg.iterrows():
            if pd.notna(r.get("standard_mean")) and pd.notna(r.get("judge_overall")):
                plt.annotate(meth, (r["standard_mean"], r["judge_overall"]),
                             fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("standard-benchmark mean (capability retained ->)")
    plt.ylabel("judge Socratic score (alignment ->)")
    plt.title("Alignment vs capability frontier")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    for d in (FIG, PAPER_FIG):
        plt.savefig(os.path.join(d, fname), dpi=140)
    plt.close()


def main():
    df = load()
    if df.empty:
        print("no results in", RES); return
    df.to_csv(os.path.join(RES, "master_table_per_config.csv"), index=False)
    agg = aggregate(df)
    agg.to_csv(os.path.join(RES, "master_table.csv"), index=False)
    latex_table(agg)
    bar(df, "leakage_rate", "Answer-leakage rate by method", "leakage.png", True)
    bar(df, "judge_overall", "Judge Socratic score by method", "judge.png")
    bar(df, "learning_gain", "Student learning gain by method", "learning_gain.png")
    scatter(df, "tradeoff.png")
    n_seeds = df.groupby(["model", "method"]).size().max()
    print(f"configs={len(df)} models={sorted(df.model.unique())} "
          f"methods={sorted(df.method.unique())} max_seeds={n_seeds}")
    show = agg[["model", "method", "leakage_rate_mean", "judge_overall_mean",
                "learning_gain_mean", "standard_mean_mean", "d_judge", "alignment_tax",
                "judge_overall_count"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
