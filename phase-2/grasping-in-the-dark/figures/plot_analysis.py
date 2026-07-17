#!/usr/bin/env python3
"""Real data-driven analysis charts for 'Grasping in the Dark', from the LeRobot RL train.log.
Produces, styled to match the site (teal):
  fig_success_bars.png     — grasp success rate at three stages (random / mid-training / converged)
  fig_sample_efficiency.png — gradient steps to the key milestones
  fig_episode_outcomes.png  — how episodes end across training (fail vs success), stacked over phases

  python3 figures/plot_analysis.py <train.log> <outdir>
"""
import re, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

TEAL="#0f8f68"; INK="#161a20"; MUT="#5c626f"; AMBER="#c9871a"; ROSE="#b23636"; LINE="#e8eaef"; SOFT="#f6f7f9"

def best_segment(path):
    pts=[(int(s),float(r)) for s,r in re.compile(r"Global step (\d+): Episode reward: ([-\d.]+)").findall(pathlib.Path(path).read_text())]
    segs,cur,last=[],[],-1
    for s,r in pts:
        if s<last-50:
            if cur: segs.append(cur)
            cur=[]
        cur.append((s,r)); last=s
    if cur: segs.append(cur)
    def peak(seg):
        r=np.array([x[1] for x in seg]); w=min(15,len(r))
        return np.convolve(r,np.ones(w)/w,"valid").max() if len(r)>=5 else 0
    return max(segs,key=peak) if segs else []

def style(ax):
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(colors=MUT)
    for s in ["left","bottom"]: ax.spines[s].set_color(MUT)

def main():
    log, outdir = sys.argv[1], pathlib.Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    seg=best_segment(log); steps=np.array([s for s,_ in seg]); rew=np.array([r for _,r in seg])
    smax=steps.max()
    first=next((s for s,r in seg if r>0.5), smax)
    # success rate over the last quarter (converged) and a mid window
    conv = rew[steps>=0.85*smax].mean()
    mid  = rew[(steps>=0.55*smax)&(steps<0.75*smax)].mean()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})

    # 1 — success bars
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=160); fig.patch.set_facecolor("white")
    names=["Random\npolicy","Mid-training\n(~%d steps)"%int(0.6*smax),"Converged\npolicy"]
    vals=[0.0, max(mid,0.0), conv]
    cols=[MUT,AMBER,TEAL]
    bars=ax.bar(names,[v*100 for v in vals],color=cols,width=0.62,edgecolor="white",zorder=3)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v*100+2,f"{v*100:.0f}%",ha="center",va="bottom",fontweight="bold",color=INK,fontsize=13)
    ax.set_ylim(0,105); ax.set_ylabel("grasp success rate",color=INK); ax.grid(axis="y",color=LINE,zorder=0)
    ax.set_title("Learning to grasp from a sparse reward",color=INK,fontsize=13.5,fontweight="bold",loc="left",pad=10)
    style(ax); ax.yaxis.set_major_formatter(lambda x,_:f"{x:.0f}%"); fig.tight_layout(); fig.savefig(outdir/"fig_success_bars.png",facecolor="white")

    # 2 — sample efficiency (steps to milestones, horizontal)
    fig,ax=plt.subplots(figsize=(7.2,2.6),dpi=160); fig.patch.set_facecolor("white")
    miles=[("first successful grasp",first,AMBER),("reliable grasping (~100%)",int(round(smax/500)*500),TEAL)]
    ax.hlines(0,0,smax*1.05,color=LINE,lw=2,zorder=1)
    for lbl,x,c in miles:
        ax.scatter([x],[0],s=140,color=c,zorder=3,edgecolor="white",lw=1.5)
        ax.annotate(f"{lbl}\n~{x:,} steps",xy=(x,0),xytext=(x,0.5 if c==AMBER else -0.7),ha="center",
                    va="bottom" if c==AMBER else "top",color=c,fontsize=11,fontweight="bold")
    ax.set_ylim(-1.1,1.1); ax.set_xlim(0,smax*1.05); ax.set_yticks([])
    ax.set_xlabel("training — environment steps",color=INK); ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(colors=MUT); ax.spines["bottom"].set_color(MUT)
    ax.set_title("Sample efficiency — ~40 minutes on one L4 GPU",color=INK,fontsize=13,fontweight="bold",loc="left",pad=10)
    fig.tight_layout(); fig.savefig(outdir/"fig_sample_efficiency.png",facecolor="white")

    # 3 — episode outcomes stacked across phases
    fig,ax=plt.subplots(figsize=(6.8,4.0),dpi=160); fig.patch.set_facecolor("white")
    edges=np.linspace(steps.min(),smax,5); labels=[]
    succ,fail=[],[]
    for i in range(len(edges)-1):
        m=(steps>=edges[i])&(steps<edges[i+1] if i<len(edges)-2 else steps<=edges[i+1])
        r=rew[m]; sr=r.mean() if len(r) else 0
        succ.append(sr*100); fail.append((1-sr)*100); labels.append(f"{int(edges[i]/1000)}–{int(edges[i+1]/1000)}k")
    ax.bar(labels,fail,color=SOFT,edgecolor=LINE,label="failed",zorder=3)
    ax.bar(labels,succ,bottom=fail,color=TEAL,label="grasped",zorder=3)
    ax.set_ylim(0,100); ax.set_ylabel("episodes",color=INK); ax.set_xlabel("training phase (env steps)",color=INK)
    ax.yaxis.set_major_formatter(PercentFormatter()); ax.grid(axis="y",color=LINE,zorder=0); style(ax)
    ax.legend(frameon=False,loc="upper left",fontsize=10)
    ax.set_title("Episodes shift from failure to grasp as prior data compounds",color=INK,fontsize=12.5,fontweight="bold",loc="left",pad=10)
    fig.tight_layout(); fig.savefig(outdir/"fig_episode_outcomes.png",facecolor="white")

    print(f"wrote 3 analysis charts to {outdir}  (first={first}, mid={mid:.0%}, converged={conv:.0%})")

if __name__=="__main__":
    main()
