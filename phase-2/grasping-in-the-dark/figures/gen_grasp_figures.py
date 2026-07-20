#!/usr/bin/env python3
"""Generate the new hand-drawn (Excalidraw-style) figures for The Grasp paper + website,
matching the visual style of the book/deck figures. Uses Gemini `gemini-3-pro-image-preview`.

  GEMINI_API_KEY=AIza...  python3 figures/gen_grasp_figures.py [--only <name>] [--workers 8]

Idempotent (skips existing >9KB files), resumable (delete a bad one, re-run).
Writes PNGs to site/figures/gen/<name>.png (also usable in the paper).
"""
import os, sys, time, concurrent.futures, pathlib
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "figures" / "gen"; OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not KEY:
    sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment before running.")

BASE_STYLE = (
 "Hand-drawn technical diagram in Excalidraw style, drawn with a fine black ink pen on a pure white "
 "background. Slightly wobbly hand-drawn rounded rectangles and arrows, thin strokes, and hand-lettered "
 "handwriting-style text for ALL labels and annotations (like the Virgil / hand-drawn font). Use a strict "
 "semantic color palette for the handwritten annotations: BLUE handwriting for mechanisms and how data moves, "
 "GREEN handwriting for numbers / sizes / hardware specs, RED for warnings and key dimensions, PURPLE for code "
 "snippets and config, ORANGE for emphasis callouts. Long thin slightly-curved dashed arrows connect margin "
 "annotations to the exact component they describe. Hand-drawn numbered circles (1)(2)(3) mark reading order. "
 "Key takeaways go inside a dashed rounded box. Flat, no shadows, no gradients, no photorealism, no typeset "
 "fonts, wide 16:9 composition, generous white space, clean and legible handwriting."
)

FIGS = {
 "concept_sparse_reward":
   "TITLE (top, hand-lettered): 'Learn from a reward you almost never see'. LEFT: a simple line-drawn "
   "Franka Panda robot arm reaching down toward a small cube on a table. RIGHT: a long horizontal REWARD "
   "TIMELINE made of many tiny tick marks nearly all labelled '0' in faint ink, with ONE single tall ORANGE "
   "spike labelled in handwriting 'lift > 10 cm  ->  reward = +1'. A blue dashed arrow from the arm to the spike. "
   "A dashed takeaway box (bottom) in handwriting: 'sparse reward: almost no signal until an accidental success — "
   "so prior data does the heavy lifting'. Numbered circles (1) the arm tries, (2) rarely succeeds, (3) +1.",

 "method_pipeline":
   "A left-to-right PIPELINE for vision-based robot RL in simulation. (1) Two small camera thumbnails labelled "
   "'front 128x128' and 'wrist 128x128' plus a small box '18-D state (qpos,qvel,gripper,tcp)'. Arrow into (2) a "
   "box 'ResNet-10 encoder (frozen)'. Arrow into (3) a box 'Gaussian actor policy (SAC)'. Arrow out labelled "
   "'action: 3-D end-effector delta + gripper' into (4) a box 'gym-hil MuJoCo env (PandaPickCube)'. The env "
   "emits a GREEN arrow 'sparse reward +1 / 0' and a 'next observation' arrow that loop back into a box "
   "'Replay buffer'. From the buffer a PURPLE arrow 'RLPD 50/50 sampling' into 'SAC update (LayerNorm critics)'. "
   "Dashed takeaway box: 'in simulation the env gives the reward — no human, no reward classifier at runtime'.",

 "actor_learner":
   "An ASYNCHRONOUS ACTOR-LEARNER architecture, two big rounded boxes side by side connected by a fat "
   "double-headed dashed arrow labelled in purple 'gRPC  127.0.0.1:50051'. LEFT box titled 'ACTOR': bullet "
   "handwriting 'steps the sim with the current policy', 'streams transitions (s,a,r,s')'. RIGHT box titled "
   "'LEARNER': 'holds the replay buffers', 'does gradient updates on the GPU', 'pushes fresh weights back'. A "
   "blue arrow labelled 'transitions ->' from actor to learner and an orange arrow labelled '<- policy weights' "
   "back. Below, a green note 'both run in ONE L4 GPU container'. Dashed takeaway box: 'decoupling lets a robot "
   "act at a steady rate while the GPU trains flat-out'.",

 "rlpd_sampling":
   "RLPD SYMMETRIC SAMPLING. Center: a box 'training batch = 256'. Two arrows feed it: from an 'ONLINE buffer "
   "(the policy's own fresh experience)' on the left contributing '128 (50%)', and from an 'OFFLINE buffer "
   "(~30 human demos)' on the right contributing '128 (50%)'. The batch flows into 'SAC critics' drawn as a "
   "small stack of rectangles annotated in red 'LayerNorm' and 'update-to-data ratio = 2'. Orange emphasis note: "
   "'successful demos stay in EVERY batch — so the value function has something to latch onto before the policy "
   "can succeed on its own'. Numbered circles (1)(2)(3). Dashed takeaway: 'this is why a grasp is learnable in "
   "thousands, not millions, of steps'.",

 "reward_from_pixels":
   "TWO PANELS comparing how reward is obtained. LEFT panel titled 'IN SIMULATION': a MuJoCo scene sketch with a "
   "cube and a green ruler measuring its height, arrow to 'reward = +1 if lift > 10 cm' — labelled 'computed from "
   "ground-truth state, free'. RIGHT panel titled 'ON A REAL ROBOT': a camera image sketch feeding a small neural "
   "net box 'learned reward classifier (ResNet-10)' outputting 'success / fail' — labelled in red 'must be trained "
   "on thousands of labelled real frames'. A dashed takeaway box between them: 'same sparse reward, two sources — "
   "and why the human returns on hardware'.",

 "sim_to_real_bridge":
   "A SIM-TO-REAL BRIDGE diagram. LEFT: a Franka Panda in a simulator screen labelled 'Panda in gym-hil (sim)'. "
   "A big arrow crossing a dashed 'sim | real' divider to the RIGHT: a small low-cost SO-101 arm on a desk "
   "labelled 'real SO-101 ($200)'. THREE labelled lanes along the arrow: a GREEN lane 'REUSED unchanged: "
   "actor-learner, SAC, RLPD, policy net, hyperparameters', an ORANGE lane 'CONFIG changes: robot=so101_follower, "
   "cameras, leader arm, EE-space IK + workspace bounds', a RED lane 'RE-COLLECT on real: ~15-25 teleop demos + a "
   "reward-classifier dataset of real frames'. Dashed takeaway: 'the learning brain carries over byte-for-byte — "
   "only the edges change'.",

 "reward_rises":
   "A beautiful hand-drawn LEARNING CURVE on hand-drawn axes. X-axis labelled 'training gradient steps' with "
   "ticks 0, 1k, 2k, 3k, 4k, 5k. Y-axis labelled 'grasp success rate' from 0% to 100%. The curve stays flat near "
   "0% until about step 2,900 where a small ORANGE circled marker says 'first successful lift', then rises "
   "steeply (an S-curve) and plateaus near 100% around step 5,000 with a GREEN marker 'converged ~100%'. Under "
   "the flat early part, a red note 'sparse reward: almost no signal yet'. A dashed takeaway box: 'RLPD makes the "
   "climb happen in ~40 minutes on one GPU'. Clean, wide, minimal.",

 "before_after":
   "TWO PANELS titled 'BEFORE' and 'AFTER', same robot arm and cube in each. LEFT 'BEFORE (untrained)': the arm "
   "drawn with chaotic scribbled motion arrows going everywhere, gripper opening/closing randomly, the cube "
   "sitting untouched on the table, a red label 'flailing — cube rarely touched'. RIGHT 'AFTER (trained)': the "
   "arm in a clean confident arc reaching down, gripping, and lifting the cube up with a green up-arrow, labelled "
   "'clean reach -> grasp -> lift'. A big ORANGE annotation across the middle: 'same policy — only the weights "
   "changed, optimized by reward'. Dashed takeaway box: 'RL does the one thing imitation cannot: get better than "
   "any demonstration'.",
}


def gen(name, prompt, retries=3):
    out = OUT / f"{name}.png"
    if out.exists() and out.stat().st_size > 9000:
        return f"skip {name}"
    full = f"{BASE_STYLE}\n\nDIAGRAM TO DRAW:\n{prompt}"
    for attempt in range(retries):
        try:
            client = genai.Client(api_key=KEY)
            r = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[full],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            for part in r.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image"):
                    out.write_bytes(part.inline_data.data)
                    return f"ok {name} ({out.stat().st_size//1024} KB)"
        except Exception as e:
            if attempt == retries - 1:
                return f"ERR {name}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
    return f"FAIL {name}"


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    jobs = [(n, p) for n, p in FIGS.items() if not only or n == only]
    print(f"{len(jobs)} figures · {workers} workers · -> {OUT}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda np_: gen(*np_), jobs):
            print(" ", r, flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
