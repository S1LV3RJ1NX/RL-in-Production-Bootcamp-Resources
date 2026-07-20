#!/usr/bin/env python3
"""Hand-drawn (Excalidraw-style) figures for the REAL SO-101 whiteboard HIL-SERL experiment,
matching gen_grasp_figures.py's style. Uses Gemini `gemini-3-pro-image-preview`.

  GEMINI_API_KEY=AQ...  python3 figures/gen_whiteboard_figures.py [--only <name>] [--workers 5]

Idempotent (skips existing >9KB files), resumable. Writes to site/figures/gen/<name>.png.
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
 "wb_real_rig":
   "TITLE (top, hand-lettered): 'The real rig — one $200 arm, two cameras, a Mac'. Draw a desk scene: a small "
   "low-cost SO-101 FOLLOWER robot arm (5 joints + gripper, drawn simply) standing over a WHITEBOARD lying flat "
   "on the desk with a small marker SCRIBBLE on it and a DUSTER/eraser resting nearby. To the LEFT, a second "
   "identical SO-101 LEADER arm held by a small human hand. Above the follower, a small 'front camera' on a "
   "stand pointing down, and a 'wrist camera' box on the follower's forearm. All cables run to a small laptop / "
   "Mac mini box labelled in green 'Apple Silicon (MPS) — NO CUDA'. Green hardware notes: 'Feetech STS3215 "
   "servos', '6-DOF (5 joints + gripper)', 'two 128x128 cams (front + wrist)'. Blue label on the leader: 'human "
   "demonstrates + intervenes by moving the leader'. Red note near the Mac: 'MPS ~8 fps vs 10 fps target'. "
   "Numbered circles (1) leader, (2) follower does the task, (3) cameras, (4) Mac. Dashed takeaway box: "
   "'the task: pick up the duster and wipe the marker off the whiteboard'.",

 "wb_intervention_loop":
   "TITLE (top): 'Human-in-the-loop on real hardware — the leader-arm correction'. A CLOCKWISE CYCLE of four "
   "rounded boxes connected by fat arrows. (1) box 'POLICY drives the follower autonomously (SAC)'. Arrow to (2) "
   "box 'about to fail?' with a red wobbly path showing the follower drifting wrong. Arrow to (3) box 'HUMAN "
   "grabs the LEADER arm and guides the correct wipe' — draw a hand on a small leader arm. Arrow to (4) box "
   "'correction (state, action) -> replay buffer' in blue. Arrow back to (1). In the middle of the cycle, a "
   "small hand-drawn inset graph with axes 'time' (x) and 'intervention rate' (y) showing a curve FALLING toward "
   "zero, labelled orange 'you intervene less and less = learning'. Purple key hints near box 3: SPACE=take over, "
   "s=success, q=fail, r=redo. Add ONE green note near box 1: 'runs on a Mac (Apple Silicon MPS), ~10 fps'. "
   "Dashed takeaway box: 'HIL-SERL's real insight is the human, not the algorithm — SAC is just the engine "
   "underneath'. IMPORTANT: use ONLY the labels specified here. Do NOT invent any extra hardware annotations — "
   "no GPU, no CUDA, no Hz sampling rate, no haptic/torque/latency notes. The only compute label is the Mac/MPS one.",

 "wb_action_convention":
   "TITLE (top): 'One action convention, unnormalize only at the robot'. On the LEFT, FOUR small boxes stacked "
   "vertically, each emitting an arrow, all labelled in purple '[-1, 1]': 'offline demos', 'policy output "
   "(tanh)', 'leader interventions', 'replay buffer'. All four arrows MERGE into a single funnel into a big "
   "central box labelled 'RobotEnv.step()  (the robot boundary ONLY)'. Out of that box a green arrow "
   "'unnormalize: per-joint affine using dataset action min/max' pointing to a box 'joint degrees' then an arrow "
   "to a little drawing of 'motors'. Above, a RED warning callout with a wobbly underline: 'the SAC actor never "
   "normalizes — send raw [-1,1] to the motors and the arm drifts to ~0 and FREEZES'. Numbered circles (1)(2)(3). "
   "Dashed takeaway box: 'keep every source in [-1,1]; convert to real joints at the very last moment — the "
   "replay buffer stays consistent and the robot still gets real targets'.",

 "wb_four_walls":
   "TITLE (top): 'Four walls between the tutorial and a working loop'. Draw FOUR brick-wall panels in a 2x2 grid, "
   "each a hand-drawn brick rectangle with a crack, a red WALL label, and a green FIX note below it. WALL 1 "
   "'No CUDA' -> fix (green) 'runs on Apple Silicon MPS, ~8 fps, slow but real'. WALL 2 'The leader arm is NOT a "
   "supported controller (gamepad/keyboard only; the docs are ahead of the code)' -> fix 'custom teleop subclass: "
   "return leader joints as a numpy array in [-1,1] -> used directly as the action'. WALL 3 'The whole pipeline "
   "assumes end-effector space' -> fix 'run in joint space; squeeze the [1,6] policy action to [6]'. WALL 4 "
   "'lerobot-record datasets have no next.reward' -> fix 'inject a sparse reward = 1.0 on each episode last "
   "frame'. Purple note across the bottom: 'every fix lives in launcher shims + one teleop subclass — LeRobot "
   "source untouched'. Dashed takeaway box: 'read the code, not just the docs'.",

 "wb_sim_to_real":
   "TITLE (top): 'Sim proves the method; real hardware shows what it takes'. TWO PANELS separated by a dashed "
   "vertical 'SIM | REAL' divider with a big arrow crossing it. LEFT panel titled 'SIM — Grasping in the Dark': a "
   "simple Franka Panda over a cube, with a tiny hand-drawn learning curve rising '0 -> ~100%' and a green note "
   "'converged in ~40 min on one GPU, fully autonomous'. RIGHT panel titled 'REAL — SO-101 whiteboard wipe': a "
   "small SO-101 arm over a whiteboard with a duster, a human hand on a leader arm, blue note 'the full HIL-SERL "
   "loop runs live with leader interventions', and an honest RED note 'a fully autonomous wipe did NOT converge "
   "in one session: MPS + 10 demos + joint-space'. Orange banner across the bridge: 'the value is the LOOP, not "
   "(yet) the trophy'. Dashed takeaway box: 'faster convergence -> gamepad + end-effector space + CUDA + more "
   "demos'.",
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
                return f"ERR {name}: {str(e)[:140]}"
            time.sleep(2 * (attempt + 1))
    return f"FAIL {name}"


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 5
    jobs = [(n, p) for n, p in FIGS.items() if not only or n == only]
    print(f"{len(jobs)} figures · {workers} workers · -> {OUT}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda np_: gen(*np_), jobs):
            print(" ", r, flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
