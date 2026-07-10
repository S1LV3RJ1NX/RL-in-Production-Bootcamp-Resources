#!/usr/bin/env python3
"""Paper-banana (gemini-3-pro-image-preview) figure generator for Dreaming to Dodge.

Two styles:
  * excali  -> hand-drawn Excalidraw, white bg (website + slides)
  * paper   -> clean, professional flat-vector academic illustration (the paper)
Parallel, idempotent (skips existing >9KB), retries, re-encodes to real PNG.

  export GEMINI_API_KEY=AQ...
  ./.plotvenv/bin/python gen_pb_figures.py [--only <name>] [--workers 12]
"""
import os, sys, io, time, concurrent.futures, pathlib
from google import genai
from google.genai import types
from PIL import Image

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "figures" / "pb"; OUT.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not KEY: sys.exit("Set GEMINI_API_KEY")

EXCALI = (
 "Hand-drawn technical diagram in Excalidraw style, drawn with a fine ink pen on a PURE WHITE background. "
 "Slightly wobbly hand-drawn rounded rectangles, thin strokes, hand-lettered handwriting-style labels "
 "(Virgil / hand-drawn font). Semantic colors: BLUE for mechanisms and boxes, GREEN for numbers/sizes, "
 "RED for warnings/death, ORANGE for fireballs and emphasis, PURPLE for the policy/agent. Thin slightly-"
 "curved dashed arrows. Flat, no shadows, no gradients, no photorealism, no typeset fonts. Wide, clean, "
 "friendly, generous white space, legible handwriting. Beginner-friendly and inviting.")

PAPER = (
 "Clean, professional academic figure in a modern flat-vector illustration style, on a PURE WHITE "
 "background. Crisp SMOOTH rounded rectangles (not wobbly), thin precise strokes, a refined muted palette "
 "(deep teal #0f8f68, slate blue #2f6fd0, warm amber #c9871a, soft neutral grey), clean sans-serif labels. "
 "Subtle flat color fills for depth, NO gradients, NO drop shadows, NO photorealism. Elegant, balanced, "
 "publication-quality composition with generous white space. Beautiful and illustrative, like a top-tier "
 "ML paper / textbook figure.")

STYLES = {"excali": EXCALI, "paper": PAPER}

# (name, style, prompt)
JOBS = [
 # ---------- concept / hero (excali, for web + slides) ----------
 ("rl_loop", "excali",
  "A clean diagram of the reinforcement-learning loop as a big circle of arrows between two boxes: a box "
  "labelled 'AGENT' (a small friendly robot) on the left and a box labelled 'ENVIRONMENT' (a little game world) "
  "on the right. Top arrow from agent to environment labelled 'action'. Bottom arrows from environment back to "
  "agent labelled 'observation (what it sees)' and 'reward (+1 survived)' in green. Big handwriting underneath: "
  "'act, see, get reward — repeat MILLIONS of times in the REAL world' with 'MILLIONS' and 'REAL' emphasised in "
  "red as the expensive part."),
 ("concept_dream", "excali",
  "A friendly robot/AI agent asleep with a big thought-bubble 'dream' above it. Inside the dream bubble: a "
  "first-person Doom-like dungeon corridor with a bright orange fireball flying toward the viewer, and the "
  "agent-avatar strafing sideways to dodge. Label the bubble 'the dream (a learned world model)'. Below the "
  "sleeping agent write in handwriting: 'practise inside a dream, then play for real'. Warm and inviting."),
 ("modelfree_vs_modelbased", "excali",
  "Two panels side by side titled 'MODEL-FREE' (left) and 'MODEL-BASED / WORLD MODEL' (right). LEFT: an agent "
  "repeatedly bumping into a real environment box over and over with many arrows and a big '$$$ millions of "
  "tries' red label. RIGHT: the agent watches the real world only a little (few arrows), builds a little "
  "'world model' box, then practises many times inside that model (many arrows into the model). Green label "
  "'learn more from less real data'."),
 ("pipeline", "excali",
  "A left-to-right pipeline of four rounded boxes: (1) 'VizDoom take_cover — 64x64 frame' with a tiny fireball "
  "doodle, (2) 'Tokenizer (VQ-VAE): frame -> 64 tokens', (3) 'World model (GPT): next tokens + reward + done', "
  "(4) 'Controller (tiny MLP)'. Solid arrows between them. A blue DASHED loop arrow from the controller back to "
  "the world model labelled 'imagination loop (no real game)'. Bottom handwriting: 'trained 100% inside the dream'."),
 ("tokenizer_intuition", "excali",
  "Show a small first-person Doom frame on the left with a bright fireball. An arrow to an 8x8 GRID of little "
  "numbered squares (discrete tokens, some numbers like 47, 12, 305) labelled '64 tokens'. Another arrow to a "
  "decoded frame on the right that looks the same. Circle the fireball in both frames with an ORANGE loop and "
  "handwriting 'keep the small bright thing!'. Label 'VQ-VAE: compress a frame into a few discrete tokens'."),
 ("wm_predicts_next", "excali",
  "A horizontal sequence of small tiles: [frame tokens][action arrow][frame tokens][action arrow] feeding into a "
  "big blue box labelled 'GPT world model'. Out of the box, three outputs: 'next frame tokens' (a new tile), "
  "'reward +1' (green), and 'done? skull' (red skull). Handwriting: 'predict the next frame, the reward, and "
  "whether you died — just like next-word prediction, but for a game'."),
 ("imagination_rollout", "excali",
  "A chain of 5 dreamed first-person Doom frames left to right (each generated from the last), connected by "
  "arrows, each arrow labelled with a tiny controller choosing left/right. A fireball approaches across the "
  "frames and the view strafes away from it. Big handwriting above: 'the agent practises on frames the "
  "Transformer PREDICTS — it never touches the real game'."),
 ("exploitation", "excali",
  "A cartoon of 'cheating the dream'. LEFT bubble labelled 'IN THE DREAM': the agent stands still / always goes "
  "left and a happy face + 'survives 55!' (green). RIGHT box labelled 'IN REALITY': the same fixed action walks "
  "straight into a fireball, a red X and 'dies at 45'. A confused world-model box in the middle. RED "
  "handwriting: 'a strong policy EXPLOITS the imperfect dream'."),
 ("cma_controller", "excali",
  "Show a TINY controller drawn as a small box with only a few knobs/dials labelled '~1800 numbers'. Around it, "
  "a population of ~8 little candidate copies (dots) being scored inside a dream bubble and the best ones kept "
  "(CMA-ES evolution arrows converging). Handwriting: 'too small to cheat + gradient-free = robust'. Purple "
  "accents for the controller."),
 ("heldout_selection", "excali",
  "A row of candidate controllers (little icons) each with a dream-score. An arrow to a gate labelled 'try on "
  "UNSEEN real levels'. Only the ones that actually survive on fresh levels pass through (green check); others "
  "are rejected (red X). Handwriting: 'pick the one that TRANSFERS, not the one that looks best in the dream'."),
 ("reactive_dodge", "excali",
  "Two small first-person Doom frames. TOP: a fireball on the LEFT side, a big curved arrow and label 'go RIGHT'. "
  "BOTTOM: a fireball on the RIGHT side, arrow and label 'go LEFT'. Center handwriting: 'react to where the "
  "fireball is — genuine dodging, not a fixed move'."),
 ("ball_outbreak_fireball", "excali",
  "Three panels titled 'Catch', 'Epidemic', 'Doom'. Panel1: a paddle and a tiny white ball. Panel2: a grid of "
  "green dots with a few red infected dots. Panel3: a first-person view with a small orange fireball. Under all "
  "three, one handwriting caption spanning them: 'the same lesson — don't let the tokenizer drop the small "
  "lethal thing'."),
 ("results_concept", "excali",
  "A friendly bar-chart drawn by hand comparing survival: 'random' (short grey bar ~67), 'model-free DQN' (blue "
  "~90), 'OUR agent, trained in a dream' (tall green ~97, with a star), 'oracle' (amber ~98). A dashed line high "
  "up labelled 'solved (188)'. Big handwriting: 'trained ONLY in imagination — beats model-free RL, matches the "
  "hand-coded expert'."),
 ("sample_efficiency", "excali",
  "A funnel: a small pile labelled '200k real frames' goes into a 'world model' box, which then produces a huge "
  "cloud of 'millions of imagined practice frames'. Handwriting: 'squeeze far more practice out of little real "
  "experience'. Green numbers."),
 ("play_the_dream", "excali",
  "A person at a keyboard pressing arrow keys, connected to a screen showing a first-person Doom dream frame, "
  "with a label 'no game engine — a Transformer draws each frame'. A small fireball on screen. Handwriting: "
  "'you can literally PLAY inside the neural network'."),

 # ---------- paper (clean academic illustrative) ----------
 ("paper_hero", "paper",
  "An elegant academic hero illustration: on the left a small stack labelled 'few real frames', flowing into a "
  "central neural 'World Model' module (a clean rounded panel with subtle token-grid and a small transformer "
  "motif), which emits a soft 'dream' region containing miniature first-person game frames with a small fireball; "
  "a compact 'policy' node practises inside the dream; an arrow exits to a 'real game' node with a trophy. Minimal, "
  "refined, teal/blue/amber palette."),
 ("paper_pipeline", "paper",
  "A clean academic pipeline diagram, left to right: 'Environment (VizDoom take_cover, 64x64)' -> 'VQ-VAE "
  "Tokenizer (64 discrete tokens)' -> 'Transformer World Model (next tokens, reward, done)' -> 'Controller (CMA-ES, "
  "~1.8k params)'. A labelled dashed feedback loop 'imagination (no real environment)' between world model and "
  "controller. Small caption strip 'policy trained 100% in imagination'. Precise, publication-quality."),
 ("paper_architecture", "paper",
  "A clean three-panel architecture figure. Panel A 'Tokenizer': a CNN encoder -> a codebook of vectors -> CNN "
  "decoder (VQ-VAE), input/output 64x64 frame. Panel B 'World Model': an autoregressive Transformer over an "
  "interleaved sequence of frame tokens and action tokens, with three prediction heads (tokens / reward / done). "
  "Panel C 'Controller': a tiny 2-layer MLP mapping a pooled reconstructed-image feature to 3 actions. Muted "
  "academic palette, crisp labels."),
 ("paper_exploitation", "paper",
  "A clean academic figure illustrating world-model exploitation: two curves/bars contrasting 'imagined return' "
  "(high) versus 'real-env survival' (low) for a naive policy, with a small inset showing the policy collapsing to "
  "a single fixed action. A short caption motif 'the policy exploits model error'. Elegant, minimal."),
 ("paper_recipe", "paper",
  "A clean academic 'method recipe' figure with four labelled stages flowing into a robust agent: (1) tiny "
  "gradient-free controller, (2) reconstructed-image feature for dream->real transfer, (3) held-out model "
  "selection, (4) best-of-N. Each as a small refined icon+label. Muted teal/blue palette, publication-quality."),
 ("paper_imagination", "paper",
  "A clean academic schematic of imagination-based policy learning: a real start frame is encoded to tokens, then "
  "the world model autoregressively unrolls a horizon of dreamed latent frames while the controller selects actions "
  "and accumulates predicted reward; a bracket labels the whole unrolled trajectory 'imagined rollout (H steps)'. "
  "Elegant, minimal, teal/blue."),
]


def gen(name, style, prompt, retries=3):
    out = OUT / f"{name}.png"
    if out.exists() and out.stat().st_size > 9000:
        return f"skip {name}"
    full = f"{STYLES[style]}\n\nDIAGRAM TO DRAW:\n{prompt}"
    for attempt in range(retries):
        try:
            client = genai.Client(api_key=KEY)
            r = client.models.generate_content(
                model="gemini-3-pro-image-preview", contents=[full],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]))
            for part in r.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image"):
                    Image.open(io.BytesIO(part.inline_data.data)).convert("RGB").save(out, "PNG")
                    return f"ok {name}"
        except Exception as e:
            if attempt == retries - 1:
                return f"ERR {name}: {str(e)[:80]}"
            time.sleep(2 * (attempt + 1))
    return f"FAIL {name}"


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    jobs = [j for j in JOBS if (only is None or j[0] == only)]
    print(f"{len(jobs)} figures · {workers} workers", flush=True)
    made = fails = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(gen, *j): j[0] for j in jobs}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r.startswith(("ERR", "FAIL")): fails += 1
            elif r.startswith("ok"): made += 1
            print(" ", r, flush=True)
    print(f"DONE. {made} generated, {fails} failed, {len(jobs)-made-fails} skipped.", flush=True)


if __name__ == "__main__":
    main()
