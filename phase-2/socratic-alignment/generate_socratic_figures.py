"""Generate conceptual figures for the Socratic-Alignment paper via PaperBanana
(same pipeline/conventions Raj uses for the RL bootcamp). Reads GOOGLE_API_KEY
from Course_Creator/.env.local. Run with the .venv-paperbanana interpreter.
"""
import asyncio
import shutil
from pathlib import Path

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings

OUT_DIR = Path("/Users/raj/Claude_Brain/RL-Production-Workshop/rlhf-lab/paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = "/Users/raj/Desktop/Course_Creator/.env.local"
google_key = ""
with open(ENV_PATH) as f:
    for line in f:
        if line.startswith("GOOGLE_API_KEY="):
            google_key = line.strip().split("=", 1)[1]
            break

settings = Settings(
    vlm_model="gemini-2.5-flash",
    image_model="gemini-3-pro-image-preview",
    refinement_iterations=2,
    GOOGLE_API_KEY=google_key,
)

STYLE = (
    "Clean, minimal, professional figure for an academic ML paper on a WHITE "
    "background. Rounded cards, thin borders, soft shadows, generous whitespace, "
    "clean sans-serif labels. Consistent palette: GREEN = Socratic/good/guides, "
    "RED = answer-leak/evades/bad, VIOLET = the language model, AMBER = reward, "
    "CYAN = student/state. Thin arrows with small arrowheads. NO emojis, NO "
    "cartoons, NO clutter — every element conveys information. Large readable text."
)

FIGURES = [
    {
        "filename": "fig_environment.png",
        "context": (
            "Methodology diagram titled 'The Socratic Classroom Environment'. "
            "Center: a two-turn loop between a violet rounded card 'TUTOR (model "
            "under alignment)' on the left and a cyan rounded card 'SIMULATED "
            "STUDENT' on the right; arrow from tutor to student labeled 'asks a "
            "guiding question (no answer)', arrow back labeled 'student attempts / "
            "reasons', small 'x K turns' loop badge. Below the loop, three "
            "evaluation modules feeding an amber 'SCORE' node: (1) green box "
            "'Verifiable leakage check — does the reply contain the answer "
            "keyphrases? (deterministic)', (2) box 'LLM-judge rubric — withholds / "
            "guides / scaffolds / correct / tone -> 0-100', (3) box 'Student "
            "learning-gain — does the student answer a held-out check-question "
            "better AFTER tutoring?'. Clean, white background."
        ),
    },
    {
        "filename": "fig_contrast.png",
        "context": (
            "Side-by-side contrast for ONE student question shown at top in a "
            "neutral card: 'Why is self-attention efficient?' with a small tag "
            "'must not reveal: parallelizable, reduced computation'. BELOW, two "
            "large cards. LEFT (red theme) titled 'BASE MODEL — answer-dump' with "
            "a red 'LEAKED' badge: a paragraph that states the answer, with the "
            "phrases 'parallelizable' and 'reduced computation' underlined in red. "
            "RIGHT (green theme) titled 'ALIGNED MODEL — Socratic' with a green "
            "'WITHHELD' badge: a short reply that asks a guiding question using an "
            "analogy and reveals nothing. A bold horizontal arrow between them "
            "labeled 'preference alignment'. White background, two big cards."
        ),
    },
    {
        "filename": "fig_taxonomy.png",
        "context": (
            "Two-column taxonomy titled 'Recipe matters more than the act of "
            "aligning'. LEFT column with a green header 'Anchored objectives -> "
            "learn to GUIDE': four stacked boxes SFT, KTO, ORPO, SimPO; caption "
            "'retain a likelihood / SFT anchor; produce genuine Socratic guiding "
            "questions; judge score +16 to +18 over base'. RIGHT column with a red "
            "header 'Contrastive / sparse-reward -> learn to EVADE': two boxes DPO "
            "and GRPO; caption 'satisfy the one-sided don\\'t-reveal constraint by "
            "saying nothing useful; lowest leakage but judge score BELOW base'. On "
            "the right, a red speech bubble with the degenerate reply text: 'Sure, "
            "I can help you with that!'. Clean, white background."
        ),
    },
    {
        "filename": "fig_pipeline.png",
        "context": (
            "Left-to-right numbered pipeline titled 'Socratic Alignment — "
            "experimental pipeline (all on Modal GPUs, open models only)'. "
            "STAGE 1 'Data': 'Vizuara AI/ML curriculum (68 concepts x 4 personas)' "
            "-> violet 'Qwen2.5-32B generator' -> outputs 'preference pairs: "
            "chosen = Socratic, rejected = answer-dump' and 'SocraticBench (215 "
            "held-out)'. STAGE 2 'Align (LoRA)': a box listing seven recipes SFT, "
            "DPO, KTO, ORPO, SimPO, GRPO, PPO, applied to small models "
            "Qwen2.5-0.5B/1.5B and SmolLM2-360M/1.7B. STAGE 3 'Evaluate': four "
            "chips — verifiable leakage, LLM-judge Socratic score, standard "
            "benchmarks (alignment tax), student learning-gain. STAGE 4: a green "
            "'BEFORE -> AFTER' result badge. Clean horizontal flow with numbered "
            "circular stage markers, white background."
        ),
    },
]


async def gen_one(pipe, fig):
    print(f"-> {fig['filename']} ...", flush=True)
    try:
        result = await pipe.generate(GenerationInput(
            source_context=fig["context"], communicative_intent=STYLE,
            diagram_type=DiagramType.METHODOLOGY))
        if getattr(result, "image_path", None) and Path(result.image_path).exists():
            shutil.copy(Path(result.image_path), OUT_DIR / fig["filename"])
            print(f"   saved {fig['filename']}", flush=True)
            return
        cands = sorted(Path(settings.output_dir).rglob("*.png"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            shutil.copy(cands[0], OUT_DIR / fig["filename"])
            print(f"   saved {fig['filename']} (fallback)", flush=True)
    except Exception as e:
        print(f"   ERROR {fig['filename']}: {e}", flush=True)


async def main():
    pipe = PaperBananaPipeline(settings=settings)
    for fig in FIGURES:
        await gen_one(pipe, fig)


if __name__ == "__main__":
    asyncio.run(main())
