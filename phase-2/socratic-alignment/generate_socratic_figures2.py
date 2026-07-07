"""Strict regeneration of the two label-critical figures (taxonomy, pipeline)
that PaperBanana hallucinated. Hard-constrain labels verbatim (Raj's known fix).
"""
import asyncio
import shutil
from pathlib import Path

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings

OUT_DIR = Path("/Users/raj/Claude_Brain/RL-Production-Workshop/rlhf-lab/paper/figures")
ENV_PATH = "/Users/raj/Desktop/Course_Creator/.env.local"
google_key = ""
with open(ENV_PATH) as f:
    for line in f:
        if line.startswith("GOOGLE_API_KEY="):
            google_key = line.strip().split("=", 1)[1]
            break

settings = Settings(vlm_model="gemini-2.5-flash", image_model="gemini-3-pro-image-preview",
                    refinement_iterations=1, GOOGLE_API_KEY=google_key)

STRICT = (
    "CRITICAL ACCURACY RULE: This is a scientific figure. Render EXACTLY the text "
    "labels specified, verbatim. Do NOT add, remove, rename, reorder, or "
    "substitute ANY box, method, metric, or benchmark. Do NOT use your own "
    "knowledge of ML methods to 'fill in' or 'improve' the list. If a column "
    "lists two items, show EXACTLY two. "
)
STYLE = (
    STRICT +
    "Clean, minimal, professional academic-paper figure on a WHITE background. "
    "Rounded cards, thin borders, soft shadows, generous whitespace, clean "
    "sans-serif labels. GREEN = guides/good, RED/ORANGE = evades/bad, VIOLET = "
    "model, AMBER = reward, CYAN = student. Thin arrows. NO emojis, NO clutter."
)

FIGS = [
    {
        "filename": "fig_taxonomy.png",
        "context": (
            STRICT +
            "Title: 'Recipe matters more than the act of aligning'. Two columns "
            "below a single violet box 'Language model' at top center. "
            "LEFT column: green header text 'Anchored objectives -> learn to "
            "GUIDE'; below it EXACTLY FOUR white boxes, top to bottom, with these "
            "exact labels and NO others: 'SFT', 'KTO', 'ORPO', 'SimPO'; under the "
            "column a green rounded box 'Guided reply' and the caption 'retain a "
            "likelihood / SFT anchor; judge score +16 to +18 over base'. "
            "RIGHT column: red/orange header text 'Contrastive / sparse-reward -> "
            "learn to EVADE'; below it EXACTLY TWO white boxes, top to bottom, "
            "with these exact labels and NO others: 'DPO', 'GRPO'. Do NOT add "
            "IPO, CPO, SPPO, KTO, or any other method to the right column - ONLY "
            "'DPO' and 'GRPO'. Under the column a red speech bubble with the exact "
            "text 'Sure, I can help you with that!' and the caption 'satisfy the "
            "one-sided do-not-reveal constraint by saying nothing useful; lowest "
            "leakage but judge BELOW base'."
        ),
    },
    {
        "filename": "fig_pipeline.png",
        "context": (
            STRICT +
            "Title: 'Socratic Alignment - experimental pipeline'. A left-to-right "
            "flow with FOUR numbered circular stage markers (1,2,3,4). "
            "STAGE 1 'Data': a gray box 'Vizuara AI/ML curriculum (68 concepts x 4 "
            "personas)' -> a violet box 'Qwen2.5-32B generator' -> two output "
            "boxes 'preference pairs: chosen = Socratic, rejected = answer-dump' "
            "and 'SocraticBench (215 held-out)'. "
            "STAGE 2 'Align (LoRA)': one box listing exactly these seven recipes "
            "'SFT, DPO, KTO, ORPO, SimPO, GRPO, PPO' and the text 'small models: "
            "Qwen2.5-0.5B/1.5B, SmolLM2-360M/1.7B'. "
            "STAGE 3 'Evaluate': EXACTLY FOUR labelled chips with these exact "
            "labels and NO others: 'Verifiable leakage', 'LLM-judge Socratic "
            "score', 'Standard benchmarks (alignment tax)', 'Student "
            "learning-gain'. Do NOT write AlpacaEval, Arena-Hard, or MMLU "
            "anywhere. "
            "STAGE 4 'Results': a green badge with the exact text 'Before -> "
            "After'. White background, clean horizontal flow."
        ),
    },
]


async def main():
    pipe = PaperBananaPipeline(settings=settings)
    for fig in FIGS:
        print(f"-> {fig['filename']} (strict) ...", flush=True)
        try:
            r = await pipe.generate(GenerationInput(
                source_context=fig["context"], communicative_intent=STYLE,
                diagram_type=DiagramType.METHODOLOGY))
            if getattr(r, "image_path", None) and Path(r.image_path).exists():
                shutil.copy(Path(r.image_path), OUT_DIR / fig["filename"])
                print(f"   saved {fig['filename']}", flush=True)
        except Exception as e:
            print(f"   ERROR {fig['filename']}: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
