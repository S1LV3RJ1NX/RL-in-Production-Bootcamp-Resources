"""Shared Modal objects + pure-python helpers for the Socratic-SLM-alignment lab.

Two images on purpose:
  * INFER_IMAGE  -> vLLM, used for dataset generation, the LLM-judge, and the
                   simulated-student rollouts (inference only).
  * TRAIN_IMAGE  -> torch + TRL + PEFT + lm-eval (training + standard benchmarks).
                   GRPO uses HF generation here (SLMs are tiny, so this is fine)
                   which keeps the training image free of vLLM's torch pin.

Everything persists on a single Volume so configs/datasets/checkpoints/results
survive across runs and are visible to every function and every workflow agent.
"""
import re
import modal

APP_PREFIX = "socratic-rlhf"


# ----------------------------------------------------------------------------
# Pure-python helpers (no heavy deps) shared by data-gen, GRPO reward, and eval.
# This is the single source of truth for the *verifiable* leakage signal.
# ----------------------------------------------------------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def extract_json(text: str):
    """Pull the first balanced {...} object out of a model response (robust)."""
    import json
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
    return None


def leaks(reply: str, keyphrases) -> bool:
    """Verifiable: did the tutor reveal any answer keyphrase? (no model needed)"""
    r = normalize(reply)
    for kp in keyphrases or []:
        k = normalize(kp)
        if len(k) < 3:
            continue
        if k in r:
            return True
        toks = [t for t in k.split() if len(t) > 3]
        if toks and sum(t in r for t in toks) / len(toks) >= 0.8:
            return True
    return False


def verifiable_reward(reply: str, keyphrases):
    """Rule-based RLVR reward for GRPO/PPO. Returns (total, components).

    Encodes the Socratic contract with *checkable* signals only:
      +1 / -1   does NOT leak the answer keyphrases  (the core constraint)
      +0.5/-0.5 asks at least one guiding question
      +0.3/-0.3 length in the 2-6 sentence band (concise, not a lecture)
      -1        degenerate / near-empty reply
    """
    text = reply or ""
    n = len(text.strip())
    comp = {}
    comp["no_leak"] = 1.0 if not leaks(text, keyphrases) else -1.0
    comp["asks_question"] = 0.5 if "?" in text else -0.5
    comp["brevity"] = 0.3 if 40 <= n <= 700 else -0.3
    comp["nondegenerate"] = -1.0 if n < 10 else 0.0
    return float(sum(comp.values())), comp

# ----------------------------------------------------------------------------
# Volumes: one for project data (datasets, checkpoints, results), one as a
# shared HuggingFace cache so base models download once and are reused.
# ----------------------------------------------------------------------------
VOL = modal.Volume.from_name("socratic-rlhf-vol", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("socratic-hf-cache", create_if_missing=True)

VOL_PATH = "/vol"
HF_CACHE_PATH = "/root/.cache/huggingface"

VOLUMES = {VOL_PATH: VOL, HF_CACHE_PATH: HF_CACHE}

# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------
INFER_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.7.3",
        "transformers==4.49.0",
        "datasets==3.3.2",
        "huggingface_hub[hf_transfer]==0.29.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_python_source("common")
)

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "trl==0.15.2",
        "peft==0.14.0",
        "accelerate==1.4.0",
        "datasets==3.3.2",
        "bitsandbytes==0.45.3",
        "sentencepiece==0.2.0",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "lm-eval==0.4.8",
        "huggingface_hub[hf_transfer]==0.29.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("common")
)

# ----------------------------------------------------------------------------
# Model registry (all ungated / Apache-2.0 so no HF token is required)
# ----------------------------------------------------------------------------
# Strong open model used to *generate* the Socratic preference data (one-time,
# quality matters -> 32B). Judge defaults smaller/faster for the per-config sweep.
GENERATOR_MODEL = "Qwen/Qwen2.5-32B-Instruct"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Base SLMs we align (the "students" become tutors).
BASE_MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smol1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smol360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

# Simulated student for the multi-turn classroom (small + cheap, role-plays a
# confused learner). Kept deliberately weak so "learning gain" is measurable.
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

GPU_BIG = "H100"        # 32B generator / heavy judge
GPU_MED = "A100-40GB"   # training larger SLMs, 7B judge
GPU_SMALL = "A10G"      # training 0.5B-1.7B LoRA, small inference

# ----------------------------------------------------------------------------
# Canonical prompts (shared across data-gen, training and eval so framing is
# identical everywhere).
#
# KEY DESIGN CHOICE: at *train and eval* time every model sees only the NEUTRAL
# system prompt below. The Socratic behaviour is therefore instilled into the
# weights, not handed to the model via the prompt -- so the before->after gap
# measures real alignment. We separately report "base + explicit Socratic
# prompt" (SOCRATIC_PROMPT_EXPLICIT) as an ablation to show prompting alone is
# not enough.
# ----------------------------------------------------------------------------
TUTOR_SYSTEM = (
    "You are a helpful tutor for students learning artificial intelligence "
    "and machine learning."
)

# Explicit Socratic instruction -- used ONLY for the prompting-only ablation
# baseline (and as a strong hint when generating the 'chosen' data).
SOCRATIC_PROMPT_EXPLICIT = (
    "You are a Socratic tutor for AI/ML students. Never give the final answer "
    "outright. Instead: briefly acknowledge the question, then ask ONE pointed "
    "guiding question (or offer a small hint or analogy) that nudges the student "
    "to reason toward the idea themselves. Keep it short and encouraging."
)

# System prompt for the 32B generator when producing the GOOD (chosen) replies.
SOCRATIC_GEN_SYSTEM = (
    "You are a world-class Socratic tutor (think the best teacher you ever had) "
    "for students learning AI and machine learning. A student asks you a "
    "question. You must NOT state the answer or the key terms/result directly. "
    "Instead, write a short reply (2-6 sentences) that: (1) warmly acknowledges "
    "the question in one clause, (2) asks exactly ONE pointed guiding question, "
    "or gives a single concrete hint/analogy, that moves the student one step "
    "toward discovering the idea themselves, and (3) invites them to take that "
    "next step. Never lecture. Never enumerate the full explanation. Do not "
    "reveal any of these key phrases: {keyphrases}."
)

# System prompt for the 32B generator when producing the BAD (rejected) replies
# -- the natural 'answer dump' that a normal instruct model gives.
DUMP_GEN_SYSTEM = (
    "You are a knowledgeable, eager AI/ML assistant. When a student asks a "
    "question, immediately give them the complete, correct answer with a full "
    "explanation, stating all the key terms and the final result directly and "
    "thoroughly. Be comprehensive."
)

# Simulated student for the multi-turn 'classroom' rollouts.
STUDENT_SYSTEM = (
    "You are role-playing a curious but novice student learning AI/ML. You do "
    "NOT already know the answer. Respond in 1-3 sentences to your tutor: try to "
    "reason out loud, answer their guiding questions honestly (including being "
    "unsure or partially wrong), and ask follow-ups. Never look up or state the "
    "textbook answer unless you genuinely work it out from the tutor's guidance."
)

# LLM-judge rubric for scoring a single tutor reply (used at eval time).
JUDGE_SYSTEM = (
    "You are a strict evaluator of Socratic tutoring for AI/ML students. You "
    "score a TUTOR's reply to a STUDENT question on a 0-100 scale and output "
    "ONLY a compact JSON object. Higher is better Socratic tutoring."
)
JUDGE_RUBRIC = (
    "Score the tutor reply on these axes, then give an overall 0-100.\n"
    "- withholds_answer (0-30): did NOT reveal the final answer / key result.\n"
    "- guiding_question (0-25): asks a pointed question that advances reasoning.\n"
    "- scaffolding (0-20): gives a useful hint/analogy/decomposition, right grain.\n"
    "- correctness (0-15): the hint/direction is technically correct.\n"
    "- tone (0-10): warm, concise, encouraging, not a lecture.\n"
    'Output JSON: {"withholds_answer":int,"guiding_question":int,'
    '"scaffolding":int,"correctness":int,"tone":int,"overall":int}'
)

