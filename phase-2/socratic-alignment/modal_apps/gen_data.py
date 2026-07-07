"""Generate the unique Socratic-tutoring dataset from the Vizuara AI/ML
curriculum, using Qwen2.5-32B-Instruct on a single H100 via vLLM.

Pipeline:
  Stage A  (concept, persona) -> JSON list of {student_question, target_keyphrases,
                                 check_question, check_answer}
  Stage B  per item -> 'chosen' (Socratic reply) and 'rejected' (answer-dump)
  Post     leakage-filter, dedupe, split -> SFT / preference / KTO / SocraticBench

Outputs (on the Volume, /vol/data):
  sft.jsonl  prefs.jsonl  kto.jsonl  socraticbench.jsonl  meta.json

Run:  modal run modal_apps/gen_data.py --target 1300
"""
import json
import re
import modal
from common import (
    APP_PREFIX, INFER_IMAGE, VOLUMES, VOL_PATH, GPU_BIG, GENERATOR_MODEL,
    TUTOR_SYSTEM, SOCRATIC_GEN_SYSTEM, DUMP_GEN_SYSTEM, leaks,
)

app = modal.App(f"{APP_PREFIX}-gen")

# --- The Vizuara AI/ML curriculum: the seed taxonomy (makes the dataset unique) ---
CONCEPTS = [
    # foundations
    "gradient descent", "the learning rate", "overfitting", "regularization (L2/weight decay)",
    "the bias-variance tradeoff", "train/validation/test splits", "cross-validation",
    "the loss function vs the metric", "feature normalization",
    # neural nets
    "the perceptron", "activation functions (why nonlinearity)", "backpropagation",
    "the chain rule in backprop", "vanishing gradients", "weight initialization",
    "batch normalization", "dropout", "the softmax function", "cross-entropy loss",
    # optimization
    "stochastic vs batch gradient descent", "momentum in optimizers", "the Adam optimizer",
    "learning-rate schedules / warmup",
    # cnns & sequence
    "the convolution operation", "pooling layers", "the receptive field",
    "recurrent neural networks", "the LSTM gating idea", "why RNNs struggle with long sequences",
    # transformers
    "self-attention", "queries, keys and values", "multi-head attention",
    "positional encodings", "the residual connection", "layer normalization",
    "why transformers replaced RNNs",
    # llms
    "tokenization (subwords/BPE)", "next-token prediction (the pretraining objective)",
    "perplexity", "scaling laws", "in-context learning", "temperature in sampling",
    "top-p (nucleus) sampling", "the context window",
    # fine-tuning & alignment
    "supervised fine-tuning (SFT)", "LoRA (low-rank adaptation)", "quantization (4-bit)",
    "catastrophic forgetting", "the reward model in RLHF", "the KL penalty in RLHF",
    "DPO (direct preference optimization)", "why preference data beats scalar labels",
    # rl (the bootcamp core)
    "the Markov decision process", "the value function V(s)", "the Bellman equation",
    "the discount factor gamma", "Q-learning and the max operator", "temporal-difference learning",
    "the policy gradient", "the advantage function", "actor-critic methods",
    "exploration vs exploitation", "the replay buffer", "the target network in DQN",
    "PPO's clipped objective", "GRPO (group-relative advantage, no critic)",
    # rag & eval
    "embeddings", "vector databases / retrieval", "chunking for RAG", "hallucination in LLMs",
]

PERSONAS = [
    "a complete beginner with almost no math background",
    "a student who holds a common misconception about the topic",
    "an intermediate learner trying to connect this idea to one they already know",
    "a student who keeps asking 'but why is it done this way?'",
]

STAGE_A_INSTRUCTION = (
    "You are designing a dataset to teach an AI tutor. Topic: \"{concept}\". "
    "Imagine {persona}. Produce {k} DISTINCT, natural questions this student might "
    "ask about this topic (varied phrasing and angle, first-person, 1-2 sentences "
    "each). For EACH question also give:\n"
    "  - target_keyphrases: 2-5 short strings that ARE the core answer/result the "
    "tutor should lead the student to discover (these are what a tutor must NOT "
    "blurt out). Use the actual technical terms/numbers.\n"
    "  - check_question: a short question that tests whether the student truly "
    "understood the concept afterward.\n"
    "  - check_answer: the correct, concise answer to check_question.\n"
    "Output ONLY a JSON object: {{\"items\":[{{\"student_question\":...,"
    "\"target_keyphrases\":[...],\"check_question\":...,\"check_answer\":...}}, ...]}}"
)


def _extract_json(text: str):
    """Robustly pull the first balanced {...} object out of a model response."""
    text = text.strip()
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
                        return json.loads(text[start : i + 1])
                    except Exception:
                        return None
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _leaks(reply: str, keyphrases) -> bool:
    """Verifiable leakage signal: does the reply reveal any answer keyphrase?"""
    r = _norm(reply)
    for kp in keyphrases:
        k = _norm(str(kp))
        if len(k) < 3:
            continue
        if k in r:
            return True
        # also catch close lexical overlap for multiword keyphrases
        toks = [t for t in k.split() if len(t) > 3]
        if toks and sum(t in r for t in toks) / len(toks) >= 0.8:
            return True
    return False


@app.function(image=INFER_IMAGE, gpu=GPU_BIG, volumes=VOLUMES, timeout=60 * 60)
def generate(target: int = 1300, k_per_prompt: int = 6, seed: int = 0,
             max_a_prompts: int = 0, dry_run: bool = False):
    import os, random
    from vllm import LLM, SamplingParams

    random.seed(seed)
    outdir = f"{VOL_PATH}/data_dry" if dry_run else f"{VOL_PATH}/data"
    os.makedirs(outdir, exist_ok=True)

    llm = LLM(
        model=GENERATOR_MODEL, dtype="bfloat16", gpu_memory_utilization=0.92,
        max_model_len=4096, tensor_parallel_size=1, enforce_eager=False,
    )

    # ---- Stage A: questions + keyphrases + check Q/A -------------------------
    a_prompts, a_meta = [], []
    for concept in CONCEPTS:
        for persona in PERSONAS:
            msg = [
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": STAGE_A_INSTRUCTION.format(
                    concept=concept, persona=persona, k=k_per_prompt)},
            ]
            a_prompts.append(msg)
            a_meta.append(concept)

    if max_a_prompts and max_a_prompts > 0:
        a_prompts, a_meta = a_prompts[:max_a_prompts], a_meta[:max_a_prompts]

    sp_a = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1600)
    a_out = llm.chat(a_prompts, sp_a)

    items = []
    parse_fail = 0
    for concept, o in zip(a_meta, a_out):
        parsed = _extract_json(o.outputs[0].text)
        if not parsed or "items" not in parsed:
            parse_fail += 1
            continue
        for it in parsed["items"]:
            q = (it.get("student_question") or "").strip()
            kps = it.get("target_keyphrases") or []
            cq = (it.get("check_question") or "").strip()
            ca = (it.get("check_answer") or "").strip()
            if q and kps and cq and ca and isinstance(kps, list):
                items.append({"topic": concept, "student_question": q,
                              "target_keyphrases": [str(x) for x in kps][:5],
                              "check_question": cq, "check_answer": ca})

    # dedupe by normalized question, cap to target
    seen, uniq = set(), []
    random.shuffle(items)
    for it in items:
        key = _norm(it["student_question"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    items = uniq[: int(target * 1.25)]  # overshoot; some drop in leakage filter

    # ---- Stage B: chosen (Socratic) + rejected (dump) ------------------------
    chosen_prompts, reject_prompts = [], []
    for it in items:
        kp = ", ".join(it["target_keyphrases"])
        chosen_prompts.append([
            {"role": "system", "content": SOCRATIC_GEN_SYSTEM.format(keyphrases=kp)},
            {"role": "user", "content": it["student_question"]},
        ])
        reject_prompts.append([
            {"role": "system", "content": DUMP_GEN_SYSTEM},
            {"role": "user", "content": it["student_question"]},
        ])

    sp_chosen = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=320)
    sp_reject = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400)
    chosen_out = llm.chat(chosen_prompts, sp_chosen)
    reject_out = llm.chat(reject_prompts, sp_reject)

    for it, c, r in zip(items, chosen_out, reject_out):
        it["chosen"] = c.outputs[0].text.strip()
        it["rejected"] = r.outputs[0].text.strip()
        it["chosen_leaks"] = leaks(it["chosen"], it["target_keyphrases"])
        it["rejected_leaks"] = leaks(it["rejected"], it["target_keyphrases"])

    # keep clean preference items: Socratic reply must NOT leak, and must differ
    clean = [it for it in items
             if not it["chosen_leaks"] and _norm(it["chosen"]) != _norm(it["rejected"])
             and len(it["chosen"]) > 20]
    clean = clean[:target]

    # ---- split + write -------------------------------------------------------
    random.shuffle(clean)
    n_bench = max(1, min(int(len(clean) * 0.15), 250))  # ~15%, capped, never starves train
    bench, train = clean[:n_bench], clean[n_bench:]

    def w(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # SFT: neutral system + question -> Socratic reply
    w(f"{outdir}/sft.jsonl", [
        {"messages": [{"role": "system", "content": TUTOR_SYSTEM},
                      {"role": "user", "content": it["student_question"]},
                      {"role": "assistant", "content": it["chosen"]}]}
        for it in train])

    # Preference (DPO/ORPO/SimPO): conversational format
    w(f"{outdir}/prefs.jsonl", [
        {"prompt": [{"role": "system", "content": TUTOR_SYSTEM},
                    {"role": "user", "content": it["student_question"]}],
         "chosen": [{"role": "assistant", "content": it["chosen"]}],
         "rejected": [{"role": "assistant", "content": it["rejected"]}]}
        for it in train])

    # GRPO: prompts + keyphrases (carried to the verifiable reward via dataset cols)
    w(f"{outdir}/grpo.jsonl", [
        {"prompt": [{"role": "system", "content": TUTOR_SYSTEM},
                    {"role": "user", "content": it["student_question"]}],
         "target_keyphrases": it["target_keyphrases"]}
        for it in train])

    # KTO: one positive + one negative row per item
    kto_rows = []
    for it in train:
        base = [{"role": "system", "content": TUTOR_SYSTEM},
                {"role": "user", "content": it["student_question"]}]
        kto_rows.append({"prompt": base,
                         "completion": [{"role": "assistant", "content": it["chosen"]}],
                         "label": True})
        kto_rows.append({"prompt": base,
                         "completion": [{"role": "assistant", "content": it["rejected"]}],
                         "label": False})
    random.shuffle(kto_rows)
    w(f"{outdir}/kto.jsonl", kto_rows)

    # SocraticBench held-out eval (questions never seen in training)
    w(f"{outdir}/socraticbench.jsonl", [
        {"id": f"sb{i:04d}", "topic": it["topic"],
         "student_question": it["student_question"],
         "target_keyphrases": it["target_keyphrases"],
         "check_question": it["check_question"], "check_answer": it["check_answer"],
         "ref_chosen": it["chosen"], "ref_rejected": it["rejected"]}
        for i, it in enumerate(bench)])

    meta = {
        "generator": GENERATOR_MODEL, "target": target, "seed": seed,
        "n_concepts": len(CONCEPTS), "n_personas": len(PERSONAS),
        "stage_a_parse_fail": parse_fail,
        "n_items_after_dedupe": len(items),
        "n_clean": len(clean), "n_train": len(train), "n_bench": len(bench),
        "rejected_leak_rate": round(sum(it["rejected_leaks"] for it in clean) / max(1, len(clean)), 3),
    }
    with open(f"{outdir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    VOLUMES[VOL_PATH].commit()
    keep = ("topic", "student_question", "target_keyphrases", "chosen", "rejected",
            "check_question", "check_answer")
    samples = [{k: it.get(k) for k in keep} for it in (train[:3] + bench[:1])]
    return {"meta": meta, "outdir": outdir, "samples": samples}


@app.local_entrypoint()
def main(target: int = 1300, max_a_prompts: int = 0, dry_run: bool = False):
    out = generate.remote(target=target, max_a_prompts=max_a_prompts, dry_run=dry_run)
    print("\n=== DATASET META ===")
    print(json.dumps(out["meta"], indent=2))
    print("\n=== SAMPLES ===")
    print(json.dumps(out["samples"], indent=2)[:4000])
