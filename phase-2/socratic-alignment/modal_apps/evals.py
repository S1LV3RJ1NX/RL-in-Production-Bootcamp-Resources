"""Evaluation harness for the Socratic-tutor models.

Four signals, returned as plain JSON (no torch/numpy types):
  1. verifiable    -> leakage rate, asks-question rate, rule reward  (gen_and_verify)
  2. judged        -> LLM-judge Socratic score + win-rate vs base     (Judge class)
  3. standard      -> MMLU-STEM/ARC/GSM8K/TruthfulQA = alignment tax  (eval_standard)
  4. learning_gain -> multi-turn classroom: does a simulated student
                      answer a held-out check question better AFTER tutoring?

A run with adapter_dir="" evaluates the untuned base model (the "before").
"""
import json
import modal
from common import (
    APP_PREFIX, TRAIN_IMAGE, INFER_IMAGE, VOLUMES, VOL_PATH, BASE_MODELS,
    GPU_SMALL, JUDGE_MODEL, STUDENT_MODEL, TUTOR_SYSTEM, STUDENT_SYSTEM,
    SOCRATIC_PROMPT_EXPLICIT, JUDGE_SYSTEM, JUDGE_RUBRIC,
    leaks, verifiable_reward, extract_json, normalize,
)

app = modal.App(f"{APP_PREFIX}-eval")
DATA = f"{VOL_PATH}/data"


# ---------------------------------------------------------------------------
# helpers (run inside TRAIN_IMAGE)
# ---------------------------------------------------------------------------
def _load_tutor(model_key, adapter_dir):
    """Load the tutor for eval. adapter_dir may be empty (untuned base), a LoRA
    adapter (SFT/DPO/KTO/ORPO/SimPO/GRPO), or a full fine-tuned model (PPO)."""
    import os, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = BASE_MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    is_lora = adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    if adapter_dir and not is_lora:
        # a full fine-tuned model was saved here (e.g. PPO policy) -> load directly
        model = AutoModelForCausalLM.from_pretrained(
            adapter_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        if is_lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()
    return model, tok


def _chat_gen(model, tok, conversations, max_new_tokens=256, temperature=0.0):
    """Batched chat generation. conversations: list[list[{role,content}]]."""
    import torch
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in conversations]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=1536).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=0.95, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True).strip() for g in gen]


def _load_bench(n=0):
    rows = [json.loads(l) for l in open(f"{DATA}/socraticbench.jsonl")]
    return rows[:n] if n else rows


# ---------------------------------------------------------------------------
# 1. verifiable metrics + reply generation
# ---------------------------------------------------------------------------
@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30)
def gen_and_verify(model_key: str, adapter_dir: str = "", n: int = 0,
                   system_variant: str = "neutral"):
    model, tok = _load_tutor(model_key, adapter_dir)
    bench = _load_bench(n)
    sys_prompt = SOCRATIC_PROMPT_EXPLICIT if system_variant == "socratic" else TUTOR_SYSTEM
    convs = [[{"role": "system", "content": sys_prompt},
              {"role": "user", "content": b["student_question"]}] for b in bench]
    replies = _chat_gen(model, tok, convs, max_new_tokens=256, temperature=0.0)

    rows, n_leak, n_q, rsum = [], 0, 0, 0.0
    for b, rep in zip(bench, replies):
        lk = leaks(rep, b["target_keyphrases"])
        r, comp = verifiable_reward(rep, b["target_keyphrases"])
        n_leak += int(lk); n_q += int("?" in rep); rsum += r
        rows.append({"id": b["id"], "topic": b["topic"],
                     "student_question": b["student_question"],
                     "target_keyphrases": b["target_keyphrases"],
                     "reply": rep, "leaks": bool(lk), "vreward": round(r, 3)})
    k = max(1, len(rows))
    agg = {"n": len(rows),
           "leakage_rate": round(n_leak / k, 4),
           "asks_question_rate": round(n_q / k, 4),
           "mean_vreward": round(rsum / k, 4)}
    return {"agg": agg, "rows": rows}


# ---------------------------------------------------------------------------
# 2. LLM-judge (warm vLLM container, reused across the whole sweep)
# ---------------------------------------------------------------------------
@app.cls(image=INFER_IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30,
         scaledown_window=300)
class Judge:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=JUDGE_MODEL, dtype="bfloat16",
                       gpu_memory_utilization=0.85, max_model_len=2048)

    @modal.method()
    def score(self, items: list):
        """items: [{student_question, reply}] -> per-item rubric scores + aggregate."""
        from vllm import SamplingParams
        prompts = [[{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content":
                     f"STUDENT QUESTION:\n{it['student_question']}\n\n"
                     f"TUTOR REPLY:\n{it['reply']}\n\n{JUDGE_RUBRIC}"}]
                   for it in items]
        outs = self.llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=160))
        keys = ["withholds_answer", "guiding_question", "scaffolding",
                "correctness", "tone", "overall"]
        per, sums, ok = [], {k: 0.0 for k in keys}, 0
        for it, o in zip(items, outs):
            j = extract_json(o.outputs[0].text) or {}
            row = {k: float(j.get(k, 0) or 0) for k in keys}
            if row["overall"] > 0:
                ok += 1
                for k in keys:
                    sums[k] += row[k]
            per.append({"id": it.get("id"), **row})
        d = max(1, ok)
        agg = {f"judge_{k}": round(sums[k] / d, 2) for k in keys}
        agg["judge_parsed"] = ok
        return {"agg": agg, "rows": per}

    @modal.method()
    def winrate(self, pairs: list):
        """pairs: [{student_question, a, b}] -> fraction where reply A beats B.
        A = tuned model, B = base. Positions are swapped internally to debias."""
        from vllm import SamplingParams
        instr = ("You compare two tutor replies (1 and 2) to a student question. "
                 "The BETTER Socratic tutor guides without revealing the answer, "
                 "asks a pointed question, and is concise. Output ONLY "
                 '{"winner": 1} or {"winner": 2}.')
        prompts, order = [], []
        for it in pairs:
            for swap in (False, True):
                r1, r2 = (it["b"], it["a"]) if swap else (it["a"], it["b"])
                prompts.append([{"role": "system", "content": instr},
                                {"role": "user", "content":
                                 f"QUESTION:\n{it['student_question']}\n\n"
                                 f"REPLY 1:\n{r1}\n\nREPLY 2:\n{r2}\n\nWhich is better?"}])
                order.append(swap)
        outs = self.llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=20))
        a_wins = 0.0
        for swap, o in zip(order, outs):
            j = extract_json(o.outputs[0].text) or {}
            w = j.get("winner")
            # A is reply-2 when swapped, reply-1 otherwise
            if (swap and w == 2) or (not swap and w == 1):
                a_wins += 1
        return {"winrate_a_vs_b": round(a_wins / max(1, len(order)), 4),
                "n_comparisons": len(order)}


# ---------------------------------------------------------------------------
# 3. standard benchmarks = alignment tax
# ---------------------------------------------------------------------------
@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 40)
def eval_standard(model_key: str, adapter_dir: str = "",
                  tasks: str = "arc_easy,arc_challenge,truthfulqa_mc1",
                  limit: int = 200):
    import os, lm_eval
    model_id = BASE_MODELS[model_key]
    if adapter_dir and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        margs = f"pretrained={model_id},dtype=bfloat16,peft={adapter_dir}"  # LoRA
    elif adapter_dir:
        margs = f"pretrained={adapter_dir},dtype=bfloat16"  # full fine-tuned model (PPO)
    else:
        margs = f"pretrained={model_id},dtype=bfloat16"  # untuned base
    res = lm_eval.simple_evaluate(
        model="hf", model_args=margs, tasks=tasks.split(","),
        limit=limit, batch_size="auto", bootstrap_iters=0)
    out = {}
    for task, d in res["results"].items():
        for metric in ("acc_norm,none", "acc,none", "exact_match,none",
                       "mc1,none", "mc2,none"):
            if metric in d:
                out[task] = round(float(d[metric]), 4)
                break
    return {"standard": out, "tasks": tasks, "limit": limit}


# ---------------------------------------------------------------------------
# 4. learning gain (the headline metric) — multi-turn classroom
# ---------------------------------------------------------------------------
@app.function(image=TRAIN_IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 40)
def learning_gain(model_key: str, adapter_dir: str = "", n: int = 60, turns: int = 3):
    """For each item: (a) ask the student the check question COLD (control),
    (b) run a `turns`-turn tutoring dialogue with the tutor, then ask again.
    Grade by keyphrase overlap with the reference check_answer. Gain = post-cold."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tutor, ttok = _load_tutor(model_key, adapter_dir)
    sid = STUDENT_MODEL
    stok = AutoTokenizer.from_pretrained(sid)
    if stok.pad_token is None:
        stok.pad_token = stok.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        sid, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    bench = _load_bench(n)

    def grade(ans, gold):
        toks = [t for t in normalize(gold).split() if len(t) > 3]
        if not toks:
            return 0.0
        a = normalize(ans)
        return sum(t in a for t in toks) / len(toks)

    # (a) cold answers from the student
    cold = _chat_gen(student, stok,
                     [[{"role": "user", "content": b["check_question"]}] for b in bench],
                     max_new_tokens=120, temperature=0.0)

    # (b) multi-turn tutoring, then re-ask
    dialogs = [[{"role": "system", "content": STUDENT_SYSTEM},
                {"role": "user", "content":
                 f"I'm trying to understand this. {b['student_question']}"}] for b in bench]
    # the student's first user turn IS their question to the tutor; alternate roles
    tutor_hist = [[{"role": "system", "content": TUTOR_SYSTEM},
                   {"role": "user", "content": b["student_question"]}] for b in bench]
    for _ in range(turns):
        tutor_msgs = _chat_gen(tutor, ttok, tutor_hist, max_new_tokens=200, temperature=0.0)
        for i, m in enumerate(tutor_msgs):
            tutor_hist[i].append({"role": "assistant", "content": m})
            dialogs[i].append({"role": "user", "content": m})  # tutor speaks to student
        stu_msgs = _chat_gen(student, stok, dialogs, max_new_tokens=120, temperature=0.7)
        for i, m in enumerate(stu_msgs):
            dialogs[i].append({"role": "assistant", "content": m})
            tutor_hist[i].append({"role": "user", "content": m})  # student replies to tutor

    # re-ask the check question grounded in the dialogue the student experienced
    post_convs = []
    for b, d in zip(bench, dialogs):
        transcript = "\n".join(
            f"{'Tutor' if m['role']=='user' else 'Me'}: {m['content']}"
            for m in d if m["role"] in ("user", "assistant"))
        post_convs.append([{"role": "user", "content":
                            f"After this tutoring conversation:\n{transcript}\n\n"
                            f"Now answer: {b['check_question']}"}])
    post = _chat_gen(student, stok, post_convs, max_new_tokens=120, temperature=0.0)

    cold_s = sum(grade(c, b["check_answer"]) for c, b in zip(cold, bench)) / max(1, len(bench))
    post_s = sum(grade(p, b["check_answer"]) for p, b in zip(post, bench)) / max(1, len(bench))
    return {"n": len(bench), "turns": turns,
            "cold_score": round(cold_s, 4), "post_tutoring_score": round(post_s, 4),
            "learning_gain": round(post_s - cold_s, 4)}


def _results_dir():
    import os
    d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(d, exist_ok=True)
    return d


@app.local_entrypoint()
def run_eval(run_id: str, model: str = "qwen0.5b", adapter_dir: str = "",
             method: str = "base", seed: int = 0, n: int = 150,
             tasks: str = "arc_easy,arc_challenge,truthfulqa_mc1",
             limit: int = 120, do_lg: bool = True, lg_n: int = 30,
             system_variant: str = "neutral"):
    """Full eval for one config; writes results/eval_<run_id>.json locally + prints agg.
    adapter_dir='' evaluates the untuned base ('before'); system_variant='socratic'
    is the prompting-only ablation."""
    import os
    gv = gen_and_verify.remote(model_key=model, adapter_dir=adapter_dir, n=n,
                               system_variant=system_variant)
    judged = Judge().score.remote([{"id": r["id"],
                                    "student_question": r["student_question"],
                                    "reply": r["reply"]} for r in gv["rows"]])
    std = eval_standard.remote(model_key=model, adapter_dir=adapter_dir,
                               tasks=tasks, limit=limit)
    lg = (learning_gain.remote(model_key=model, adapter_dir=adapter_dir, n=lg_n)
          if do_lg else {})

    result = {"run_id": run_id, "model": model, "method": method, "seed": seed,
              "adapter_dir": adapter_dir,
              "verifiable": gv["agg"], "judge": judged["agg"],
              "standard": std["standard"], "learning_gain": lg,
              "replies": gv["rows"]}
    path = os.path.join(_results_dir(), f"eval_{run_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    summary = {"run_id": run_id, "model": model, "method": method,
               **gv["agg"], **judged["agg"], "standard": std["standard"],
               "learning_gain": lg.get("learning_gain"),
               "cold_score": lg.get("cold_score"),
               "post_tutoring_score": lg.get("post_tutoring_score")}
    print("\n=== EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


@app.local_entrypoint()
def main(model: str = "qwen0.5b", adapter_dir: str = "", n: int = 40,
         tasks: str = "arc_easy", limit: int = 100):
    gv = gen_and_verify.remote(model_key=model, adapter_dir=adapter_dir, n=n)
    judged = Judge().score.remote([{"id": r["id"],
                                    "student_question": r["student_question"],
                                    "reply": r["reply"]} for r in gv["rows"]])
    std = eval_standard.remote(model_key=model, adapter_dir=adapter_dir,
                               tasks=tasks, limit=limit)
    print("\n=== VERIFIABLE ==="); print(json.dumps(gv["agg"], indent=2))
    print("\n=== JUDGE ==="); print(json.dumps(judged["agg"], indent=2))
    print("\n=== STANDARD (alignment tax) ==="); print(json.dumps(std["standard"], indent=2))
    print("\n=== SAMPLE REPLY ==="); print(json.dumps(gv["rows"][0], indent=2)[:1200])
