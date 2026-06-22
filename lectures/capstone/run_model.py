"""
run_model.py — turn a checkpoint into predictions.jsonl, with the FIXED eval config.

This is the bridge between a submitted model and evaluate.py. Students run it on
dev_public to self-score; the instructor runs the SAME script on the private test
set. Greedy decoding + fixed prompt => deterministic, reproducible, fair.

    python run_model.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --test data/dev_public.jsonl --out dev_preds.jsonl

    # LoRA adapter on top of the frozen base:
    python run_model.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter ./model \
        --test data/test_private.jsonl --out test_preds.jsonl

Requires: transformers, torch  (and peft if --adapter is used). Run on Colab/Modal GPU.
"""
from __future__ import annotations

import argparse
import json

# Frozen eval config — DO NOT change these between submissions.
MAX_NEW_TOKENS = 640
TEMPERATURE = 0.0          # greedy
BATCH_SIZE = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (base or full ckpt)")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir to load on the base")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from countdown import Puzzle, chat_messages

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    rows = [json.loads(l) for l in open(args.test) if l.strip()]
    prompts = [
        tok.apply_chat_template(chat_messages(Puzzle.from_dict(r)),
                                tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    out_f = open(args.out, "w")
    n_done = 0
    for i in range(0, len(rows), args.batch_size):
        batch_rows = rows[i:i + args.batch_size]
        batch_prompts = prompts[i:i + args.batch_size]
        enc = tok(batch_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,                # greedy => deterministic
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for r, completion, ntok in zip(batch_rows, texts,
                                       (new_tokens != tok.pad_token_id).sum(dim=1).tolist()):
            out_f.write(json.dumps({"id": r["id"], "completion": completion,
                                    "num_tokens": int(ntok)}) + "\n")
        n_done += len(batch_rows)
        print(f"  {n_done}/{len(rows)} generated", end="\r")
    out_f.close()
    print(f"\n  wrote {n_done} predictions -> {args.out}")


if __name__ == "__main__":
    main()
