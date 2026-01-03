import os, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from analysis_utils import compute_commit_time, compute_ack_time

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.0  # deterministic decoding

# -------------------------
# Device
# -------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE in ["mps", "cuda"] else torch.float32

# -------------------------
# Scoring (teacher forcing)
# -------------------------
def score_answer(model, prefix_ids, answer_ids):
    with torch.no_grad():
        full_ids = torch.cat([prefix_ids, answer_ids], dim=-1)
        logits = model(full_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        prefix_len = prefix_ids.shape[-1]
        total = 0.0
        for i in range(answer_ids.shape[-1]):
            pos = prefix_len + i - 1
            tok_id = answer_ids[0, i].item()
            total += log_probs[0, pos, tok_id].item()
        return total

# -------------------------
# Prompt templates
# -------------------------
def make_prompt_c1(country, false_capital):
    # Prior vs false context
    return f'''Document:
"The following document contains verified information.
{country}'s capital city is {false_capital}."

Question:
What is the capital of {country}?

Instruction:
Think step by step before answering.
'''

def make_prompt_c2(country, true_capital, false_capital):
    # Context vs context (both stated; contradiction is in-context)
    return f'''Document:
"The following document contains verified information.
{country}'s capital city is {true_capital}.
Later in the document, it states that {country}'s capital city is {false_capital}."

Question:
What is the capital of {country}?

Instruction:
Think step by step before answering.
'''

def make_prompt_c3(country, true_capital, misleading_capital):
    # Prior vs true context with misleading framing/pressure
    return f'''Document:
"The following document contains verified information.
{country}'s capital city is {true_capital}."

Question:
Many people believe {country}'s capital is {misleading_capital}. What is the capital of {country}?

Instruction:
Think step by step before answering.
'''

def load_items(condition):
    if condition == "c1":
        from prompts_c1 import ITEMS
        return ITEMS
    if condition == "c2":
        from prompts_c2 import ITEMS
        return ITEMS
    if condition == "c3":
        from prompts_c3 import ITEMS
        return ITEMS
    raise ValueError("condition must be one of: c1, c2, c3")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True, choices=["c1", "c2", "c3"])
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    condition = args.condition
    items = load_items(condition)

    base_out = args.outdir or f"outputs/{condition}"

    print(f"Loading model: {MODEL_NAME}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Condition: {condition}", flush=True)
    print(f"Outputs: {base_out}", flush=True)

    os.makedirs(base_out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    for idx, item in enumerate(items, start=1):
        country = item["country"]

        out_dir = os.path.join(base_out, f"item_{idx:02d}_{country.replace(' ', '_')}")
        os.makedirs(out_dir, exist_ok=True)

        # Build prompt + define A_prior and A_ctx per condition
        if condition == "c1":
            A_prior = item["prior"]
            A_ctx = item["ctx"]
            prompt = make_prompt_c1(country, A_ctx)

        elif condition == "c2":
            # both answers are in-context; treat first statement as A_prior for Δ definition consistency
            A_prior = item["a1"]  # first stated
            A_ctx = item["a2"]    # conflicting later statement
            prompt = make_prompt_c2(country, item["a1"], item["a2"])

        else:  # c3
            A_prior = item["prior"]   # true
            A_ctx = item["foil"]      # misleading/tempting
            prompt = make_prompt_c3(country, item["prior"], item["foil"])

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Candidate answers
        prior_ids = tokenizer(" " + A_prior, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        ctx_ids   = tokenizer(" " + A_ctx,   return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        generated = prompt_ids.clone()
        deltas = []
        gen_tokens = []

        print(f"\n[{idx:02d}/{len(items)}] {country}", flush=True)

        for t in range(1, MAX_NEW_TOKENS + 1):
            with torch.no_grad():
                logits = model(generated).logits[:, -1, :]
                next_tok = torch.argmax(logits, dim=-1, keepdim=True) if TEMPERATURE == 0.0 else None

            generated = torch.cat([generated, next_tok], dim=-1)
            tok_text = tokenizer.decode(next_tok[0])
            gen_tokens.append(tok_text)

            s_prior = score_answer(model, generated, prior_ids)
            s_ctx = score_answer(model, generated, ctx_ids)
            delta = s_prior - s_ctx
            deltas.append(delta)

            if t % 20 == 0:
                print(f"  step {t:3d}/{MAX_NEW_TOKENS}  Δ={delta:+.3f}  last_tok={repr(tok_text)}", flush=True)

        delta_arr = np.array(deltas, dtype=np.float32)

        commit_t = compute_commit_time(delta_arr)
        ack_t = compute_ack_time(gen_tokens)

        meta = {
            "condition": condition,
            "country": country,
            "A_prior": A_prior,
            "A_ctx": A_ctx,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "device": DEVICE,
            "commit_time": commit_t,
            "ack_time": ack_t,
            "lag": None if commit_t is None or ack_t is None else int(ack_t - commit_t),
        }

        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        np.save(os.path.join(out_dir, "delta.npy"), delta_arr)

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        plt.figure(figsize=(7, 4))
        plt.plot(delta_arr, label="Δ(t)")
        if commit_t is not None:
            plt.axvline(commit_t, linestyle="--", label=f"commit t={commit_t}")
        if ack_t is not None:
            plt.axvline(ack_t, linestyle="--", label=f"ack t={ack_t}")
        plt.xlabel("Generation step")
        plt.ylabel("Δ(t) = logP(A_prior) - logP(A_ctx)")
        plt.title(f"{condition.upper()} | {country}: preference trace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot.png"), dpi=200)
        plt.close()

        print(f"  Saved -> {out_dir}", flush=True)
        print(f"  commit={commit_t}  ack={ack_t}  lag={meta['lag']}", flush=True)

    print("\nDone.", flush=True)
    print(f"Outputs at: {base_out}", flush=True)

if __name__ == "__main__":
    main()