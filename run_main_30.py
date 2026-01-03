import os, json
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import ITEMS
from analysis_utils import compute_commit_time, compute_ack_time

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.0

BASE_OUT = "outputs/main_30"

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
    """
    Compute log P(answer | prefix) by teacher forcing.
    prefix_ids: [1, T]
    answer_ids: [1, K]
    returns float
    """
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
# Prompt template
# -------------------------
def make_prompt(country, false_capital):
    return f'''Document:
"The following document contains verified information.
{country}'s capital city is {false_capital}."

Question:
What is the capital of {country}?

Instruction:
Think step by step before answering.
'''


# -------------------------
# Main
# -------------------------
def main():
    print(f"Loading model: {MODEL_NAME}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    os.makedirs(BASE_OUT, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # Run all items
    for idx, item in enumerate(ITEMS, start=1):
        country = item["country"]
        prior = item["prior"]
        ctx = item["ctx"]

        out_dir = os.path.join(BASE_OUT, f"item_{idx:02d}_{country.replace(' ', '_')}")
        os.makedirs(out_dir, exist_ok=True)

        prompt = make_prompt(country, ctx)

        # tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # tokenize candidate answers (prepend space for consistency)
        prior_ids = tokenizer(" " + prior, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        ctx_ids = tokenizer(" " + ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        generated = prompt_ids.clone()
        deltas = []
        gen_tokens = []

        print(f"\n[{idx:02d}/30] {country}", flush=True)

        for t in range(1, MAX_NEW_TOKENS + 1):
            with torch.no_grad():
                logits = model(generated).logits[:, -1, :]

                if TEMPERATURE == 0.0:
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)

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

        # compute times
        commit_t = compute_commit_time(delta_arr)
        ack_t = compute_ack_time(gen_tokens)

        meta = {
            "country": country,
            "A_prior": prior,
            "A_ctx": ctx,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "device": DEVICE,
            "commit_time": commit_t,
            "ack_time": ack_t,
            "lag": None if commit_t is None or ack_t is None else int(ack_t - commit_t),
        }

        # save full output text
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        # save delta
        np.save(os.path.join(out_dir, "delta.npy"), delta_arr)

        # save metadata
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # plot with markers
        plt.figure(figsize=(7, 4))
        plt.plot(delta_arr, label="Δ(t)")

        if commit_t is not None:
            plt.axvline(commit_t, linestyle="--", label=f"commit t={commit_t}")
        if ack_t is not None:
            plt.axvline(ack_t, linestyle="--", label=f"ack t={ack_t}")

        plt.xlabel("Generation step")
        plt.ylabel("Δ(t) = logP(A_prior) - logP(A_ctx)")
        plt.title(f"{country}: preference trace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot.png"), dpi=200)
        plt.close()

        print(f"  Saved -> {out_dir}", flush=True)
        print(f"  commit={commit_t}  ack={ack_t}  lag={meta['lag']}", flush=True)

    print("\nDone. Collected 30 items.", flush=True)
    print(f"Outputs at: {BASE_OUT}", flush=True)


if __name__ == "__main__":
    main()
