import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from analysis_utils import compute_commit_time, compute_ack_time
from prompts_b1 import ITEMS as B1_ITEMS
from prompts_b3 import ITEMS as B3_ITEMS

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.0

OUT_B1 = "outputs/b1_15"
OUT_B3 = "outputs/b3_15"

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
# Prompts
# -------------------------
def prompt_b1(country):
    return f"""Question:
What is the capital of {country}?

Instruction:
Think step by step before answering.
"""

def prompt_b3(country, false_capital):
    return f"""Question:
What is the capital of {country}?

Document:
"The following document contains verified information.
{country}'s capital city is {false_capital}."

Instruction:
Think step by step before answering.
"""


def run_condition(items, out_base, condition_name):
    os.makedirs(out_base, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    print(f"\nRunning {condition_name} on {len(items)} items", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    for idx, item in enumerate(items, start=1):
        country = item["country"]
        out_dir = os.path.join(out_base, f"item_{idx:02d}_{country.replace(' ', '_')}")
        os.makedirs(out_dir, exist_ok=True)

        if condition_name == "B1":
            prompt = prompt_b1(country)
            A_prior = item["prior"]
            A_other = item["foil"]
        elif condition_name == "B3":
            prompt = prompt_b3(country, item["ctx"])
            A_prior = item["prior"]
            A_other = item["ctx"]
        else:
            raise ValueError("unknown condition")

        # tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # answers (prepend space for consistent tokenization)
        prior_ids = tokenizer(" " + A_prior, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        other_ids = tokenizer(" " + A_other, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        generated = prompt_ids.clone()
        deltas = []
        gen_tokens = []

        print(f"[{condition_name}] {idx:02d}/{len(items)} {country}", flush=True)

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
            s_other = score_answer(model, generated, other_ids)
            delta = s_prior - s_other
            deltas.append(delta)

        delta_arr = np.array(deltas, dtype=np.float32)

        commit_t = compute_commit_time(delta_arr)
        ack_t = compute_ack_time(gen_tokens)

        meta = {
            "country": country,
            "A_prior": A_prior,
            "A_other": A_other,
            "condition": condition_name,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "device": DEVICE,
            "commit_time": commit_t,
            "ack_time": ack_t,
            "lag": None if commit_t is None or ack_t is None else int(ack_t - commit_t),
        }

        # save output text
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        np.save(os.path.join(out_dir, "delta.npy"), delta_arr)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # plot
        plt.figure(figsize=(7, 4))
        plt.plot(delta_arr, label="Δ(t)")
        if commit_t is not None:
            plt.axvline(commit_t, linestyle="--", label=f"commit t={commit_t}")
        if ack_t is not None:
            plt.axvline(ack_t, linestyle="--", label=f"ack t={ack_t}")
        plt.xlabel("Generation step")
        plt.ylabel("Δ(t) = logP(A_prior) - logP(A_other)")
        plt.title(f"{condition_name} {country}: preference trace")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot.png"), dpi=200)
        plt.close()

    print(f"Saved outputs to: {out_base}", flush=True)


def main():
    run_condition(B1_ITEMS, OUT_B1, "B1")
    run_condition(B3_ITEMS, OUT_B3, "B3")


if __name__ == "__main__":
    main()
