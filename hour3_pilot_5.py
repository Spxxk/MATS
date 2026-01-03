import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.0

TAU = 1.0
N_STABLE = 5

ACK_KEYWORDS = [
    "document", "the document", "states", "however", "but",
    "incorrect", "factually", "despite"
]

ITEMS = [
    {
        "name": "France",
        "false_fact": "France’s capital city is Lyon.",
        "question": "What is the capital of France?",
        "prior": " Paris",
        "ctx": " Lyon",
    },
    {
        "name": "Australia",
        "false_fact": "Australia’s capital city is Sydney.",
        "question": "What is the capital of Australia?",
        "prior": " Canberra",
        "ctx": " Sydney",
    },
    {
        "name": "Canada",
        "false_fact": "Canada’s capital city is Toronto.",
        "question": "What is the capital of Canada?",
        "prior": " Ottawa",
        "ctx": " Toronto",
    },
    {
        "name": "Japan",
        "false_fact": "Japan’s capital city is Osaka.",
        "question": "What is the capital of Japan?",
        "prior": " Tokyo",
        "ctx": " Osaka",
    },
    {
        "name": "Brazil",
        "false_fact": "Brazil’s capital city is Rio de Janeiro.",
        "question": "What is the capital of Brazil?",
        "prior": " Brasília",
        "ctx": " Rio de Janeiro",
    },
]

os.makedirs("outputs/hour3", exist_ok=True)

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
# Helpers
# -------------------------
def score_answer(model, prefix_ids, answer_ids):
    with torch.no_grad():
        full = torch.cat([prefix_ids, answer_ids], dim=-1)
        logits = model(full).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        pref_len = prefix_ids.shape[-1]
        total = 0.0
        for i in range(answer_ids.shape[-1]):
            pos = pref_len + i - 1
            tok = answer_ids[0, i].item()
            total += log_probs[0, pos, tok].item()
        return total

def commit_time(deltas):
    deltas = np.array(deltas)
    for i in range(len(deltas)):
        if abs(deltas[i]) < TAU:
            continue
        s = np.sign(deltas[i])
        if np.all(np.sign(deltas[i:i+N_STABLE]) == s):
            return i + 1
    return None

def ack_time(text):
    text = text.lower()
    best = None
    for kw in ACK_KEYWORDS:
        idx = text.find(kw)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best

# -------------------------
# Main
# -------------------------
def main():
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    all_results = []

    for item in ITEMS:
        prompt = f'''Document:
"The following document contains verified information.
{item["false_fact"]}"

Question:
{item["question"]}

Instruction:
Think step by step before answering.
'''

        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prior_ids = tok(item["prior"], return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        ctx_ids = tok(item["ctx"], return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

        generated = prompt_ids.clone()
        deltas = []

        for _ in range(MAX_NEW_TOKENS):
            with torch.no_grad():
                logits = model(generated).logits[:, -1, :]
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=-1)

            s_prior = score_answer(model, generated, prior_ids)
            s_ctx = score_answer(model, generated, ctx_ids)
            deltas.append(s_prior - s_ctx)

        text = tok.decode(generated[0], skip_special_tokens=True)
        t_commit = commit_time(deltas)
        t_ack = ack_time(text)

        all_results.append({
            "name": item["name"],
            "commit": t_commit,
            "ack_char": t_ack,
            "deltas": deltas,
        })

        # plot individual trace
        plt.figure()
        plt.plot(deltas)
        if t_commit is not None:
            plt.axvline(t_commit, linestyle="--")
        plt.title(f"{item['name']}: Δ(t)")
        plt.xlabel("Generation step")
        plt.ylabel("Δ(t)")
        plt.tight_layout()
        plt.savefig(f"outputs/hour3/{item['name']}_trace.png", dpi=200)
        plt.close()

    # summary plot
    plt.figure()
    for r in all_results:
        plt.plot(r["deltas"], label=r["name"])
    plt.legend()
    plt.title("Pilot (5 items): Δ(t) traces")
    plt.xlabel("Generation step")
    plt.ylabel("Δ(t)")
    plt.tight_layout()
    plt.savefig("outputs/hour3/all_traces.png", dpi=200)

    print("Saved outputs/hour3/*")

if __name__ == "__main__":
    main()