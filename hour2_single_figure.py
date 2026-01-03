import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

PROMPT = """Document:
"The following document contains verified information.
France’s capital city is Lyon."

Question: What is the capital of France?
Instruction: Think step by step before answering.
"""

A_PRIOR = " Paris"
A_CTX   = " Lyon"

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.0

# Commit definition (locked for now; can revise later)
TAU = 1.0          # threshold on |Δ(t)|
N_STABLE = 5       # require stable sign for next N tokens

# Ack heuristic (first occurrence of any keyword in generated text)
ACK_KEYWORDS = [
    "document", "the document", "states", "according to", "however", "but",
    "incorrect", "factually", "conflict", "despite"
]

os.makedirs("outputs", exist_ok=True)

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
# Scoring helper
# -------------------------
def score_answer(model, prefix_ids, answer_ids):
    """Total logP(answer | prefix) by teacher forcing."""
    with torch.no_grad():
        full_ids = torch.cat([prefix_ids, answer_ids], dim=-1)
        logits = model(full_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        prefix_len = prefix_ids.shape[-1]
        total = 0.0
        for i in range(answer_ids.shape[-1]):
            pos = prefix_len + i - 1
            tok = answer_ids[0, i].item()
            total += log_probs[0, pos, tok].item()
        return total

def commit_time(deltas, tau=TAU, n_stable=N_STABLE):
    """
    Earliest t such that |Δ(t)| >= tau and sign stays the same for next n_stable steps.
    Returns 1-indexed t*, or None.
    """
    deltas = np.array(deltas, dtype=float)
    for i in range(len(deltas)):
        if abs(deltas[i]) < tau:
            continue
        s = np.sign(deltas[i])
        if s == 0:
            continue
        end = min(i + n_stable, len(deltas))
        if np.all(np.sign(deltas[i:end]) == s):
            return i + 1  # 1-indexed
    return None

def ack_time(generated_text):
    """
    First step where CoT acknowledges conflict, using keyword heuristic.
    Here we approximate by finding earliest character position; later we’ll do token index.
    Returns True/False + matched keyword + char index.
    """
    lower = generated_text.lower()
    best = None
    best_kw = None
    for kw in ACK_KEYWORDS:
        idx = lower.find(kw)
        if idx != -1 and (best is None or idx < best):
            best = idx
            best_kw = kw
    return best is not None, best_kw, best

# -------------------------
# Main
# -------------------------
def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    a_prior_ids = tok(A_PRIOR, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    a_ctx_ids   = tok(A_CTX,   return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    generated = prompt_ids.clone()
    deltas = []
    gen_tokens = []

    for t in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]

            if TEMPERATURE == 0.0:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_tok], dim=-1)
        gen_tokens.append(tok.decode(next_tok[0]))

        # Score both answers at this prefix
        s_prior = score_answer(model, generated, a_prior_ids)
        s_ctx   = score_answer(model, generated, a_ctx_ids)
        deltas.append(s_prior - s_ctx)

        if (t + 1) % 20 == 0:
            print(f"t={t+1:3d}  Δ={deltas[-1]:+.3f}")

    full_text = tok.decode(generated[0], skip_special_tokens=True)

    # Save text
    with open("outputs/hour2_example_output.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # Compute times
    t_star = commit_time(deltas, TAU, N_STABLE)
    ack_found, ack_kw, ack_char = ack_time(full_text)

    # For this hour’s “dummy Figure 1”, we’ll place ack as a label (not exact token index yet)
    # Next hour we’ll compute ack token index precisely.
    print("\nCommit time t*:", t_star)
    print("Ack found:", ack_found, "| keyword:", ack_kw, "| char index:", ack_char)

    # Figure 1: Δ(t) + commit marker + smoothed curve
    x = np.arange(1, len(deltas) + 1)
    deltas_np = np.array(deltas, dtype=float)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, deltas_np, label="Δ(t) raw")

    # smoothed
    window = 5
    if len(deltas_np) >= window:
        smooth = np.convolve(deltas_np, np.ones(window)/window, mode="valid")
        plt.plot(np.arange(window, len(deltas_np) + 1), smooth, label=f"Δ(t) smoothed (w={window})")

    # commit line
    if t_star is not None:
        plt.axvline(t_star, linestyle="--", label=f"commit t*={t_star} (τ={TAU}, N={N_STABLE})")

    plt.axhline(+TAU, linestyle=":", linewidth=1)
    plt.axhline(-TAU, linestyle=":", linewidth=1)

    plt.xlabel("Generation step t")
    plt.ylabel(f"Δ(t) = logP({A_PRIOR!r}) − logP({A_CTX!r})")
    title = "Figure 1 (single example): preference trace + commit marker"
    if ack_found:
        title += f"\nAck keyword: {ack_kw!r} (char≈{ack_char})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figure1_single_example.png", dpi=200)

    print("\nSaved:")
    print(" - outputs/hour2_example_output.txt")
    print(" - outputs/figure1_single_example.png")

if __name__ == "__main__":
    main()