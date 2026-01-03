import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Python exe:", sys.executable)
print("Torch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE in ["mps", "cuda"] else torch.float32

PROMPT = """Document:
"The following document contains verified information.
France’s capital city is Lyon."

Question: What is the capital of France?
Instruction: Think step by step before answering.
"""

CANDIDATE_ANSWERS = [" Paris", " Lyon"]
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.0

os.makedirs("outputs", exist_ok=True)

def score_answer(model, prefix_ids, answer_ids):
    """
    Compute log P(answer | prefix) by teacher forcing.
    """
    with torch.no_grad():
        full_ids = torch.cat([prefix_ids, answer_ids], dim=-1)
        logits = model(full_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        prefix_len = prefix_ids.shape[-1]
        total_logprob = 0.0

        for i in range(answer_ids.shape[-1]):
            pos = prefix_len + i - 1
            token_id = answer_ids[0, i].item()
            total_logprob += log_probs[0, pos, token_id].item()

        return total_logprob

def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)

    model.eval()

    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    candidate_ids = [
        tokenizer(ans, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        for ans in CANDIDATE_ANSWERS
    ]

    generated = prompt_ids.clone()
    deltas = []
    generated_tokens = []

    for t in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]

            if TEMPERATURE == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)
        token_text = tokenizer.decode(next_token[0])
        generated_tokens.append(token_text)

        score_prior = score_answer(model, generated, candidate_ids[0])  # " Paris"
        score_ctx = score_answer(model, generated, candidate_ids[1])    # " Lyon"
        delta = score_prior - score_ctx
        deltas.append(delta)

        if (t + 1) % 25 == 0:
            print(f"t={t+1:3d}  Δ={delta:+.3f}  token={repr(token_text)}")

    # Save full output text
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    with open("outputs/pilot_output.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # -------------------------
    # Figure P1: Raw Δ(t)
    # -------------------------
    plt.figure()
    plt.plot(range(1, len(deltas) + 1), deltas)
    plt.xlabel("Generation step t")
    plt.ylabel("Δ(t) = logP(' Paris') − logP(' Lyon')")
    plt.title("Pilot: answer preference over generation (raw)")
    plt.tight_layout()
    plt.savefig("outputs/pilot_delta.png", dpi=200)

    # -------------------------
    # Figure P2: Smoothed Δ(t)
    # -------------------------
    window = 5
    if len(deltas) >= window:
        smooth = np.convolve(np.array(deltas), np.ones(window) / window, mode="valid")

        plt.figure()
        plt.plot(range(window, len(deltas) + 1), smooth)
        plt.xlabel("Generation step t")
        plt.ylabel(f"Smoothed Δ(t), moving average window={window}")
        plt.title("Pilot: answer preference over generation (smoothed)")
        plt.tight_layout()
        plt.savefig("outputs/pilot_delta_smooth.png", dpi=200)
    else:
        print("Not enough points to compute smoothed Δ(t).")

    print("\nSaved:")
    print(" - outputs/pilot_output.txt")
    print(" - outputs/pilot_delta.png")
    print(" - outputs/pilot_delta_smooth.png (if generated)")

if __name__ == "__main__":
    main()
