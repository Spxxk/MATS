import numpy as np

TAU = 2.0
N_STABLE = 8

ACK_KEYWORDS = [
    "but",
    "however",
    "although",
    "the document",
    "this statement",
    "incorrect",
    "despite"
]

def compute_commit_time(delta):
    """
    delta: list or np.array of Δ(t)
    """
    T = len(delta)
    for t in range(T - N_STABLE):
        if abs(delta[t]) >= TAU:
            sign = np.sign(delta[t])
            if all(np.sign(delta[t + i]) == sign for i in range(1, N_STABLE)):
                return t
    return None

def compute_ack_time(tokens):
    """
    tokens: list of decoded token strings
    """
    for i, tok in enumerate(tokens):
        lower = tok.lower()
        for kw in ACK_KEYWORDS:
            if kw in lower:
                return i
    return None
