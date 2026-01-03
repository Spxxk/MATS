import os
import json
import csv
import math
from collections import defaultdict

RUN_DIRS = {
    "C1_prior_vs_false_ctx": "outputs/main_30",
    "C2_ctx_vs_ctx": "outputs/c2_15",
    "C3_misleading_framing": "outputs/c3_15",
    "B1_no_doc": "outputs/b1_15",
    "B3_q_before_doc": "outputs/b3_15",
}

OUT_DIR = "outputs/summary"
CSV_PATH = os.path.join(OUT_DIR, "all_items.csv")
REPORT_PATH = os.path.join(OUT_DIR, "main_findings_report.txt")


def safe_int(x):
    return None if x is None else int(x)


def mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted([x for x in xs if x is not None])
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return 0.5 * (xs[mid - 1] + xs[mid])


def stdev(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def read_meta_files(run_root):
    """Return list of dict rows, one per item folder containing meta.json."""
    rows = []
    if not os.path.isdir(run_root):
        return rows

    for name in sorted(os.listdir(run_root)):
        item_dir = os.path.join(run_root, name)
        if not os.path.isdir(item_dir):
            continue
        meta_path = os.path.join(item_dir, "meta.json")
        if not os.path.isfile(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        country = meta.get("country", name)
        commit_t = safe_int(meta.get("commit_time"))
        ack_t = safe_int(meta.get("ack_time"))

        # Prefer meta["lag"] if present, else compute
        lag = meta.get("lag", None)
        lag = safe_int(lag) if lag is not None else (None if commit_t is None or ack_t is None else int(ack_t - commit_t))

        rows.append({
            "item_dir": name,
            "country": country,
            "commit_time": commit_t,
            "ack_time": ack_t,
            "lag": lag,
            "A_prior": meta.get("A_prior", ""),
            "A_ctx": meta.get("A_ctx", ""),
            "device": meta.get("device", ""),
            "temperature": meta.get("temperature", ""),
            "max_new_tokens": meta.get("max_new_tokens", ""),
        })

    return rows


def ordering_counts(rows):
    """Return counts for commit<ack, ack<commit, equal, missing."""
    c_before_a = 0
    a_before_c = 0
    equal = 0
    missing = 0

    for r in rows:
        c = r["commit_time"]
        a = r["ack_time"]
        if c is None or a is None:
            missing += 1
        elif c < a:
            c_before_a += 1
        elif a < c:
            a_before_c += 1
        else:
            equal += 1

    return {
        "commit_before_ack": c_before_a,
        "ack_before_commit": a_before_c,
        "equal": equal,
        "missing": missing,
        "N_total": len(rows),
        "N_with_both": len(rows) - missing,
    }


def format_float(x, nd=2):
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    by_cond = {}

    # Load rows for each condition
    for cond, path in RUN_DIRS.items():
        rows = read_meta_files(path)
        if rows:
            for r in rows:
                r["condition"] = cond
            by_cond[cond] = rows
            all_rows.extend(rows)

    if not all_rows:
        print("No meta.json files found. Check your RUN_DIRS paths at top of script.")
        return

    # Write CSV
    fieldnames = [
        "condition", "country", "item_dir",
        "commit_time", "ack_time", "lag",
        "A_prior", "A_ctx",
        "device", "temperature", "max_new_tokens"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Build report text
    lines = []
    lines.append("MAIN FINDINGS SUMMARY (auto-generated)\n")

    # Condition-by-condition summary
    for cond in sorted(by_cond.keys()):
        rows = by_cond[cond]
        counts = ordering_counts(rows)
        lags = [r["lag"] for r in rows if r["lag"] is not None]

        lines.append(f"== {cond} ==")
        lines.append(f"Run folder: {RUN_DIRS[cond]}")
        lines.append(f"N total: {counts['N_total']}")
        lines.append(f"N with both times: {counts['N_with_both']}")
        lines.append(f"commit < ack: {counts['commit_before_ack']}")
        lines.append(f"ack < commit: {counts['ack_before_commit']}")
        lines.append(f"equal: {counts['equal']}")
        lines.append(f"missing (commit or ack None): {counts['missing']}")
        lines.append(f"lag (ack-commit) mean: {format_float(mean(lags))}")
        lines.append(f"lag (ack-commit) median: {format_float(median(lags))}")
        lines.append(f"lag (ack-commit) stdev: {format_float(stdev(lags))}")
        lines.append("")

    # Also print a compact block for easy copy/paste
    compact = []
    compact.append("COPY/PASTE BLOCK:")
    compact.append("cond, N_total, N_with_both, commit<ack, ack<commit, equal, missing, lag_mean, lag_median")
    for cond in sorted(by_cond.keys()):
        rows = by_cond[cond]
        counts = ordering_counts(rows)
        lags = [r["lag"] for r in rows if r["lag"] is not None]
        compact.append(
            f"{cond}, {counts['N_total']}, {counts['N_with_both']}, {counts['commit_before_ack']}, "
            f"{counts['ack_before_commit']}, {counts['equal']}, {counts['missing']}, "
            f"{format_float(mean(lags))}, {format_float(median(lags))}"
        )

    lines.append("\n".join(compact))
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Terminal output
    print(f"Saved:\n - {CSV_PATH}\n - {REPORT_PATH}\n")
    print("\n".join(compact))


if __name__ == "__main__":
    main()