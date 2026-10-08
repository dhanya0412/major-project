import json
from collections import defaultdict

def is_correct(r):
    return r["model_prediction"] == str(r["gt_answer"])

def compute_metrics(rs):
    aAcc = sum(is_correct(r) for r in rs) / len(rs) * 100

    # qAcc: the same question across all its image variants (incl. the no-image version for VS)
    pairs = defaultdict(list)
    for r in rs:
        key = (r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"]))
        pairs[key].append(is_correct(r))
    qAcc = sum(all(v) for v in pairs.values()) / len(pairs) * 100

    # fAcc: all questions on one specific image; skip VS figure_id 0 like the official code
    figs = defaultdict(list)
    for r in rs:
        if r["category"] == "VS" and str(r["figure_id"]) == "0":
            continue
        key = (r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"]))
        figs[key].append(is_correct(r))
    fAcc = sum(all(v) for v in figs.values()) / len(figs) * 100

    return {"aAcc": round(aAcc, 2), "qAcc": round(qAcc, 2), "fAcc": round(fAcc, 2),
            "n_questions": len(rs), "n_pairs": len(pairs), "n_figures": len(figs)}

files = {
    "LLaVA-1.5-7B":   "results/llava_baseline.json",
    "Qwen2.5-VL-7B":  "results/qwen2.5vl_results.json",
    "InternVL3-8B":   "results/internvl3_8b_results.json",
    "Phi-3.5-vision": "results/phi35vision_results.json",
}
for name, path in files.items():
    with open(path) as f:
        print(name, compute_metrics(json.load(f)))

def full_summary(name, rs):
    by_cat, by_sub = defaultdict(list), defaultdict(list)
    for r in rs:
        by_cat[r["category"]].append(r)
        by_sub[(r["category"], r["subcategory"])].append(r)
    yes_rate = sum(r["model_prediction"] == "1" for r in rs) / len(rs) * 100
    return {
        "model": name,
        "overall": compute_metrics(rs),
        "yes_rate": round(yes_rate, 2),
        "by_category": {c: compute_metrics(v) for c, v in by_cat.items()},
        "by_category_subcategory": {f"{c}_{s}": compute_metrics(v) for (c, s), v in by_sub.items()},
    }

for name, path in files.items():
    with open(path) as f:
        rs = json.load(f)
    out = full_summary(name, rs)
    print(name, "yes-rate:", out["yes_rate"])
    with open(path.replace(".json", "_metrics_v2.json"), "w") as f:
        json.dump(out, f, indent=2)