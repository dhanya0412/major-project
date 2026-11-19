# Extra analyses for the Phase 3 paper. No GPU needed. Uses the same 12 result files as phase3_compare.py.
#   A. Image-insensitivity of control pairs (incl. the video / reversed-order pairs), on the FULL Phase 2 files
#   B. Self-consistency vote agreement: are the remaining errors stable or unstable?
#   C. Format-robust re-score of CoT / CoVe / SC (unparseable answers handled two ways)
#   D. A stratified sample of CoT reasoning traces to label by hand (CSV), plus a helper to summarise the labels
import csv, json, os, random, collections

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") 
SEED = 0
TRACES_PER_TYPE = 5   # per model: this many 'fixed', 'broken', 'both wrong' traces (plus up to 3 unparseable)

MODELS = {
    "Qwen2.5-VL-7B":  dict(zs="qwen2.5vl_results.json",    tag="qwen25vl"),
    "InternVL3-8B":   dict(zs="internvl3_8b_results.json", tag="internvl3"),
    "Phi-3.5-vision": dict(zs="phi35vision_results.json",  tag="phi35"),
}
S = lambda x: str(x).strip()
valid = lambda p: S(p) in ("0", "1")
rowkey = lambda r: "|".join([S(r["split"]), S(r["category"]), S(r["subcategory"]), S(r["set_id"]), S(r["figure_id"]), S(r["question_id"])])
is_ok = lambda r: S(r["model_prediction"]) == S(r["gt_answer"])
pct = lambda a, b: 100 * a / b if b else float("nan")


def load(name):
    path = os.path.join(DIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


def metrics(rows):                                  # official grouping, same as phase3_compare.py
    q, f = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        q[(r["category"], r["subcategory"], r["set_id"], r["question_id"])].append(is_ok(r))
        if not (r["category"] == "VS" and S(r["figure_id"]) == "0"):
            f[(r["category"], r["subcategory"], r["set_id"], r["figure_id"])].append(is_ok(r))
    P = lambda xs: 100 * sum(xs) / len(xs) if xs else float("nan")
    return dict(aAcc=P([is_ok(r) for r in rows]), qAcc=P([all(v) for v in q.values()]), fAcc=P([all(v) for v in f.values()]), n=len(rows))


data = {}
for m, cfg in MODELS.items():
    zs_all = load(cfg["zs"])
    if zs_all is None:
        print(f"[skip] {m}: missing {cfg['zs']}"); continue
    d = {"zs_all": zs_all}
    for s in ("cot", "cove", "sc"):
        rows = load(f"{cfg['tag']}_{s}.json")
        if rows is not None:
            d[s] = {r["key"]: r for r in rows}
    zs = {rowkey(r): r for r in zs_all}
    d["zs"] = {k: zs[k] for k in set.intersection(*[set(v) for s, v in d.items() if s in ("cot", "cove", "sc")]) & set(zs)} if any(s in d for s in ("cot", "cove", "sc")) else {}
    data[m] = d

# ====================================================================== A. control-pair image insensitivity
print("\n=== A. Control pairs: does the answer change when the image changes? (full Phase 2 files, zero-shot) ===")
print("Only pairs whose two true answers differ are used. 'same' = the model gave the same answer to both images (ignored the change);")
print("'expected' = same-answer rate if the answer were independent of the image, from the model's own Yes share.")
A = []
for m, d in data.items():
    pairs = collections.defaultdict(list)
    for r in d["zs_all"]:
        if S(r["split"]) == "image":
            pairs[(S(r["category"]), S(r["subcategory"]), S(r["set_id"]), S(r["question_id"]))].append(r)
    groups = collections.defaultdict(list)
    for k, rs in pairs.items():
        if len(rs) == 2 and valid(rs[0]["gt_answer"]) and valid(rs[1]["gt_answer"]) and S(rs[0]["gt_answer"]) != S(rs[1]["gt_answer"]):
            groups[f"{k[0]}_{k[1]}"].append(rs)
            groups["ALL"].append(rs)
            groups[f"all_{k[0]}"].append(rs)
    for g in sorted(groups):
        ps = [p for p in groups[g] if valid(p[0]["model_prediction"]) and valid(p[1]["model_prediction"])]
        if not ps:
            continue
        same = sum(S(p[0]["model_prediction"]) == S(p[1]["model_prediction"]) for p in ps)
        both_ok = sum(is_ok(p[0]) and is_ok(p[1]) for p in ps)
        yes = sum(S(r["model_prediction"]) == "1" for p in ps for r in p) / (2 * len(ps))
        A.append(dict(model=m, group=g, pairs=len(ps), same_pct=pct(same, len(ps)), expected_same_pct=100 * (yes ** 2 + (1 - yes) ** 2),
                      both_correct_pct=pct(both_ok, len(ps)), swapped_pct=pct(len(ps) - same - both_ok, len(ps)),
                      skipped=len(groups[g]) - len(ps)))
print(f"{'model':<15}{'group':<14}{'pairs':>6}{'same %':>9}{'expected %':>12}{'both right %':>14}{'swapped %':>11}")
for a in A:
    print(f"{a['model']:<15}{a['group']:<14}{a['pairs']:>6}{a['same_pct']:>9.1f}{a['expected_same_pct']:>12.1f}{a['both_correct_pct']:>14.1f}{a['swapped_pct']:>11.1f}")
with open(os.path.join(DIR, "extra_A_pair_insensitivity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(A[0])) if A else None
    if w: w.writeheader(); w.writerows(A)

# ====================================================================== B. self-consistency vote agreement
print("\n=== B. Self-consistency: accuracy by vote agreement (sample questions) ===")
print("agreement = size of the largest group of Yes/No votes among the K samples; 'unanimous' = all K agree.")
B = []
for m, d in data.items():
    if "sc" not in d or not d["zs"]:
        print(f"[skip] {m}: needs the SC file and the zero-shot file"); continue
    rows = []
    for k, r in d["sc"].items():
        votes = r.get("votes") or []
        vs = [v for v in votes if v in ("yes", "no")]
        lvl = max(collections.Counter(vs).values()) if vs else 0
        grp = "unanimous" if votes and lvl == len(votes) else "split" if vs else "no valid vote"
        rows.append((k, lvl, grp, is_ok(d["zs"][k]), is_ok(d["cot"][k]) if "cot" in d and k in d["cot"] else None, is_ok(r)))
    for grp in ("unanimous", "split", "no valid vote"):
        sub = [x for x in rows if x[2] == grp]
        if not sub: continue
        cot_vals = [x[4] for x in sub if x[4] is not None]
        B.append(dict(model=m, group=grp, n=len(sub), share_pct=pct(len(sub), len(rows)), zero_shot_acc=pct(sum(x[3] for x in sub), len(sub)),
                      cot_acc=pct(sum(cot_vals), len(cot_vals)) if cot_vals else float("nan"), sc_acc=pct(sum(x[5] for x in sub), len(sub))))
    errs = [x for x in rows if not x[5]]
    print(f"{m}: of SC's {len(errs)} errors, {sum(x[2]=='unanimous' for x in errs)} were on unanimous questions "
          f"({pct(sum(x[2]=='unanimous' for x in errs), len(errs)):.0f}%): errors the model makes every time.")
print(f"\n{'model':<15}{'votes':<14}{'n':>5}{'share %':>9}{'zero-shot':>11}{'CoT':>8}{'SC':>8}")
for b in B:
    print(f"{b['model']:<15}{b['group']:<14}{b['n']:>5}{b['share_pct']:>9.1f}{b['zero_shot_acc']:>11.1f}{b['cot_acc']:>8.1f}{b['sc_acc']:>8.1f}")
with open(os.path.join(DIR, "extra_B_sc_agreement.csv"), "w", newline="") as f:
    if B:
        w = csv.DictWriter(f, fieldnames=list(B[0])); w.writeheader(); w.writerows(B)

# ====================================================================== C. format-robust re-score
print("\n=== C. Unparseable answers: two robust re-scores (sample questions) ===")
print("'parseable-only': aAcc of the strategy vs zero-shot on the SAME questions the strategy answered in a parseable way.")
print("'fallback': unparseable answers replaced by the model's own zero-shot answer, then the full metrics.")
C = []
for m, d in data.items():
    if not d["zs"]:
        continue
    base = metrics([d["zs"][k] for k in sorted(d["zs"])])
    C.append(dict(model=m, strategy="zero-shot", n=base["n"], unparseable=sum(not valid(r["model_prediction"]) for r in d["zs"].values()),
                  aAcc_parseable_only=float("nan"), zs_aAcc_same_questions=float("nan"), aAcc_fallback=base["aAcc"], qAcc_fallback=base["qAcc"], fAcc_fallback=base["fAcc"]))
    for s in ("cot", "cove", "sc"):
        if s not in d: continue
        keys = sorted(d["zs"])
        rows = [d[s][k] for k in keys]
        ok_keys = [k for k in keys if valid(d[s][k]["model_prediction"])]
        imputed = [dict(d[s][k], model_prediction=(d["zs"][k]["model_prediction"] if not valid(d[s][k]["model_prediction"]) else d[s][k]["model_prediction"])) for k in keys]
        mi = metrics(imputed)
        C.append(dict(model=m, strategy=s, n=len(rows), unparseable=len(keys) - len(ok_keys),
                      aAcc_parseable_only=pct(sum(is_ok(d[s][k]) for k in ok_keys), len(ok_keys)),
                      zs_aAcc_same_questions=pct(sum(is_ok(d["zs"][k]) for k in ok_keys), len(ok_keys)),
                      aAcc_fallback=mi["aAcc"], qAcc_fallback=mi["qAcc"], fAcc_fallback=mi["fAcc"]))
print(f"{'model':<15}{'strategy':<10}{'unparse':>8}{'aAcc parseable':>16}{'ZS same qs':>12}{'aAcc fb':>9}{'qAcc fb':>9}{'fAcc fb':>9}")
for c in C:
    print(f"{c['model']:<15}{c['strategy']:<10}{c['unparseable']:>8}{c['aAcc_parseable_only']:>16.1f}{c['zs_aAcc_same_questions']:>12.1f}"
          f"{c['aAcc_fallback']:>9.1f}{c['qAcc_fallback']:>9.1f}{c['fAcc_fallback']:>9.1f}")
with open(os.path.join(DIR, "extra_C_format_robust.csv"), "w", newline="") as f:
    if C:
        w = csv.DictWriter(f, fieldnames=list(C[0])); w.writeheader(); w.writerows(C)

# ====================================================================== D. CoT traces to label by hand
print("\n=== D. Traces to label ===")
rg = random.Random(SEED)
T = []
for m, d in data.items():
    if "cot" not in d or not d["zs"]:
        continue
    pools = collections.defaultdict(list)
    for k, r in d["cot"].items():
        if k not in d["zs"]: continue
        z = d["zs"][k]
        if not valid(r["model_prediction"]): t = "unparseable"
        elif is_ok(r) and not is_ok(z): t = "fixed"          # wrong in zero-shot, right with CoT
        elif not is_ok(r) and is_ok(z): t = "broken"         # right in zero-shot, wrong with CoT
        elif not is_ok(r) and not is_ok(z): t = "both_wrong"
        else: continue
        pools[t].append(k)
    for t, n in (("fixed", TRACES_PER_TYPE), ("broken", TRACES_PER_TYPE), ("both_wrong", TRACES_PER_TYPE), ("unparseable", 3)):
        for k in rg.sample(sorted(pools[t]), min(n, len(pools[t]))):
            r, z = d["cot"][k], d["zs"][k]
            ans = lambda p: {"1": "Yes", "0": "No"}.get(S(p), "unparseable")
            T.append(dict(model=m, flip_type=t, subcategory=f"{r['category']}_{r['subcategory']}", question=S(r["question"]),
                          true_answer=ans(r["gt_answer"]), zero_shot_answer=ans(z["model_prediction"]), cot_answer=ans(r["model_prediction"]),
                          reasoning=S(r["raw_response"]), key=k, label="", note=""))
with open(os.path.join(DIR, "traces_to_label.csv"), "w", newline="", encoding="utf-8-sig") as f:
    if T:
        w = csv.DictWriter(f, fieldnames=list(T[0])); w.writeheader(); w.writerows(T)
print(f"wrote traces_to_label.csv with {len(T)} traces. Fill the 'label' column using the codes below, then run summarize_labels().")
print("  C = reasoning and answer both correct for the right reasons      M = misread the image (the description is wrong)")
print("  P = accepted the claim in the question without checking it       I = reasoning and final answer disagree")
print("  F = cut off or bad format                                       O = other (explain in 'note')")


def export_trace_images(csv_path="traces_to_label.csv", out_dir="trace_images"):
    """Optional: save the image of each trace (needs the HuggingFace dataset and internet) so 'M' can be judged."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/HallusionBench")
    os.makedirs(out_dir, exist_ok=True)
    idx = {}
    for sp in ("image", "non_image"):
        d = ds[sp]
        light = d.remove_columns([c for c in ["image"] if c in d.column_names])
        for i, r in enumerate(light):
            idx["|".join([sp, S(r.get("category", "")), S(r.get("subcategory", "")), S(r.get("set_id", "")), S(r.get("figure_id", "")), S(r.get("question_id", ""))])] = (sp, i)
    n = 0
    for i, row in enumerate(csv.DictReader(open(csv_path, encoding="utf-8-sig"))):
        sp_i = idx.get(row["key"])
        if sp_i and "image" in ds[sp_i[0]].column_names:
            img = ds[sp_i[0]][sp_i[1]]["image"]
            if img is not None:
                img.convert("RGB").save(os.path.join(out_dir, f"{i+2:03d}_{row['model'].split('-')[0]}_{row['flip_type']}.png")); n += 1
    print(f"saved {n} images to {out_dir}/ (file name starts with the CSV row number)")


def summarize_labels(csv_path="traces_to_label.csv"):
    """After labelling: counts of each label overall and by flip type, per model."""
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")) if r["label"].strip()]
    print(f"{len(rows)} labelled traces")
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m]
        print(f"\n{m}: {dict(collections.Counter(r['label'].strip().upper() for r in sub))}")
        for t in ("fixed", "broken", "both_wrong", "unparseable"):
            c = collections.Counter(r["label"].strip().upper() for r in sub if r["flip_type"] == t)
            if c: print(f"   {t:<12} {dict(c)}")