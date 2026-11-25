# Phase 3: compare zero-shot, CoT, CoVe and self-consistency for all models.
# Needs no GPU and no dataset. Put the 12 result JSON files in DIR (or edit the file names below), then run.
import json, os, random, collections
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIR = "."          # folder holding the result files (Colab: "." after uploading; Kaggle: "/kaggle/working")
B = 1000           # bootstrap resamples for confidence intervals

MODELS = {         # display name -> zero-shot file from Phase 2, and the prefix of the Phase 3 files
    "Qwen2.5-VL-7B":  dict(zs="qwen2.5vl_results.json",    tag="qwen25vl"),
    "InternVL3-8B":   dict(zs="internvl3_8b_results.json", tag="internvl3"),
    "Phi-3.5-vision": dict(zs="phi35vision_results.json",  tag="phi35"),
}
STRATS = ["cot", "cove", "sc"]
LABEL = {"zero-shot": "Zero-shot", "cot": "CoT", "cove": "CoVe", "sc": "Self-consistency", "vote": "Vote (all 4)"}

S = lambda x: str(x).strip()
rowkey = lambda sp, m: "|".join([S(sp), S(m["category"]), S(m["subcategory"]), S(m["set_id"]), S(m["figure_id"]), S(m["question_id"])])
is_ok = lambda r: S(r["model_prediction"]) == S(r["gt_answer"])
pct = lambda xs: 100 * sum(xs) / len(xs) if len(xs) else float("nan")


def metrics(rows):
    """Official HallusionBench grouping, as in Phase 2: qAcc by (category, subcategory, set, question),
    fAcc by (category, subcategory, set, figure), skipping VS figure 0."""
    q, f = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        q[(r["category"], r["subcategory"], r["set_id"], r["question_id"])].append(is_ok(r))
        if not (r["category"] == "VS" and S(r["figure_id"]) == "0"):
            f[(r["category"], r["subcategory"], r["set_id"], r["figure_id"])].append(is_ok(r))
    return dict(aAcc=pct([is_ok(r) for r in rows]), qAcc=pct([all(v) for v in q.values()]),
                fAcc=pct([all(v) for v in f.values()]), n=len(rows))


def boot_ci(rows, B=B, seed=0):
    """95% interval from resampling whole sets (so pairs and figures stay together)."""
    rg = random.Random(seed)
    sets = collections.defaultdict(list)
    for r in rows:
        sets[(r["category"], r["subcategory"], r["set_id"])].append(r)
    ks, res = list(sets), []
    for _ in range(B):
        samp = []
        for j, k in enumerate(rg.choice(ks) for _ in ks):
            samp += [{**r, "set_id": S(r["set_id"]) + f"#{j}"} for r in sets[k]]
        res.append(metrics(samp))
    ci = lambda key: (sorted(x[key] for x in res)[int(B * 0.025)], sorted(x[key] for x in res)[int(B * 0.975) - 1])
    return {k: ci(k) for k in ("aAcc", "qAcc", "fAcc")}


def sign_p(a, b):
    """Exact two-sided sign test on the questions that changed (wrong->right = a, right->wrong = b)."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name):
    return json.load(open(os.path.join(DIR, name)))


# ------------------------------------------------------------------ load everything
data, sample_sets = {}, {}
for mname, cfg in MODELS.items():
    try:
        zs_all = load(cfg["zs"])
    except FileNotFoundError:
        print(f"[skip] {mname}: missing {cfg['zs']}")
        continue
    runs = {}
    for s in STRATS:
        try:
            runs[s] = {r["key"]: r for r in load(f"{cfg['tag']}_{s}.json")}
        except FileNotFoundError:
            print(f"[missing] {cfg['tag']}_{s}.json")
    if not runs:
        continue
    zs = {rowkey(r["split"], r): r for r in zs_all}
    keys = set.intersection(*[set(v) for v in runs.values()]) & set(zs)
    for s, v in runs.items():
        if len(v) != len(keys):
            print(f"[warning] {mname} {s}: {len(v)} rows, but only {len(keys)} common to all runs (unfinished run?)")
    keys = sorted(keys)
    data[mname] = {"zero-shot": {k: zs[k] for k in keys}, **{s: {k: v[k] for k in keys} for s, v in runs.items()}}
    sample_sets[mname] = set(keys)
    print(f"{mname}: {len(keys)} questions, strategies: {list(runs)}")

if len({frozenset(v) for v in sample_sets.values()}) > 1:
    print("[warning] the models were not run on identical samples: check SEED and FRACTION.")

# ------------------------------------------------------------------ vote across strategies (no extra GPU)
for mname, d in data.items():
    keys, voters = sorted(d["zero-shot"]), [s for s in ("zero-shot",) + tuple(STRATS) if s in d]
    vote = {}
    for k in keys:
        preds = [S(d[s][k]["model_prediction"]) for s in voters]
        c = collections.Counter(p for p in preds if p in ("0", "1"))
        top = c.most_common()
        if not top:
            ans = "-1"
        elif len(top) == 1 or top[0][1] > top[1][1]:
            ans = top[0][0]
        else:
            ans = S(d["zero-shot"][k]["model_prediction"])      # tie: fall back to zero-shot
        vote[k] = {**d["zero-shot"][k], "model_prediction": ans}
    d["vote"] = vote

# ------------------------------------------------------------------ tables
summary, flips, subs = [], [], []
for mname, d in data.items():
    base = {k: is_ok(r) for k, r in d["zero-shot"].items()}
    for s, rows_d in d.items():
        rows = [rows_d[k] for k in sorted(rows_d)]
        m, ci = metrics(rows), boot_ci(rows)
        yes_pred = pct([S(r["model_prediction"]) == "1" for r in rows])
        yes_gt = pct([S(r["gt_answer"]) == "1" for r in rows])
        acc_y = pct([is_ok(r) for r in rows if S(r["gt_answer"]) == "1"])
        acc_n = pct([is_ok(r) for r in rows if S(r["gt_answer"]) == "0"])
        summary.append(dict(model=mname, strategy=LABEL[s], n=m["n"],
                            aAcc=m["aAcc"], aAcc_lo=ci["aAcc"][0], aAcc_hi=ci["aAcc"][1],
                            qAcc=m["qAcc"], qAcc_lo=ci["qAcc"][0], qAcc_hi=ci["qAcc"][1],
                            fAcc=m["fAcc"], fAcc_lo=ci["fAcc"][0], fAcc_hi=ci["fAcc"][1],
                            pred_yes_pct=yes_pred, true_yes_pct=yes_gt, yes_minus_true=yes_pred - yes_gt,
                            acc_on_true_yes=acc_y, acc_on_true_no=acc_n, balanced_acc=(acc_y + acc_n) / 2,
                            unparseable=sum(S(r["model_prediction"]) not in ("0", "1") for r in rows)))
        if s != "zero-shot":
            w2r = sum(1 for r in rows if is_ok(r) and not base[r["key"] if "key" in r else rowkey(r["split"], r)])
            r2w = sum(1 for r in rows if not is_ok(r) and base[r["key"] if "key" in r else rowkey(r["split"], r)])
            flips.append(dict(model=mname, strategy=LABEL[s], wrong_to_right=w2r, right_to_wrong=r2w,
                              net=w2r - r2w, sign_test_p=sign_p(w2r, r2w)))
        for (c, sc) in sorted({(r["category"], r["subcategory"]) for r in rows}):
            sub = [r for r in rows if (r["category"], r["subcategory"]) == (c, sc)]
            mm = metrics(sub)
            subs.append(dict(model=mname, strategy=LABEL[s], subcategory=f"{c}_{sc}", n=mm["n"],
                             aAcc=mm["aAcc"], qAcc=mm["qAcc"], fAcc=mm["fAcc"]))

summary, flips, subs = pd.DataFrame(summary), pd.DataFrame(flips), pd.DataFrame(subs)

pd.set_option("display.width", 250, "display.max_columns", 30)
show = summary.copy()
for k in ("aAcc", "qAcc", "fAcc"):
    show[k] = show.apply(lambda r: f"{r[k]:.1f} [{r[k + '_lo']:.1f}, {r[k + '_hi']:.1f}]", axis=1)
print("\n=== Main results (95% CI from resampling sets) ===")
print(show[["model", "strategy", "n", "aAcc", "qAcc", "fAcc"]].to_string(index=False))
print("\n=== Yes/No bias ===")
print(summary[["model", "strategy", "pred_yes_pct", "true_yes_pct", "yes_minus_true", "acc_on_true_yes", "acc_on_true_no",
               "balanced_acc", "unparseable"]].round(1).to_string(index=False))
print("\n=== Changed answers vs zero-shot (sign test on changed questions) ===")
print(flips.round(3).to_string(index=False))
print("\n=== Per subcategory: aAcc / qAcc / fAcc (about 10-40 questions each: read with care) ===")
print(subs.assign(v=subs.apply(lambda r: f"{r.aAcc:.0f}/{r.qAcc:.0f}/{r.fAcc:.0f}", axis=1))
      .pivot(index=["model", "subcategory"], columns="strategy", values="v").to_string())

summary.round(2).to_csv(os.path.join(DIR, "phase3_summary.csv"), index=False)
flips.round(4).to_csv(os.path.join(DIR, "phase3_flips.csv"), index=False)
subs.round(2).to_csv(os.path.join(DIR, "phase3_by_subcategory.csv"), index=False)

# ------------------------------------------------------------------ figures for the report
order = [LABEL[s] for s in ("zero-shot",) + tuple(STRATS)]
models = list(data)
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True, squeeze=False)
for ax, mname in zip(axes[0], models):
    sub = summary[(summary.model == mname) & (summary.strategy.isin(order))].set_index("strategy").reindex([o for o in order if o in set(summary.strategy)])
    w = 0.25
    for j, (k, colr) in enumerate((("aAcc", "#4C78A8"), ("qAcc", "#F58518"), ("fAcc", "#54A24B"))):
        xs = [i + (j - 1) * w for i in range(len(sub))]
        err = [[sub[k] - sub[k + "_lo"]], [sub[k + "_hi"] - sub[k]]]
        ax.bar(xs, sub[k], w, label=k, color=colr, yerr=[(sub[k] - sub[k + "_lo"]).values, (sub[k + "_hi"] - sub[k]).values],
               capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(range(len(sub))); ax.set_xticklabels([s.replace("Self-consistency", "Self-cons.") for s in sub.index], fontsize=8)
    ax.set_title(mname, fontsize=10); ax.set_ylim(0, 100)
axes[0][0].set_ylabel("Accuracy (%)"); axes[0][0].legend(fontsize=8)
plt.tight_layout(); fig.savefig(os.path.join(DIR, "phase3_metrics.png"), dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.6), sharey=True, squeeze=False)
for ax, mname in zip(axes[0], models):
    sub = summary[(summary.model == mname) & (summary.strategy.isin(order))].set_index("strategy").reindex([o for o in order if o in set(summary.strategy)])
    ax.bar(range(len(sub)), sub.pred_yes_pct, color="#4C78A8")
    ax.axhline(sub.true_yes_pct.iloc[0], color="black", ls="--", lw=1, label="true Yes share")
    ax.set_xticks(range(len(sub))); ax.set_xticklabels([s.replace("Self-consistency", "Self-cons.") for s in sub.index], fontsize=8)
    ax.set_title(mname, fontsize=10); ax.set_ylim(0, 100)
axes[0][0].set_ylabel("Answers that are Yes (%)"); axes[0][0].legend(fontsize=8)
plt.tight_layout(); fig.savefig(os.path.join(DIR, "phase3_yes_rate.png"), dpi=200); plt.close(fig)
print("\nSaved: phase3_summary.csv, phase3_flips.csv, phase3_by_subcategory.csv, phase3_metrics.png, phase3_yes_rate.png")
