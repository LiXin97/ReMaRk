"""Train/test overlap sensitivity for Reddit-v2: test requests whose submission also occurs in the GRPO training set.

Submission = normalised opening post (crs_neighbor_prior.submission_key; the released data carry no submission id).
Overlap criteria (test context vs GRPO train contexts):
  submission (primary, used for exclusion): same submission key
  full_prompt: identical prompt (strict subset of the above)
  loose_opening_post: opening post equal after lower-casing and keeping only [a-z0-9] tokens (report only)
The SFT training split is checked the same way (report only). The GRPO train prompt and ground-truth columns are streamed
once in batches; for overlapping submissions the training ground truth is kept to test whether the same responder answer
(identical item set) is in training.
Then the primary comparison (a - b, per-row R@k / N@k, paired bootstrap over contexts, 2,000, seed 42; crs_compare.py
functions) is recomputed on all test rows and without the overlapping requests; the submission-clustered CI is added.

Usage: env/bin/python remark/scripts/crs_overlap_sensitivity.py \
           [--a remark/outputs/crs/test/test_2B_union_s3407_step1500] \
           [--b remark/outputs/crs/test/test_2B_single_group_s3407_step2000] \
           [--out remark/outputs/crs/analysis/test_2B_overlap_sensitivity.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import SUBMISSION_KEY, context_groups, paired_stats, per_row, submission_clusters  # noqa: E402
from crs_eval import DATA, REPO, RG  # noqa: E402
from crs_neighbor_prior import submission_key  # noqa: E402

GRPO = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
TEST = REPO / "remark/outputs/crs/test"
CRITERIA = ("submission", "full_prompt", "loose_opening_post")


def loose_key(sub_key: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", sub_key.lower()))


def h(s: str) -> bytes:
    return hashlib.blake2b(s.encode(), digest_size=16).digest()


def keys_of(prompt: str) -> dict[str, bytes]:
    sk = submission_key(prompt)
    return {"submission": h(sk), "full_prompt": h(prompt), "loose_opening_post": h(loose_key(sk))}


def item_set(gts) -> frozenset:
    return frozenset((g[0], int(g[1])) for g in gts)


def train_keys(path: Path, want: set[bytes], batch_size: int = 4096
               ) -> tuple[dict[str, set[bytes]], int, int, dict[bytes, list[frozenset]]]:
    """One streamed pass over the prompt and GT columns; hashed keys, plus the GT item sets of rows whose submission
    key is in `want`."""
    ds = load_from_disk(str(path)).select_columns(["prompt", "groundtruth_with_release_year"])
    out: dict[str, set[bytes]] = {c: set() for c in CRITERIA}
    gt_of: dict[bytes, list[frozenset]] = {}
    n_rows = 0
    for batch in ds.iter(batch_size=batch_size):
        for p, gts in zip(batch["prompt"], batch["groundtruth_with_release_year"]):
            n_rows += 1
            text = p[0]["content"]
            hp = h(text)
            if hp not in out["full_prompt"]:
                for c, v in keys_of(text).items():
                    out[c].add(v)
            hs = h(submission_key(text))
            if hs in want:
                gt_of.setdefault(hs, []).append(item_set(gts))
    return out, n_rows, len(out["full_prompt"]), gt_of


def compare(xa: dict, xb: dict, groups: list[list[int]], contexts: list[str], keep: list[int], seed: int,
            n_boot: int) -> dict:
    """Per-row paired bootstrap on the kept contexts: context unit (primary) and submission unit."""
    g = [groups[j] for j in keep]
    cl = submission_clusters([contexts[j] for j in keep])
    out = {"n_rows": int(sum(map(len, g))), "n_requests": len(g), "n_submissions": len(cl)}
    for unit, clusters in (("context", None), ("submission", cl)):
        n = len(g) if clusters is None else len(clusters)
        draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
        rows = np.sort(np.concatenate(g))  # original row order: with all contexts kept this is crs_compare.py exactly
        # paired_stats reads rows through the groups; restrict the arrays so means are over kept rows only
        remap = {r: i for i, r in enumerate(rows)}
        g_local = [[remap[r] for r in grp] for grp in g]
        out[unit] = {k: paired_stats(xa[k][rows], xb[k][rows], g_local, draws, "row", clusters) for k in xa}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(TEST / "test_2B_union_s3407_step1500"))
    ap.add_argument("--b", default=str(TEST / "test_2B_single_group_s3407_step2000"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "remark/outputs/crs/analysis/test_2B_overlap_sensitivity.json"))
    a = ap.parse_args()
    ds = load_from_disk(str(DATA / a.split))
    ctx = context_groups([p[0]["content"] for p in ds["prompt"]])
    contexts, groups = list(ctx), list(ctx.values())
    test_keys = [keys_of(c) for c in contexts]
    test_sub = [submission_key(c) for c in contexts]

    overlap, flags, train_gt = {}, {}, {}
    want = {k["submission"] for k in test_keys}
    for name, path in (("grpo_train", GRPO), ("sft_train", DATA / "train")):
        tk, n_rows, n_ctx, train_gt[name] = train_keys(path, want)
        o = {"train_rows": n_rows, "train_contexts": n_ctx, "train_submissions": len(tk["submission"])}
        for c in CRITERIA:
            f = [test_keys[j][c] in tk[c] for j in range(len(contexts))]
            flags[(name, c)] = f
            o[c] = {"requests": int(sum(f)), "rows": int(sum(len(groups[j]) for j in range(len(f)) if f[j])),
                    "submissions": len({test_sub[j] for j in range(len(f)) if f[j]})}
        overlap[name] = o
        print(name, json.dumps(o))

    flag = flags[("grpo_train", "submission")]
    examples = sorted({" ".join(test_sub[j].split())[:160] for j in range(len(flag)) if flag[j]})
    # does the test responder's answer itself occur in training (same submission)?
    gt_ds = ds["groundtruth_with_release_year"]
    same_set = items_in = items_all = 0
    for j in (j for j in range(len(flag)) if flag[j]):
        tr_sets = train_gt["grpo_train"][test_keys[j]["submission"]]
        tr_items = frozenset().union(*tr_sets)
        for i in groups[j]:
            s = item_set(gt_ds[i])
            same_set += s in tr_sets
            items_in += len(s & tr_items)
            items_all += len(s)
    overlap["grpo_train"]["overlapping_rows_answer_in_train"] = {
        "rows_with_identical_item_set_in_same_submission_train_rows": int(same_set),
        "test_gt_items_named_in_same_submission_train_rows": int(items_in), "test_gt_items": int(items_all),
        "train_rows_of_overlapping_submissions": int(sum(len(train_gt["grpo_train"][k]) for k in
                                                          {test_keys[j]["submission"] for j in range(len(flag)) if flag[j]}))}
    print("answer in train:", json.dumps(overlap["grpo_train"]["overlapping_rows_answer_in_train"]))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    xa, xb = per_row(Path(a.a), ds, catalog), per_row(Path(a.b), ds, catalog)
    keep_all = list(range(len(contexts)))
    keep = [j for j in keep_all if not flag[j]]
    only = [j for j in keep_all if flag[j]]
    res = {"all_test": compare(xa, xb, groups, contexts, keep_all, a.seed, a.n_boot),
           "excluding_grpo_overlap": compare(xa, xb, groups, contexts, keep, a.seed, a.n_boot)}
    if only:
        res["grpo_overlap_only"] = compare(xa, xb, groups, contexts, only, a.seed, a.n_boot)
    rel = lambda p: str(Path(p).resolve().relative_to(REPO))  # noqa: E731
    out = {"a": rel(a.a), "b": rel(a.b), "split": a.split, "data": str(DATA.relative_to(REPO)),
           "grpo_train": str(GRPO.relative_to(REPO)), "submission_key": SUBMISSION_KEY,
           "test": {"rows": len(ds), "requests": len(contexts), "submissions": len(set(test_sub))},
           "criteria": {"submission": "same normalised opening post (primary; used for the exclusion)",
                        "full_prompt": "identical prompt (whole conversation)",
                        "loose_opening_post": "opening post equal after lower-casing, [a-z0-9] tokens only"},
           "overlap": overlap, "excluded_by": "grpo_train / submission",
           "overlapping_opening_posts_first160": examples,
           "bootstrap": {"n_boot": a.n_boot, "seed": a.seed, "metric": "per-row truncate (crs_eval.py protocol)",
                         "diff": "a - b"},
           "comparison": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    for k, v in res.items():
        r, s = v["context"]["R@20"], v["submission"]["R@20"]
        print(f"{k}: rows {v['n_rows']} requests {v['n_requests']} submissions {v['n_submissions']} | R@20 "
              f"{r['diff']:+.4f} ctx [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] sub [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
