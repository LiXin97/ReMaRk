"""Instability check (rule of A, applied to the batch-scaled arms of Table 2).

A run is unstable at the first 10-step logging window after step 100 in which any of
  (i)   KL > 5x the maximum KL of the matched group-scaled run of the same column and reward (all its logged windows),
  (ii)  mean items per list outside [19.5, 20.5] or exact-20 share < 90%, over a 50-step window (5 log entries),
  (iii) truncated-completion share > 10% over a 50-step window,
  (iv)  validation R@20 more than 3 points below the run's running best.
Logs: outputs/crs/train_logs/<tag>/grpo_<run>.log. Writes outputs/crs/analysis/n934_stability.json.
Usage: python3 remark/scripts/n934_stability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n933_a_stability import val_curve, logged  # noqa: E402
from n928_step_cost import entries_from_log, L  # noqa: E402

C = Path(__file__).resolve().parents[2] / "remark/outputs/crs"
RUNS = {  # arm -> matched group-scaled run (same column, same reward)
    "llama3b_single_batch_s3407_p0c": "llama3b_single_group_s3407_p0c",
    "llama3b_union_batch_s3407_p0c": "llama3b_union_group_s3407_p0c",
    "llama8b_single_batch_s3407_p0c": "llama8b_single_group_s3407_p0c",
    "llama8b_union_batch_s3407_p0c": "llama8b_union_group_s3407_p0c",
    "music_2B_single_batch_s3407_p0c_ex": "music_2B_single_group_s3407_p0c_ex",
    "music_2B_union_batch_s3407_p0c_ex": "music_2B_union_group_s3407_p0c_ex",
    "reddit_books_4B_union_batch_s3407_p0c_cap384_ex": "reddit_books_4B_union_group_s3407_p0c_cap384_ex",
}


def check(run: str, ref: str) -> dict:
    ref_kl = max((x.get("kl", 0.0) for p in list(L.glob(f"*/grpo_{ref}.log")) + list(L.glob(f"*/grpo_{ref}_seg*.log"))
                  for x in entries_from_log(p)), default=None)
    lg = logged(run)
    res = {"run": run, "ref_run": ref, "ref_max_kl": ref_kl, "kl_limit": 5 * ref_kl if ref_kl else None,
           "last_step": max(lg) if lg else 0, "unstable": None}
    steps = list(lg)
    for j, s in enumerate(steps):
        if s <= 100:
            continue
        x = lg[s]
        if ref_kl and x.get("kl", 0) > 5 * ref_kl:
            res["unstable"] = {"step": s, "criterion": "i", "value": x["kl"]}
            break
        win = [lg[t] for t in steps[max(0, j - 4): j + 1]]
        if len(win) == 5:
            items = sum(w["rank/items_emitted"] for w in win) / 5
            exact = sum(w["rank/frac_exact_len"] for w in win) / 5
            trunc = sum(w["completions/clipped_ratio"] for w in win) / 5
            if not 19.5 <= items <= 20.5 or exact < 0.90:
                res["unstable"] = {"step": s, "criterion": "ii", "items": round(items, 3), "exact": round(exact, 4)}
                break
            if trunc > 0.10:
                res["unstable"] = {"step": s, "criterion": "iii", "truncated": trunc}
                break
    cv, best = val_curve(run), -1.0
    for s, v in cv.items():
        best = max(best, v)
        if s > 100 and v < best - 3 and (res["unstable"] is None or s < res["unstable"]["step"]):
            res["unstable"] = {"step": s, "criterion": "iv", "val": v, "running_best": best}
            break
    res["val"] = cv
    return res


def main() -> None:
    out = {r: check(r, ref) for r, ref in RUNS.items()}
    (C / "analysis/n934_stability.json").write_text(json.dumps(out, indent=1))
    for r, x in out.items():
        print(f"{r:50s} last {x['last_step']:5d} KL limit {x['kl_limit'] and round(x['kl_limit'], 2)} ",
              "UNSTABLE " + json.dumps(x["unstable"]) if x["unstable"] else "stable so far")


if __name__ == "__main__":
    main()
