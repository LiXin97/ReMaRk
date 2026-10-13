"""List length at the locked checkpoint of each batch-scaled arm: from the test and validation
per_instance.jsonl of the locked step (crs_eval keeps the first 20 parsed items, so "20" means >= 20 parseable lines and
over-long lists are not visible here), and from the training log (rank/items_emitted, rank/frac_exact_len; temperature-1
samples) averaged over the 50 steps up to the locked step, where the log covers it.
Usage: python3 remark/scripts/n934_lengths.py -> outputs/crs/analysis/n934_lengths.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n934_stability import RUNS, logged  # noqa: E402

C = Path(__file__).resolve().parents[2] / "remark/outputs/crs"


def lens(p: Path) -> dict | None:
    if not p.exists():
        return None
    n = [len(json.loads(line)["recs"]) for line in open(p)]
    return {"rows": len(n), "share_20": sum(x == 20 for x in n) / len(n), "share_lt20": sum(x < 20 for x in n) / len(n),
            "mean_items_capped20": sum(n) / len(n)}


def main() -> None:
    out = {}
    for run in RUNS:
        step = json.load(open(C / f"locks/{run}.json"))["best_step"]
        res = {"locked_step": step, "test": lens(C / f"test/test_{run}_step{step}/per_instance.jsonl"),
               "val": lens(C / f"val/grpo_qwen_{run}_step{step}/per_instance.jsonl")}
        lg = logged(run)
        win = [lg[s] for s in sorted(lg) if step - 50 < s <= step]
        res["train_log"] = ({"steps": [min(s for s in lg if step - 50 < s <= step), step], "mean_items":
                             sum(w["rank/items_emitted"] for w in win) / len(win),
                             "exact_share": sum(w["rank/frac_exact_len"] for w in win) / len(win)}
                            if win else {"note": f"log covers steps <= {max(lg) if lg else 0} only (node lost)"})
        out[run] = res
        t, tl = res["test"], res["train_log"]
        print(f"{run:50s} @{step:5d} test share20 {t['share_20']:.3f} mean {t['mean_items_capped20']:.2f} | "
              + (f"train items {tl['mean_items']:.2f} exact {tl['exact_share']:.3f}" if "mean_items" in tl else tl["note"]))
    (C / "analysis/n934_lengths.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
