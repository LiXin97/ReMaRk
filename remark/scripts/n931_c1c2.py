"""(per-checkpoint paired comparison) and C2 (steps-to-match with a bootstrap CI), registered 2026-10-04 15:06.

--cache: per-row validation R@20 of every validated checkpoint (and the selected SFT, step 0) of the settings below,
  crs_eval truncate matching (crs_breakdown.per_row), dataset by run prefix; saved as
  outputs/crs/analysis/c1_cache/<val dir>.npy. CPU-heavy: run on a node (--workers).
--analyze: C1: at every step validated in both arms, ReMaRk - baseline (per row), paired bootstrap over requests
  (2,000 resamples, seed 42); "ahead at k of K" = CI above 0 (and the count with a positive estimate).
  C2: target T = baseline's validation at 2,000 (2k pairs) or the baseline's best (plateau pairs); s* = first crossing of
  T by ReMaRk's curve (linear interpolation; step 0 = SFT where validated); bootstrap over requests recomputing both
  curves, T (and, for plateau pairs, the baseline's best step) and s*; resamples without a crossing count as > last step.
  GPU-hours from same-node s/step (n928_step_cost) at 2B and 4B only.
Output: outputs/crs/analysis/n931_c1c2.json
Usage: python3 remark/scripts/n931_c1c2.py --cache --workers 32   (node);   env/bin/python ... --analyze
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
V, A = C / "val", C / "analysis"
CACHE = A / "c1_cache"
DATA = {"reddit": (REPO / "remark/third_party/Rank-GRPO/reddit_sft_dataset", None),
        "reddit_books": (REPO / "remark/data/crs/reddit_books/sft_dataset", REPO / "remark/data/crs/reddit_books/catalog.pkl"),
        "music": (REPO / "remark/data/crs/music/sft_dataset", REPO / "remark/data/crs/music/catalog.pkl")}
# setting -> (dataset, ReMaRk run, {baseline label: run}, SFT validation dir or None, kind)
SETTINGS = {
    "2B": ("reddit", "2B_union_group_s3407_p0c", {"rankgrpo": "2B_single_group_s3407_p0c"}, "sft_qwen35_2B_step1500", "2k"),
    "4B": ("reddit", "4B_union_group_s3407_p0c", {"rankgrpo": "4B_single_group_s3407_p0c"}, "sft_qwen35_4B_step1000", "2k"),
    "9B": ("reddit", "9B_union_group_s3407_p0c", {"rankgrpo": "9B_single_group_s3407_p0c"}, "sft_qwen35_9B_step1000", "2k"),
    "27B": ("reddit", "27B_union_group_s3407_p0c", {"rankgrpo": "27B_single_group_s3407_p0c"}, "sft_27B_step1000", "2k"),
    "35BA3B": ("reddit", "35BA3B_union_group_s3407_p0c", {"rankgrpo": "35BA3B_single_group_s3407_p0c"}, "sft_35BA3B_step1500", "2k"),
    "llama3b": ("reddit", "llama3b_union_group_s3407_p0c", {"rankgrpo": "llama3b_single_group_s3407_p0c",
                "perresp_batch": "llama3b_single_batch_s3407_p0c"}, None, "2k"),
    "llama8b": ("reddit", "llama8b_union_group_s3407_p0c", {"rankgrpo": "llama8b_single_group_s3407_p0c"}, "sft_llama8b_step1000", "2k"),
    "books2B": ("reddit_books", "reddit_books_2B_union_group_s3407_p0c_cap384_ex",
                {"rankgrpo": "reddit_books_2B_single_group_s3407_p0c_cap384_ex",
                 "perresp_batch": "reddit_books_2B_single_batch_s3407_p0c_cap384_ex"}, "sft_reddit_books_2B_step1000", "2k"),
    "books4B": ("reddit_books", "reddit_books_4B_union_group_s3407_p0c_cap384_ex",
                {"rankgrpo": "reddit_books_4B_single_group_s3407_p0c_cap384_ex",
                 "perresp_batch": "reddit_books_4B_single_batch_s3407_p0c_cap384_ex"}, "sft_reddit_books_4B_step1500", "2k"),
    "music2B": ("music", "music_2B_union_group_s3407_p0c_ex", {"rankgrpo": "music_2B_single_group_s3407_p0c_ex",
                "perresp_batch": "music_2B_single_batch_s3407_p0c_ex"},
                "sft_music_2B_step1500", "2k"),
    "R1_2B_plateau": ("reddit", "2B_union_group_s3407_p0c_ms8000", {"rankgrpo": "2B_single_group_s3407_p0c_ms8000"},
                      "sft_qwen35_2B_step1500", "plateau"),
    "R1b_2B_s42_plateau": ("reddit", "2B_union_group_s42_p0c_ms8000", {"rankgrpo": "2B_single_group_s42_p0c_ms8000"},
                           "sft_qwen35_2B_step1500", "plateau"),
    "R1c_4B_plateau": ("reddit", "4B_union_group_s3407_p0c_ms8000", {"rankgrpo": "4B_single_group_s3407_p0c_ms8000"},
                       "sft_qwen35_4B_step1000", "plateau"),
    "R1d_books2B_plateau": ("reddit_books", "reddit_books_2B_union_group_s3407_p0c_cap384_ex_ms8000",
                            {"rankgrpo": "reddit_books_2B_single_group_s3407_p0c_cap384_ex_ms8000"},
                            "sft_reddit_books_2B_step1000", "plateau"),
    "G3_music2B_plateau": ("music", "music_2B_union_group_s3407_p0c_ex_ms8000",
                           {"rankgrpo": "music_2B_single_group_s3407_p0c_ex_ms8000"}, "sft_music_2B_step1500", "plateau"),
}
# other seeds of the 2,000-step pairs (independent audit E2, author 2026-10-07): per-seed steps to match
for _sd in ("s42", "s1234"):
    SETTINGS[f"2B_{_sd}"] = ("reddit", f"2B_union_group_{_sd}_p0c", {"rankgrpo": f"2B_single_group_{_sd}_p0c",
                             "perresp_batch": f"2B_single_batch_{_sd}_p0c"}, "sft_qwen35_2B_step1500", "2k")
    SETTINGS[f"4B_{_sd}"] = ("reddit", f"4B_union_group_{_sd}_p0c", {"rankgrpo": f"4B_single_group_{_sd}_p0c"},
                             "sft_qwen35_4B_step1000", "2k")
    SETTINGS[f"books2B_{_sd}"] = ("reddit_books", f"reddit_books_2B_union_group_{_sd}_p0c_cap384_ex",
                                  {"perresp_batch": f"reddit_books_2B_single_batch_{_sd}_p0c_cap384_ex"},
                                  "sft_reddit_books_2B_step1000", "2k")
SETTINGS["9B_s1234"] = ("reddit", "9B_union_group_s1234_p0c", {"rankgrpo": "9B_single_group_s1234_p0c"},
                        "sft_qwen35_9B_step1000", "2k")
SETTINGS["books4B_s42"] = ("reddit_books", "reddit_books_4B_union_group_s42_p0c_cap384_ex",
                           {"perresp_batch": "reddit_books_4B_single_batch_s42_p0c_cap384_ex"}, "sft_reddit_books_4B_step1500", "2k")
SETTINGS["llama3b_s42"] = ("reddit", "llama3b_union_group_s42_p0c", {"rankgrpo": "llama3b_single_group_s42_p0c"}, None, "2k")
# same-node s/step (n928_step_cost.json, 2k pairs on one node): ReMaRk / Rank-GRPO
SAME_NODE = {"2B": (5.89, 5.65), "4B": (12.24, 11.50)}
# plateau pairs on one node each (mean s/step over every logged step):
# ReMaRk / Rank-GRPO. Added by the author 2026-10-05 with the R1d and G3 settings, together with "c2_2k" below.
# perresp_batch of llama3b / music2B added by the author 2026-10-07 (the better per-responder arm on validation).
PLATEAU_NODE = {"R1d_books2B_plateau": (4.82, 4.89), "G3_music2B_plateau": (3.90, 3.70)}


def steps_of(run: str) -> dict[int, Path]:
    out = {}
    for d in V.glob(f"grpo_qwen_{run}_step*"):
        m = re.search(r"_step(\d+)$", d.name)
        if m and d.name == f"grpo_qwen_{run}_step{m.group(1)}" and (d / "per_instance.jsonl").exists():
            out[int(m.group(1))] = d
    return dict(sorted(out.items()))


def _load(ds_key: str):
    from datasets import load_from_disk
    from crs_eval import RG
    data, cat = DATA[ds_key]
    if ds_key == "reddit":
        from crs_eval import DATA as RD
        data, cat = RD, RG / "reddit_catalog/gt_catalog.pkl"
    ds = load_from_disk(str(Path(data) / "validation"))
    catalog = set(tuple(x) for x in pickle.load(open(cat, "rb")))
    return ds, catalog


_STATE: dict = {}


def _work(job: tuple[str, str]) -> str:
    ds_key, d = job
    out = CACHE / f"{Path(d).name}.npy"
    if out.exists():
        return "cached"
    if ds_key not in _STATE:
        _STATE[ds_key] = _load(ds_key)
    from crs_breakdown import per_row
    ds, catalog = _STATE[ds_key]
    r20 = per_row(Path(d), ds, catalog)[0]
    np.save(out, r20)
    return "done"


def cache(workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ds_key, rm, bases, sft, _ in SETTINGS.values():
        for run in [rm, *bases.values()]:
            jobs += [(ds_key, str(d)) for d in steps_of(run).values()]
        if sft and (V / sft / "per_instance.jsonl").exists():
            jobs.append((ds_key, str(V / sft)))
    jobs = sorted(set(jobs))
    print(len(jobs), "validation runs", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_work, jobs, chunksize=2)):
            if i % 50 == 0:
                print(i, r, flush=True)
    print("CACHE_DONE", flush=True)


def groups(ds_key: str) -> np.ndarray:
    """Request id per validation row (rows sharing a prompt)."""
    f = CACHE / f"_groups_{ds_key}.npy"
    if f.exists():
        return np.load(f)
    from datasets import load_from_disk
    from crs_eval import DATA as RD
    data = RD if ds_key == "reddit" else DATA[ds_key][0]
    ds = load_from_disk(str(Path(data) / "validation"))
    ids, g = {}, []
    for p in ds["prompt"]:
        g.append(ids.setdefault(p[0]["content"], len(ids)))
    g = np.array(g); np.save(f, g)
    return g


def curve(run: str, sft: str | None) -> dict[int, np.ndarray]:
    cur = {k: np.load(CACHE / f"{d.name}.npy") for k, d in steps_of(run).items() if (CACHE / f"{d.name}.npy").exists()}
    if sft and (CACHE / f"{sft}.npy").exists():
        cur[0] = np.load(CACHE / f"{sft}.npy")
    return dict(sorted(cur.items()))


def crossing(steps: np.ndarray, vals: np.ndarray, target: float) -> float:
    """First step at which the piecewise-linear curve reaches target (np.inf if never)."""
    if vals[0] >= target:
        return float(steps[0])
    for i in range(1, len(steps)):
        if vals[i] >= target:
            v0, v1, s0, s1 = vals[i - 1], vals[i], steps[i - 1], steps[i]
            return float(s0 + (target - v0) / (v1 - v0) * (s1 - s0)) if v1 > v0 else float(s1)
    return float("inf")


def analyze() -> None:
    rng = np.random.default_rng(42)
    res = {"note": __doc__.split("\n")[0], "settings": {}}
    for name, (ds_key, rm, bases, sft, kind) in SETTINGS.items():
        g = groups(ds_key); n_req = g.max() + 1
        sizes = np.bincount(g, minlength=n_req)
        draws = rng.integers(0, n_req, size=(2000, n_req))
        w = np.stack([np.bincount(d, minlength=n_req) for d in draws])  # resample weights per request
        rows_w = (w * sizes).sum(1)
        R = curve(rm, sft)
        if len(R) < 2:
            continue
        res["settings"][name] = st = {"dataset": ds_key, "kind": kind, "remark": rm}
        for lab, br in bases.items():
            B = curve(br, sft)
            common = [k for k in R if k in B and k > 0]
            # C1
            per_ck = {}
            for k in common:
                d = R[k] - B[k]
                sums = np.bincount(g, weights=d, minlength=n_req)
                boot = (w @ sums) / rows_w
                per_ck[k] = [float(d.mean()), *map(float, np.percentile(boot, [2.5, 97.5]))]
            ahead = sum(v[1] > 0 for v in per_ck.values()); pos = sum(v[0] > 0 for v in per_ck.values())
            # C2
            ks_r, ks_b = np.array(sorted(R)), np.array(sorted(B))
            sums_r = np.stack([np.bincount(g, weights=R[k], minlength=n_req) for k in ks_r])
            sums_b = np.stack([np.bincount(g, weights=B[k], minlength=n_req) for k in ks_b])
            def est(wv: np.ndarray, rw: float, at2k: bool = False):
                cr = (sums_r @ wv) / rw; cb = (sums_b @ wv) / rw
                bmask = ks_b > 0
                if kind == "2k" or at2k:
                    if 2000 not in ks_b:
                        return None
                    t = cb[list(ks_b).index(2000)]; ref = 2000.0
                else:
                    j = int(np.argmax(np.where(bmask, cb, -1))); t = cb[j]; ref = float(ks_b[j])
                return crossing(ks_r.astype(float), cr, t), ref, t
            point = est(np.ones(n_req), float(sizes.sum()))
            if point is None:
                st[lab] = {"c1": per_ck, "ahead": [ahead, pos, len(per_ck)], "c2": None}
                continue
            bs = [est(w[i].astype(float), float(rows_w[i])) for i in range(2000)]
            s_b = np.array([b[0] for b in bs]); ref_b = np.array([b[1] for b in bs]); ratio_b = s_b / ref_b
            fin = np.isfinite(s_b)
            c2 = {"target": float(point[2]), "ref_step": point[1], "s_star": point[0], "ratio": point[0] / point[1],
                  "s_star_ci": [float(np.percentile(np.where(fin, s_b, 1e9), q)) for q in (2.5, 97.5)],
                  "ratio_ci": [float(np.percentile(np.where(fin, ratio_b, 1e9), q)) for q in (2.5, 97.5)],
                  "share_no_crossing": float(1 - fin.mean())}
            # plateau pairs: also the target "baseline's validation at step 2,000" (same resamples, no extra draws)
            if kind == "plateau" and 2000 in ks_b:
                p2 = est(np.ones(n_req), float(sizes.sum()), True)
                b2 = np.array([est(w[i].astype(float), float(rows_w[i]), True)[0] for i in range(2000)])
                f2 = np.isfinite(b2)
                c2["c2_2k"] = {"target": float(p2[2]), "s_star": p2[0], "ratio": p2[0] / 2000.0,
                               "ratio_ci": [float(np.percentile(np.where(f2, b2 / 2000.0, 1e9), q)) for q in (2.5, 97.5)],
                               "share_no_crossing": float(1 - f2.mean())}
            if name in PLATEAU_NODE and lab == "rankgrpo":
                sr, sb = PLATEAU_NODE[name]
                c2["gpu_hour_ratio"] = c2["ratio"] * sr / sb
                c2["gpu_hour_ratio_ci"] = [x * sr / sb for x in c2["ratio_ci"]]
                c2["s_per_step_same_node"] = [sr, sb]
            size = name.split("_")[0] if name.startswith(("2B", "4B")) else ("2B" if name.startswith("R1_") or name.startswith("R1b") else "4B" if name.startswith("R1c") else None)
            if size in SAME_NODE and lab == "rankgrpo":
                sr, sb = SAME_NODE[size]
                c2["gpu_hour_ratio"] = c2["ratio"] * sr / sb
                c2["gpu_hour_ratio_ci"] = [x * sr / sb for x in c2["ratio_ci"]]
                c2["s_per_step_same_node"] = [sr, sb]
            st[lab] = {"c1": {str(k): v for k, v in per_ck.items()}, "ahead": [ahead, pos, len(per_ck)], "c2": c2}
            print(name, lab, "ahead", ahead, "/", len(per_ck), "pos", pos, "| s*", round(point[0], 1), "ratio",
                  round(c2["ratio"], 3), [round(x, 3) for x in c2["ratio_ci"]], flush=True)
    (A / "n931_c1c2.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cache:
        cache(a.workers)
    if a.analyze:
        analyze()
