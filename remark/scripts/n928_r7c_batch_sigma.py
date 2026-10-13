"""Per-position reward sigma per arm at the start of training (E10(b) 2B SFT samples) and the logged
shaping scale (rank/shaping_scale) over training. CPU only, existing outputs.

Inputs
  samples:  remark/outputs/crs/e10b/2B/samples/shard{0..3}.jsonl, one line per training request {"idx", "texts",
            "scores"}: 8 lists sampled from the selected 2B SFT checkpoint (temperature 1, max 320 tokens); idx indexes
            remark/data/crs/grpo_union/train (72,233 requests). Used as the training-start list distribution of every
            arm (all arms start from this SFT checkpoint).
  rows:     the training sets of the arms (remark/scratchpad launcher mapping, launch5.sh):
            single     data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train   (383,013 rows = request x responder)
            union      remark/data/crs/grpo_union/train                               (72,233, union items, weight 1)
            consensus  remark/data/crs/grpo_consensus/train                           (72,233, weight = count / n)
            singlethr  remark/data/crs/redditthr/grpo/train                           (72,233, one responder per request)
            meanm2/4   remark/data/crs/grpo_mean_m{2,4}_rw/train                      (383,013, mean over m responders)
            consensusrw remark/data/crs/grpo_consensus_rw/train (383,013): verified to equal the consensus row of its
            request, so it is represented by the consensus arrays with requests drawn in proportion to n_x.
  catalogue: remark/third_party/Rank-GRPO/reddit_catalog/gt_catalog.pkl
  logs:     remark/outputs/crs/train_logs/<node>/grpo_<run>.log (TRL log records every 10 steps; each record's
            rank/shaping_scale is the mean over those 10 optimizer steps of 1 / (mean nonzero per-position std + 1e-4)).
Definitions
  Lists are parsed with crs_eval's parser (n927_e10b_sample.parse: Rank-GRPO process_rec_raw, first 20 items); the reward
  of row (x, r) at position k of list l is the weight of the row's ground-truth item matched at k by crs_eval.hits_for
  (truncate: catalogue items only, the row's own seen titles skipped, exact title + |year diff| <= 2); weight 1 for 0/1
  datasets. Deviations from training: P0c gives rank positions to non-blank lines (malformed lines are zero-reward
  positions) while this parser numbers parsed items only (malformed lines are rare in SFT samples); the trainers match with
  Rank-GRPO's matchers (title normalisation) instead of hits_for's exact title.
  Population grid (per reward definition x sampling): per-position mean and standard deviation (ddof 0) over all
  (row, list) pairs, weights: row sampling = every (request, responder) row once (requests in proportion to n_x), request
  sampling = every request once. Reward definitions: per-responder 0/1 (one responder per row; under request sampling the
  responder is drawn uniformly, i.e. weight 1/n_x per row), mean over responders (count / n), bounded sum = union 0/1,
  sum (responder count; reference), mean over m = 2 / 4 responders (dose-response datasets, row sampling).
  Batch simulation per arm (the trainer's own scale): 2,000 batches of 48 distinct rows of the arm's training set
  (uniform; numpy default_rng(42)); 8 lists per row. Batch scale: per-position std over the 384 lists (ddof 1, as
  torch.std) and shaping_scale = 1 / (mean of the per-position stds > 1e-6 + 1e-4). Group scale: per (row, position) std
  over its 8 lists (ddof 1) and shaping_scale = 1 / (mean of the (row, position) stds > 1e-6 + 1e-4)
  (lib/rank_grpo_trl.py _rank_advantages). The simulated shaping_scale is compared with the first logged record (steps
  1-10). Filtering of prompts > 2,048 tokens (<= 0.2% of rows) is ignored here.
Outputs: remark/outputs/crs/analysis/n928_r7c_batch_sigma.json; cache remark/outputs/crs/analysis/n928_cache/
         (per-row reward arrays, reused by n928_r7d_*.py; rebuilt if missing).
Usage:  OMP_NUM_THREADS=4 env/bin/python remark/scripts/n928_r7c_batch_sigma.py [--rebuild] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import REPO, RG, hits_for  # noqa: E402
from n927_e10b_sample import parse  # noqa: E402

C = REPO / "remark/outputs/crs"
A = C / "analysis"
CACHE = A / "n928_cache"
OUT = A / "n928_r7c_batch_sigma.json"
SAMPLES = C / "e10b/2B/samples"
LOGS = C / "train_logs"
R = 20
G = 8
DATASETS = {  # name -> (path, weighted)
    "single": (REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train", False),
    "union": (REPO / "remark/data/crs/grpo_union/train", False),
    "consensus": (REPO / "remark/data/crs/grpo_consensus/train", True),
    "singlethr": (REPO / "remark/data/crs/redditthr/grpo/train", False),
    "meanm2": (REPO / "remark/data/crs/grpo_mean_m2_rw/train", True),
    "meanm4": (REPO / "remark/data/crs/grpo_mean_m4_rw/train", True),
}
CONSENSUS_RW = REPO / "remark/data/crs/grpo_consensus_rw/train"
# arm -> (reward dataset, sampling, scale); sampling "rows" = uniform over the dataset's rows, "rows_n" = requests in
# proportion to n_x with the request's (consensus) reward, "requests" = uniform over requests
ARMS = {
    "single_group (1, Rank-GRPO)": ("single", "rows", "group"),
    "single_batch (2)": ("single", "rows", "batch"),
    "singlethr_batch (3)": ("singlethr", "requests", "batch"),
    "consensusrw_batch (4)": ("consensus", "rows_n", "batch"),
    "consensus_batch (5)": ("consensus", "requests", "batch"),
    "union_batch (6)": ("union", "requests", "batch"),
    "union_group (7, ReMaRk)": ("union", "requests", "group"),
    "meanm2rw_batch (dose m=2)": ("meanm2", "rows", "batch"),
    "meanm4rw_batch (dose m=4)": ("meanm4", "rows", "batch"),
}
# arm -> seed -> log files (relative to train_logs); consensus / consensusrw s3407 logs are the _cap320 reruns (identical
# configuration; their test runs are the ablation table's rows 4 and 5, configs/paper_test_tables.json)
ARM_LOGS = {
    "single_group (1, Rank-GRPO)": {"s3407": ["run_logs/grpo_2B_single_group_s3407_p0c.log"],
                                    "s42": ["run_logs/grpo_2B_single_group_s42_p0c.log"],
                                    "s1234": ["run_logs/grpo_2B_single_group_s1234_p0c.log"]},
    "single_batch (2)": {"s3407": ["run_logs/grpo_2B_single_batch_s3407_p0c.log"],
                         "s42": ["run_logs/grpo_2B_single_batch_s42_p0c.log"],
                         "s1234": ["run_logs/grpo_2B_single_batch_s1234_p0c.log"]},
    "singlethr_batch (3)": {"s3407": ["run_logs/grpo_2B_singlethr_batch_s3407_p0c.log"]},
    "consensusrw_batch (4)": {"s3407": ["run_logs/grpo_2B_consensusrw_batch_s3407_p0c_cap320.log"],
                              "s42": ["run_logs/grpo_2B_consensusrw_batch_s42_p0c.log"],
                              "s1234": ["run_logs/grpo_2B_consensusrw_batch_s1234_p0c.log"]},
    "consensus_batch (5)": {"s3407": ["run_logs/grpo_2B_consensus_batch_s3407_p0c_cap320.log"]},
    "union_batch (6)": {"s3407": ["run_logs/grpo_2B_union_batch_s3407_p0c.log"],
                        "s42": ["run_logs/grpo_2B_union_batch_s42_p0c.log"],
                        "s1234": ["run_logs/grpo_2B_union_batch_s1234_p0c.log"]},
    "union_group (7, ReMaRk)": {"s3407": ["run_logs/grpo_2B_union_group_s3407_p0c.log"]},
    "meanm2rw_batch (dose m=2)": {"s3407": ["run_logs/grpo_2B_meanm2rw_batch_s3407_p0c.log"],
                                  "s42": ["run_logs/grpo_2B_meanm2rw_batch_s42_p0c.log"],
                                  "s1234": ["run_logs/grpo_2B_meanm2rw_batch_s1234_p0c.log"]},
    "meanm4rw_batch (dose m=4)": {"s3407": ["run_logs/grpo_2B_meanm4rw_batch_s3407_p0c.log"],
                                  "s42": ["run_logs/grpo_2B_meanm4rw_batch_s42_p0c.log"],
                                  "s1234": ["run_logs/grpo_2B_meanm4rw_batch_s1234_p0c.log"]},
}
N_BATCH, BATCH_ROWS, SEED = 2000, 48, 42


# ----------------------------------------------------------------------------------------------- training-log parsing
REC = re.compile(r"\{'loss': .*?\}")
RESUME = re.compile(r"\[resume\] policy initialised from \S*?checkpoint-(\d+)")


def parse_log(path: Path, key: str = "rank/shaping_scale") -> dict:
    """Records of one TRL log file. Record j covers optimizer steps 10j+1..10j+10 of the segment; a segment started by
    the resume (log line '[resume] policy initialised from .../checkpoint-k') is offset by k."""
    txt = open(path, errors="replace").read()
    vals = []
    for rec in REC.findall(txt):
        m = re.search(rf"'{re.escape(key)}': '?([-0-9.e+]+)", rec)
        if m:
            vals.append(float(m.group(1)))
    res = RESUME.search(txt)
    offset = int(res.group(1)) if res else 0
    steps = [offset + 10 * (j + 1) for j in range(len(vals))]
    return {"file": str(path.relative_to(REPO)), "resumed_from": offset if res else None, "n_records": len(vals),
            "steps_covered": [steps[0] - 9, steps[-1]] if steps else None, "steps": steps, "values": vals}


def summarize(vals: list[float]) -> dict:
    v = np.array(vals)
    return {"n": len(v), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}


# ------------------------------------------------------------------------------------------------------ reward cache
_ROWS: dict = {}      # dataset -> request -> [(row id, gts, weights or None, seen)]
_CATALOG: set = set()


def _req_index() -> dict[str, int]:
    u = load_from_disk(str(DATASETS["union"][0])).select_columns(["prompt"])
    return {p[0]["content"]: i for i, p in enumerate(u["prompt"])}


def _load_rows(path: Path, weighted: bool, index: dict[str, int]) -> tuple[dict, np.ndarray]:
    ds = load_from_disk(str(path))
    cols = ["prompt", "seen_titles", "groundtruth_with_release_year"] + (["groundtruth_weights"] if weighted else [])
    by_req: dict = defaultdict(list)
    req = np.empty(len(ds), np.int32)
    i = 0
    for b in ds.select_columns(cols).iter(batch_size=20_000):
        for j in range(len(b["prompt"])):
            x = index[b["prompt"][j][0]["content"]]
            gts = [(g[0], int(g[1])) for g in b["groundtruth_with_release_year"][j]]
            w = [float(v) for v in b["groundtruth_weights"][j]] if weighted else None
            by_req[x].append((i, gts, w, frozenset(b["seen_titles"][j])))
            req[i] = x
            i += 1
    return by_req, req


def _work(shard: Path) -> dict:
    out = {d: ([], []) for d in _ROWS}
    union_match_rows, union_match = [], []
    n_lists, n_items, n_full = 0, 0, 0
    for line in open(shard):
        rec = json.loads(line)
        x = rec["idx"]
        lists = [parse(t) for t in rec["texts"]]
        assert len(lists) == G
        n_lists += G
        n_items += sum(len(L) for L in lists)
        n_full += sum(len(L) == R for L in lists)
        for d, rows in _ROWS.items():
            for rid, gts, w, seen in rows.get(x, ()):
                arr = np.zeros((G, R), np.float32)
                if d == "union":
                    mt = -np.ones((G, R), np.int16)
                for li, L in enumerate(lists):
                    _, matched = hits_for(L, gts, seen, _CATALOG, "truncate")
                    for gi, pos in matched:
                        if pos < R:
                            arr[li, pos] = 1.0 if w is None else w[gi]
                            if d == "union":
                                mt[li, pos] = gi
                out[d][0].append(rid)
                out[d][1].append(arr)
                if d == "union":
                    union_match_rows.append(rid)
                    union_match.append(mt)
    return {"rewards": {d: (np.array(v[0], np.int64), np.stack(v[1]) if v[1] else np.zeros((0, G, R), np.float32))
                        for d, v in out.items()},
            "union_match": (np.array(union_match_rows, np.int64), np.stack(union_match)),
            "parse": (n_lists, n_items, n_full)}


def build_cache(workers: int) -> None:
    global _ROWS, _CATALOG
    CACHE.mkdir(parents=True, exist_ok=True)
    _CATALOG = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    index = _req_index()
    reqs, checks = {}, {}
    for d, (path, weighted) in DATASETS.items():
        _ROWS[d], reqs[d] = _load_rows(path, weighted, index)
        print(f"[cache] {d}: {len(reqs[d])} rows", flush=True)
    # consensus = union items with weights: item lists and seen sets must agree request by request
    checks["consensus_items_equal_union"] = all(
        [g for g in _ROWS["consensus"][x][0][1]] == _ROWS["union"][x][0][1] and
        _ROWS["consensus"][x][0][3] == _ROWS["union"][x][0][3] for x in _ROWS["union"])
    # consensusrw rows: the consensus row of their request (items, weights)
    cw = load_from_disk(str(CONSENSUS_RW)).select_columns(["prompt", "groundtruth_with_release_year", "groundtruth_weights"])
    ok, n_rw, rw_req = True, 0, []
    for b in cw.iter(batch_size=20_000):
        for j in range(len(b["prompt"])):
            x = index[b["prompt"][j][0]["content"]]
            _, gts, w, _ = _ROWS["consensus"][x][0]
            ok &= [(g[0], int(g[1])) for g in b["groundtruth_with_release_year"][j]] == gts and \
                np.allclose([float(v) for v in b["groundtruth_weights"][j]], w)
            rw_req.append(x)
            n_rw += 1
    checks["consensusrw_rows_equal_consensus_of_request"] = bool(ok)
    checks["consensusrw_rows"] = n_rw
    checks["consensusrw_request_counts_equal_n"] = bool(
        (np.bincount(np.array(rw_req), minlength=len(index)) == np.bincount(reqs["single"], minlength=len(index))).all())
    print("[cache] checks", checks, flush=True)
    shards = sorted(SAMPLES.glob("shard*.jsonl"))
    with get_context("fork").Pool(workers) as pool:
        parts = pool.map(_work, shards)
    for d in DATASETS:
        rid = np.concatenate([p["rewards"][d][0] for p in parts])
        arr = np.concatenate([p["rewards"][d][1] for p in parts])
        assert len(rid) == len(reqs[d]) and len(set(rid.tolist())) == len(rid), d
        full = np.zeros((len(reqs[d]), G, R), np.float32)
        full[rid] = arr
        weighted = DATASETS[d][1]
        np.savez_compressed(CACHE / f"rewards_{d}.npz", R=full if weighted else full.astype(np.uint8), req=reqs[d])
        print(f"[cache] saved {d} {full.shape} mean reward {full.mean():.5f}", flush=True)
    rid = np.concatenate([p["union_match"][0] for p in parts])
    mt = np.concatenate([p["union_match"][1] for p in parts])
    full = -np.ones((len(reqs["union"]), G, R), np.int16)
    full[rid] = mt
    # union item responder counts c_i (from the per-responder rows) and consensus weight x n for the same items
    counts = []
    for x in range(len(reqs["union"])):
        _, ugts, _, _ = _ROWS["union"][x][0]
        c = Counter(g for _, gts, _, _ in _ROWS["single"][x] for g in set(gts))
        counts.append([c.get(g, 0) for g in ugts])
    n_x = np.bincount(reqs["single"], minlength=len(reqs["union"]))
    cons_counts = [[w * n_x[x] for w in _ROWS["consensus"][x][0][2]] for x in range(len(reqs["union"]))]
    checks["union_item_counts_equal_consensus_w_times_n"] = all(
        np.allclose(a, b) for a, b in zip(counts, cons_counts))
    pickle.dump({"union_item_counts": counts, "n_x": n_x}, open(CACHE / "union_item_counts.pkl", "wb"))
    np.savez_compressed(CACHE / "union_match.npz", M=full)
    n_lists = sum(p["parse"][0] for p in parts)
    checks["parse"] = {"lists": n_lists, "mean_items": sum(p["parse"][1] for p in parts) / n_lists,
                       "share_lists_with_20_items": sum(p["parse"][2] for p in parts) / n_lists}
    (CACHE / "checks.json").write_text(json.dumps(checks, indent=1))
    print("[cache] done", checks, flush=True)


def load_cache() -> dict:
    out = {}
    for d in DATASETS:
        z = np.load(CACHE / f"rewards_{d}.npz")
        out[d] = (z["R"].astype(np.float32), z["req"])
    return out


# ---------------------------------------------------------------------------------------------------------- analysis
def weighted_moments(Rarr: np.ndarray, w: np.ndarray) -> dict:
    """Per-position mean and std (ddof 0) over (row, list) pairs, row weights w (each list of a row weight w/8)."""
    w = w.astype(np.float64)
    s1 = np.einsum("i,ilk->k", w, Rarr.astype(np.float64)) / (G * w.sum())
    s2 = np.einsum("i,ilk->k", w, Rarr.astype(np.float64) ** 2) / (G * w.sum())
    sd = np.sqrt(np.maximum(s2 - s1 ** 2, 0))
    return {"mean_reward": s1.round(6).tolist(), "std": sd.round(6).tolist(), "std_mean_over_positions": float(sd.mean())}


def simulate(Rarr: np.ndarray, pick_rows, scale: str, rng: np.random.Generator) -> dict:
    stds, scales, zero_share = [], [], []
    for _ in range(N_BATCH):
        b = Rarr[pick_rows(rng)].astype(np.float64)  # (48, 8, 20)
        if scale == "batch":
            sd = b.reshape(-1, R).std(axis=0, ddof=1)  # (20,)
            used = sd[sd > 1e-6]
            stds.append(sd)
        else:
            sd = b.std(axis=1, ddof=1)  # (48, 20)
            used = sd[sd > 1e-6]
            zero_share.append((sd <= 1e-6).mean(axis=0))
            stds.append(np.where(sd > 1e-6, sd, np.nan))
        scales.append(1.0 / (used.mean() + 1e-4) if used.size else 1.0)
    out = {"shaping_scale": summarize(scales)}
    if scale == "batch":
        s = np.array(stds)
        out["batch_std_per_position_mean"] = s.mean(axis=0).round(6).tolist()
        out["batch_std_mean"] = float(s.mean())
    else:
        s = np.concatenate(stds)  # (N_BATCH*48, 20) with NaN for zero-std groups
        out["group_std_nonzero_per_position_mean"] = np.nanmean(s, axis=0).round(6).tolist()
        out["group_std_nonzero_mean"] = float(np.nanmean(s))
        out["share_zero_std_groups_per_position"] = np.array(zero_share).mean(axis=0).round(4).tolist()
        out["share_zero_std_groups_mean"] = float(np.array(zero_share).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.rebuild or not (CACHE / "checks.json").exists():
        build_cache(a.workers)
    cache = load_cache()
    checks = json.load(open(CACHE / "checks.json"))
    req_single = cache["single"][1]
    n_req = len(cache["union"][1])
    n_x = np.bincount(req_single, minlength=n_req)
    res = {"definition": __doc__.split("Definitions", 1)[1].split("Outputs:", 1)[0].strip(), "cache_checks": checks,
           "n_requests": int(n_req), "n_rows": int(len(req_single))}
    try:
        res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    # population grid
    order = np.argsort(req_single, kind="stable")
    starts = np.r_[0, np.cumsum(n_x)[:-1]]
    assert (n_x > 0).all()
    S = np.add.reduceat(cache["single"][0][order], starts, axis=0)  # sum reward: responders hitting each position
    grid = {}
    ones_req = np.ones(n_req)
    grid["per_responder (0/1, one responder per row)"] = {
        "row_sampling": weighted_moments(cache["single"][0], np.ones(len(req_single))),
        "request_sampling": weighted_moments(cache["single"][0], 1.0 / n_x[req_single])}
    grid["mean over responders (count / n)"] = {"row_sampling": weighted_moments(cache["consensus"][0], n_x),
                                                "request_sampling": weighted_moments(cache["consensus"][0], ones_req)}
    grid["bounded sum = union (0/1)"] = {"row_sampling": weighted_moments(cache["union"][0], n_x),
                                         "request_sampling": weighted_moments(cache["union"][0], ones_req)}
    grid["sum (responder count; reference)"] = {"row_sampling": weighted_moments(S, n_x),
                                                "request_sampling": weighted_moments(S, ones_req)}
    grid["per_responder, fixed draw (redditthr)"] = {"request_sampling": weighted_moments(cache["singlethr"][0], ones_req)}
    for m in (2, 4):
        grid[f"mean over m={m} responders (dose-response rows)"] = {
            "row_sampling": weighted_moments(cache[f"meanm{m}"][0], np.ones(len(cache[f"meanm{m}"][1])))}
    res["population_grid"] = grid
    # checks: union = min(1, sum) and mean = sum / n on the same lists
    res["consistency"] = {"share_positions_union_ne_min1_sum": float((cache["union"][0] != np.minimum(S, 1)).mean()),
                          "max_abs_mean_minus_sum_over_n": float(np.abs(cache["consensus"][0] - S / n_x[:, None, None]).max()),
                          "share_positions_mean_ne_sum_over_n":
                              float((np.abs(cache["consensus"][0] - S / n_x[:, None, None]) > 1e-6).mean()),
                          "max_sum_per_position": int(S.max())}
    # arms: simulated training-start batches vs logged shaping scale
    arms = {}
    for arm, (d, sampling, scale) in ARMS.items():
        Rarr = cache[d][0]
        rng = np.random.default_rng(SEED)
        if sampling in ("rows", "requests"):
            n = len(Rarr)
            pick = lambda g, n=n: g.choice(n, size=BATCH_ROWS, replace=False)  # noqa: E731
        else:  # rows_n: a uniform row of the per-responder data names the request
            pick = lambda g: req_single[g.choice(len(req_single), size=BATCH_ROWS, replace=False)]  # noqa: E731
        sim = simulate(Rarr, pick, scale, rng)
        entry = {"reward_dataset": d, "sampling": sampling, "scale": scale, "training_start_simulation": sim,
                 "logs": {}}
        for seed, files in ARM_LOGS.get(arm, {}).items():
            for f in files:
                p = LOGS / f
                if not p.exists():
                    entry["logs"][seed] = {"file": str(p.relative_to(REPO)), "missing": True}
                    continue
                lg = parse_log(p)
                v = lg["values"]
                entry["logs"][seed] = {k: lg[k] for k in ("file", "resumed_from", "n_records", "steps_covered")}
                entry["logs"][seed].update({
                    "first_record_steps_1_10": v[0] if lg["resumed_from"] is None else None,
                    "shaping_scale_over_training": summarize(v),
                    "implied_mean_sigma_over_training (mean of 1/scale - 1e-4)": float(np.mean(1 / np.array(v) - 1e-4)),
                    "first_200_steps_mean": float(np.mean([x for s, x in zip(lg["steps"], v) if s <= 200])),
                    "last_200_steps_mean": float(np.mean([x for s, x in zip(lg["steps"], v) if s > lg["steps"][-1] - 200])),
                })
        s0 = entry["logs"].get("s3407", {}).get("first_record_steps_1_10")
        if s0:
            entry["simulated_over_logged_first_record"] = sim["shaping_scale"]["mean"] / s0
        arms[arm] = entry
        print(f"{arm:30s} sim scale {sim['shaping_scale']['mean']:.3f} logged first {s0}", flush=True)
    res["arms"] = arms
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
