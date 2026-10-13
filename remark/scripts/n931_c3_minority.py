"""C3 (registered 2026-10-04 15:06): minority answers on test.

Per request (rows sharing a prompt): union items U1 (named by exactly one responder) and U2 (named by >= 2), after the
crs_eval convention for seen titles (an item whose title is in seen_titles is never counted). The model's list for a
request is its first row's list (greedy decoding: every row of a request gets the same list); an item counts as recalled
when crs_eval's truncate matcher matches it within the first 20 parsed items. Per arm: recall on U1 and on U2 (mean over
requests with a non-empty class); ReMaRk - baseline per class with a paired bootstrap over requests (2,000, seed 42).
Also ReMaRk - baseline per-row R@20 within tertiles of the request's agreement share |U2| / |U| (cut points = tertiles
of the test requests' shares; ties keep all requests at a cut point in the lower tertile).
Output: outputs/crs/analysis/n931_c3_minority.json
Usage: env/bin/python remark/scripts/n931_c3_minority.py
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, REPO, RG, hits_for, recall_ndcg  # noqa: E402

T = REPO / "remark/outputs/crs/test"
BOOKS, MUSIC = REPO / "remark/data/crs/reddit_books", REPO / "remark/data/crs/music"
PAIRS = {  # setting -> (dataset, ReMaRk test run, {baseline label: test run})
    "2B": ("reddit", "test_2B_union_group_step2000", {"rankgrpo": "test_2B_single_group_s3407_step2000"}),
    "4B": ("reddit", "test_4B_union_group_s3407_step2000", {"rankgrpo": "test_4B_single_group_s3407_step2000"}),
    "9B": ("reddit", "test_9B_union_group_s3407_p0c_step2000", {"rankgrpo": "test_9B_single_group_s3407_step2000"}),
    "books2B": ("books", "test_books_union_group_p0d_step1500",
                {"rankgrpo": "test_books_single_group_p0d_step2000", "perresp_batch": "test_books_single_batch_p0d_step2000"}),
    "music2B": ("music", "test_music_2B_union_group_s3407_p0c_ex_step1250",
                {"rankgrpo": "test_music_2B_single_group_s3407_p0c_ex_step2000"}),
}


def load(ds_key: str):
    if ds_key == "reddit":
        return load_from_disk(str(DATA / "test")), set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    B = BOOKS if ds_key == "books" else MUSIC
    return load_from_disk(str(B / "sft_dataset/test")), set(tuple(x) for x in pickle.load(open(B / "catalog.pkl", "rb")))


def boot_ci(diff_per_req: np.ndarray, rng) -> list[float]:
    n = len(diff_per_req)
    b = diff_per_req[rng.integers(0, n, size=(2000, n))].mean(1)
    return [float(diff_per_req.mean()), *map(float, np.percentile(b, [2.5, 97.5]))]


def main() -> None:
    out = {"note": __doc__.split("\n")[0], "settings": {}}
    for name, (ds_key, rm, bases) in PAIRS.items():
        ds, cat = load(ds_key)
        req = defaultdict(list)
        for i, p in enumerate(ds["prompt"]):
            req[p[0]["content"]].append(i)
        reqs = list(req.values())
        # union item classes per request
        U1, U2, share = [], [], []
        for rows in reqs:
            seen = set(ds[rows[0]]["seen_titles"])
            cnt = Counter()
            for i in rows:
                items = {(g[0], int(g[1])) for g in ds[i]["groundtruth_with_release_year"] if g[0] not in seen}
                cnt.update(items)
            u1 = [it for it, c in cnt.items() if c == 1]; u2 = [it for it, c in cnt.items() if c >= 2]
            U1.append(u1); U2.append(u2); share.append(len(u2) / max(1, len(u1) + len(u2)))
        share = np.array(share)
        cuts = np.quantile(share, [1 / 3, 2 / 3])
        tert = np.digitize(share, cuts, right=True)  # 0, 1, 2

        def class_recall(run: str):
            rows = [json.loads(l) for l in open(T / run / "per_instance.jsonl")]
            r1, r2, r20 = [], [], np.zeros(len(ds))
            for k, rws in enumerate(reqs):
                recs = [(t, int(y)) for t, y in rows[rws[0]]["recs"]]
                seen = set(ds[rws[0]]["seen_titles"])
                for cls, store in ((U1[k], r1), (U2[k], r2)):
                    if not cls:
                        store.append(np.nan); continue
                    hits, matched = hits_for(recs, cls, seen, cat, "truncate")
                    got = {gi for gi, pos in matched if pos < 20}
                    store.append(len(got) / len(cls))
                for i in rws:
                    gts = [(g[0], int(g[1])) for g in ds[i]["groundtruth_with_release_year"]]
                    hits, _ = hits_for([(t, int(y)) for t, y in rows[i]["recs"]], gts, set(ds[i]["seen_titles"]), cat, "truncate")
                    r20[i] = recall_ndcg(hits, len(gts), 20)[0]
            return np.array(r1), np.array(r2), r20

        rng = np.random.default_rng(42)
        res = out["settings"][name] = {"dataset": ds_key, "requests": len(reqs),
                                       "union_items_named_once_share": float(sum(map(len, U1)) / max(1, sum(map(len, U1)) + sum(map(len, U2)))),
                                       "requests_with_U2": int(sum(bool(u) for u in U2)), "tertile_cuts": cuts.tolist()}
        a1, a2, ar = class_recall(rm)
        res["remark"] = {"recall_U1": float(np.nanmean(a1)), "recall_U2": float(np.nanmean(a2))}
        for lab, br in bases.items():
            b1, b2, br20 = class_recall(br)
            m1, m2 = ~np.isnan(a1) & ~np.isnan(b1), ~np.isnan(a2) & ~np.isnan(b2)
            d_row = ar - br20
            tert_res = {}
            for t in range(3):
                idx = [k for k in range(len(reqs)) if tert[k] == t]
                if not idx:  # tied shares can leave a tertile empty (e.g. music: most requests have no shared item)
                    tert_res[["low", "mid", "high"][t]] = {"requests": 0}; continue
                sums = np.array([d_row[reqs[k]].sum() for k in idx]); sizes = np.array([len(reqs[k]) for k in idx])
                draws = rng.integers(0, len(idx), size=(2000, len(idx)))
                bt = sums[draws].sum(1) / sizes[draws].sum(1)
                tert_res[["low", "mid", "high"][t]] = {"requests": len(idx), "rows": int(sizes.sum()),
                                                        "share_range": [float(share[idx].min()), float(share[idx].max())],
                                                        "row_R@20_diff": [float(sums.sum() / sizes.sum()), *map(float, np.percentile(bt, [2.5, 97.5]))]}
            res[lab] = {"recall_U1": float(np.nanmean(b1)), "recall_U2": float(np.nanmean(b2)),
                        "diff_U1": boot_ci(a1[m1] - b1[m1], rng), "diff_U2": boot_ci(a2[m2] - b2[m2], rng),
                        "by_agreement_tertile": tert_res}
            print(name, lab, "U1", [round(100 * x, 2) for x in res[lab]["diff_U1"]], "U2",
                  [round(100 * x, 2) for x in res[lab]["diff_U2"]], flush=True)
    (REPO / "remark/outputs/crs/analysis/n931_c3_minority.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
