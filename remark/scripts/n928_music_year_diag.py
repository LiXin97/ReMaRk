"""Table 2 (music references): how much of the GPT-5.6 zero-shot gap on music is the year convention.

Music items are "Artist (MusicBrainz begin year)": the birth year of a solo artist, the formation year of a group
(music_common.py). The trained arms learn this convention from rendered items; a zero-shot model may write the year of
the first release instead. For each music test run this reports
  year_only_miss_share  share of recommended lines whose artist name is in the catalogue but never with the written year
  exact_in_catalog      share of lines that are exactly (name, year) in the catalogue
  R@20 (crs_eval truncate, per row) and name-only R@20: a GT artist counts as found when its name appears in the first 20
  lines regardless of year (titles in seen_titles excluded, each GT name counted once). Name-only is a diagnostic, not the
  evaluation protocol. Paired differences of name-only R@20 (ReMaRk and Rank-GRPO minus GPT-5.6) use the paper's bootstrap
  (requests = distinct prompts, 2,000 resamples, seed 42), per row and per request.
Usage: env/bin/python remark/scripts/n928_music_year_diag.py -> outputs/crs/analysis/n928_music_year_diag.json
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import REPO, hits_for, recall_ndcg  # noqa: E402

M = REPO / "remark/data/crs/music"
T = REPO / "remark/outputs/crs/test"
RUNS = {"sft": "test_music_sft_step1500", "rankgrpo": "test_music_2B_single_group_s3407_p0c_ex_step2000",
        "remark": "test_music_2B_union_group_s3407_p0c_ex_step1250", "gpt56": "test_zeroshot_gpt56_music",
        "gpt56_yearconv": "test_zeroshot_gpt56_music_yearconv",
        "knn": "test_knn_prior_k200_music", "popularity": "popularity_train_top20_music"}


def main() -> None:
    ds = load_from_disk(str(M / "sft_dataset/test"))
    catalog = set(tuple(x) for x in pickle.load(open(M / "catalog.pkl", "rb")))
    names = defaultdict(set)
    for n, y in catalog:
        names[n].add(int(y))
    out = {"note": __doc__.split("\n")[0], "runs": {}, "name_only_pairs": {}}
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    groups = list(ctx.values())
    per_row = {}
    for key, run in RUNS.items():
        rd = T / run
        if not (rd / "per_instance.jsonl").exists():
            out["runs"][key] = None
            continue
        lines = exact = year_only = 0
        r20, nr20 = [], []
        for r, it in zip(map(json.loads, open(rd / "per_instance.jsonl")), ds):
            recs = [(t, int(y)) for t, y in r["recs"]]
            seen = set(it["seen_titles"])
            gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
            for n, y in recs:
                lines += 1
                exact += (n, y) in catalog
                year_only += (n, y) not in catalog and n in names
            hits, _ = hits_for(recs, gts, seen, catalog, "truncate")
            r20.append(recall_ndcg(hits, len(gts), 20)[0])
            top = {n for n, _ in recs[:20] if n not in seen}
            nr20.append(len({n for n, _ in gts} & top) / len(gts))
        per_row[key] = np.array(nr20)
        out["runs"][key] = {"run": run, "lines": lines, "exact_in_catalog": exact / lines,
                            "year_only_miss_share": year_only / lines, "R@20": float(np.mean(r20)),
                            "name_only_R@20": float(np.mean(nr20))}
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["runs"][key].items()})
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    sizes = np.array([len(g) for g in groups])
    for a in ("remark", "rankgrpo"):
        if a not in per_row or "gpt56" not in per_row:
            continue
        d = per_row[a] - per_row["gpt56"]
        sums = np.array([d[g].sum() for g in groups]); means = sums / sizes
        row_b = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1); req_b = means[draws].mean(axis=1)
        out["name_only_pairs"][f"{a} - gpt56"] = {
            "row": [float(d.mean()), *map(float, np.percentile(row_b, [2.5, 97.5]))],
            "req": [float(means.mean()), *map(float, np.percentile(req_b, [2.5, 97.5]))]}
        print(a, "- gpt56 name-only", {k: [round(100 * x, 2) for x in v] for k, v in out["name_only_pairs"][f"{a} - gpt56"].items()})
    (REPO / "remark/outputs/crs/analysis/n928_music_year_diag.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
