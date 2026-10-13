"""GPT-5.6 zero-shot on the test requests, with the dataset's own instruction (the test "prompt" message,
exactly what crs_eval.py sends to the trained models) and crs_eval's parser and matching.

The API model is a reasoning model; it takes no temperature, so it runs at the API default. max_tokens 2048 covers the
reasoning and the list. Outputs: outputs/crs/test/test_zeroshot_gpt56[_books] (crs_eval format), resumable API cache
outputs/crs/api_cache/n917_e5_gpt56_<dataset>.jsonl, usage totals in summary.json.
Usage: env/bin/python remark/scripts/n917_e5_zeroshot_gpt.py --dataset reddit|books|music [--workers 16] [--limit N]
(music:  Table 2 reference, data/crs/music, output test_zeroshot_gpt56_music)
--year_convention (music only): adds a system message that states the year convention of the music items
(YEAR_SYSTEM, wording of music_sft_targets.py); output test_zeroshot_gpt56_music_yearconv, cache ..._music_yearconv.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n917_common import DATA, REPO, score_and_write  # noqa: E402
from lib.llm import batch_chat, default_model  # noqa: E402

RG = REPO / "remark/third_party/Rank-GRPO"
YEAR_SYSTEM = ("Write each recommendation as \"Artist Name (start_year)\", where start_year is the birth year of a solo "
               "artist or the formation year of a group.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["reddit", "books", "music"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--year_convention", action="store_true")
    a = ap.parse_args()
    if a.dataset == "reddit":
        data, catalog, suffix = DATA, RG / "reddit_catalog/gt_catalog.pkl", ""
    else:
        B = REPO / ("remark/data/crs/reddit_books" if a.dataset == "books" else "remark/data/crs/music")
        data, catalog, suffix = B / "sft_dataset", B / "catalog.pkl", "_" + a.dataset
    ds = load_from_disk(str(Path(data) / "test"))
    ctxs = list(OrderedDict.fromkeys(p[0]["content"] for p in ds["prompt"]))
    if a.limit:
        ctxs = ctxs[:a.limit]
    assert not a.year_convention or a.dataset == "music"
    sysm = [{"role": "system", "content": YEAR_SYSTEM}] if a.year_convention else []
    suffix += "_yearconv" if a.year_convention else ""
    reqs = [{"id": hashlib.sha256(c.encode()).hexdigest()[:20], "messages": sysm + [{"role": "user", "content": c}],
             "max_tokens": 2048, "meta": {"ctx_i": i}} for i, c in enumerate(ctxs)]
    cache = REPO / f"remark/outputs/crs/api_cache/n917_e5_gpt56_{a.dataset}{'_yearconv' if a.year_convention else ''}.jsonl"
    done = batch_chat(reqs, cache_path=cache, workers=a.workers)
    done = {r["id"]: done[r["id"]] for r in reqs if r["id"] in done}  # the cache also holds retry entries
    # an empty answer means the reasoning used the whole budget: retry once with 4,096 tokens (reported)
    empty = [r for r in reqs if r["id"] in done and not (done[r["id"]]["text"] or "").strip()]
    retry = [dict(r, id=r["id"] + ":mt4096", max_tokens=4096) for r in empty]
    done_retry = batch_chat(retry, cache_path=cache, workers=a.workers) if retry else {}
    done_retry = {r["id"]: done_retry[r["id"]] for r in retry if r["id"] in done_retry}
    raw = {}
    for c, r in zip(ctxs, reqs):
        if r["id"] not in done:
            continue
        t = done[r["id"]]["text"] or ""
        if not t.strip() and r["id"] + ":mt4096" in done_retry:
            t = done_retry[r["id"] + ":mt4096"]["text"] or ""
        raw[c] = t
    allres = list(done.values()) + list(done_retry.values())
    usage = {k: sum((x.get("usage") or {}).get(k, 0) or 0 for x in allres)
             for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if a.limit:
        print(f"limit run: {len(raw)}/{len(ctxs)} outputs, usage {usage}")
        return
    summary = score_and_write(ds, raw, catalog, data, REPO / f"remark/outputs/crs/test/test_zeroshot_gpt56{suffix}",
                              extra={"model": f"{default_model()} (API, zero-shot, default sampling, max_tokens 2048)"
                                              + (", system: year convention stated" if a.year_convention else ""),
                                     "system_message": sysm[0]["content"] if sysm else None,
                                     "api_calls": len(done) + len(done_retry), "api_missing": len(ctxs) - len(raw),
                                     "empty_first_answers": len(empty), "retried_4096": len(retry),
                                     "still_empty": sum(not v.strip() for v in raw.values()), "usage": usage,
                                     "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                           cwd=REPO).stdout.strip()})
    print({k: v for k, v in summary.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
