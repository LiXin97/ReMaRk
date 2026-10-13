"""CPU unit tests: parser, rewards (duplicates, alternatives, order, multi-cluster answers, the identities
between arms), the trainer adapter (line / position mapping, token segmentation, the RB hits and advantages), datasets
(round trip, per-row sampling) and reward speed.

Run (CPU host; nltk from a scratch --target dir, NLTK data from the zips on shared storage):
  PYTHONPATH=<scratch>/r9pkgs NLTK_DATA=remark/data/protoqa/nltk_data env/bin/python remark/scripts/n928_r9_unit.py
or on a node after n928_r9_node_setup.sh: /tmp/qenv/bin/python remark/scripts/n928_r9_unit.py
Output: remark/outputs/protoqa/unit/n928_r9_unit.json (every check with its numbers); exits non-zero on a failure.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "remark"))
from lib.r9_common import (CRS_DATA, OUT, REC_NUM, answers_from_text, line_answers, load_split, nonblank_lines,  # noqa: E402
                           parse_answer, prompt_text)
from lib.r9_rewards import (M, cluster_hit_matrix, greedy_hits, item_rewards, line_cluster_hits,  # noqa: E402
                            make_r9_reward, matches)
from lib.rank_grpo_trl import RankGRPOTrainer, group_advantages, rb_advantages  # noqa: E402
from lib.r9_trainer import R9RankGRPOTrainer  # noqa: E402

RES: dict = {}
FAIL: list = []


def check(name):
    def deco(fn):
        try:
            RES[name] = {"ok": True, **(fn() or {})}
        except Exception as e:  # noqa: BLE001
            RES[name] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-1500:]}
            FAIL.append(name)
        print(f"[{'ok' if RES[name]['ok'] else 'FAIL'}] {name}", flush=True)
        return fn
    return deco


Q0 = {"clusters": [["umbrella"], ["sunscreen"], ["sun hat"], ["sunglasses"], ["cover up"], ["shade"]],
      "counts": [38, 36, 14, 5, 3, 3]}  # train_q0 ("at the beach, name something that might protect you from sun")


def single_vecs(answers, clusters):
    """Reference per-respondent reward per cluster: the gate's list_hits on that cluster alone."""
    return [[h for h, _ in M.list_hits(list(answers), [list(c)])] for c in clusters]


@check("parser")
def _():
    cases = {"1. Dog": "Dog", "2) sun hat": "sun hat", "- **Umbrella**": "Umbrella", "24 hours": "24 hours",
             "7-eleven": "7-eleven", "9 to 5": "9 to 5", "1.5 liters": "1.5 liters", "10.": "", "**": "",
             '"Shade."': "Shade", "  * cover up ": "cover up", "3. - **Sun hat**": "Sun hat", "(4) towel": "towel",
             "Bread/yeast": "Bread/yeast", "</think>": "</think>", "...": "", "**1. Dog**": "Dog", "'Dog'.": "Dog"}
    got = {k: parse_answer(k) for k in cases}
    assert got == cases, {k: (got[k], v) for k, v in cases.items() if got[k] != v}
    text = "\n\n</think>\n\n1. Umbrella\n\n2. Sunscreen\n**\n<think>\nshade\n"
    nb = nonblank_lines(text, None)
    ref = [x for x in text.split("\n") if not RankGRPOTrainer._blank(x)]
    assert nb == ref == ["1. Umbrella", "2. Sunscreen", "**", "shade"], nb
    assert line_answers(text) == ["Umbrella", "Sunscreen", "", "shade"]
    assert answers_from_text(text) == ["Umbrella", "Sunscreen", "shade"]
    return {"cases": len(cases)}


@check("rewards_hand_lists")
def _():
    cl, cnt = Q0["clusters"], Q0["counts"]
    lst = ["Umbrella", "umbrellas", "Sunscreen", "hat", "sun hat", "shades", "Shade", "towel", "cover-up", "sun block"]
    X = np.array([[matches(a, c) for c in cl] for a in lst])
    H = cluster_hit_matrix(lst, cl)
    # duplicates: "umbrellas" matches the umbrella cluster already hit at position 1 -> no reward
    assert X[1, 0] and H[0].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # alternatives / partial: "hat" does not match "sun hat" (score 0.5 rounds to 0); "sun hat" does
    assert not X[3, 2] and H[2, 4] == 1
    single = {c: item_rewards("single", lst, cl, cnt, c) for c in range(len(cl))}
    assert all(single[c] == [float(x) for x in single_vecs(lst, cl)[c]] for c in single)
    mean = item_rewards("mean", lst, cl, cnt, -1)
    exp = [sum(cnt[c] * single[c][k] for c in range(len(cl))) / sum(cnt) for k in range(REC_NUM)]
    assert np.allclose(mean, exp)
    union = item_rewards("union", lst, cl, cnt, -1)
    assert union == [float(h) for h, _ in M.list_hits(lst, cl)]
    # order: the duplicate pair reversed moves the hit to the first of the two
    rev = ["umbrellas", "Umbrella"] + lst[2:]
    assert item_rewards("single", rev, cl, cnt, 0)[:2] == [1.0, 0.0]
    # alternatives inside one cluster ("bread/yeast" -> ["bread/yeast", "bread", "yeast"] by C5)
    cl2 = [["bread/yeast", "bread", "yeast"], ["eggs"], ["flour"]]
    r = item_rewards("single", ["Yeast", "bread", "eggs"], cl2, [5, 3, 2], 0)
    assert r[:3] == [1.0, 0.0, 0.0], r
    multi = [k for k in range(len(lst)) if X[k].sum() >= 2]
    return {"list": lst, "match_matrix": X.astype(int).tolist(), "single_by_cluster": single, "mean": mean,
            "union": union, "answers_matching_two_clusters": [lst[k] for k in multi]}


@check("rewards_identities_random_lists")
def _():
    """On real training questions with lists mixing their cluster strings, near-paraphrases and noise:
    (a) single == list_hits(answers, [c_r]); (c) mean == mean over expanded respondents of (a); (d) union == registered
    greedy list_hits; counts of multi-cluster answers and greedy-vs-anymatch differences."""
    qs = load_split("train")
    rng = random.Random(0)
    pool = [s for q in qs for c in q["clusters"] for s in c]
    n_lists = n_multi = n_diff_union = 0
    for q in rng.sample(qs, 300):
        own = [s for c in q["clusters"] for s in c]
        for _ in range(4):
            cand = [rng.choice([s, s + "s", s.upper(), "the " + s]) for s in own] + rng.sample(pool, 8)
            rng.shuffle(cand)
            lst = cand[:REC_NUM]
            n_lists += 1
            single = [item_rewards("single", lst, q["clusters"], q["counts"], c) for c in range(len(q["clusters"]))]
            ref = single_vecs(lst, q["clusters"])
            assert all(single[c] == [float(x) for x in ref[c]] + [0.0] * (REC_NUM - len(ref[c]))
                       for c in range(len(single)))
            exp = np.repeat(np.array(single), q["counts"], axis=0).mean(axis=0)  # mean over respondents
            assert np.allclose(item_rewards("mean", lst, q["clusters"], q["counts"], -1), exp)
            u = item_rewards("union", lst, q["clusters"], q["counts"], -1)
            assert u[:len(lst)] == [float(h) for h, _ in M.list_hits(lst, q["clusters"])]
            H = cluster_hit_matrix(lst, q["clusters"])
            n_diff_union += int(sum(int(a) != int(b) for a, b in zip(H.max(axis=0), u)))
            n_multi += int(sum(sum(matches(a, c) for c in q["clusters"]) >= 2 for a in lst))
    return {"lists": n_lists, "answers_matching_ge2_clusters": n_multi,
            "union_greedy_vs_anymatch_position_differences": n_diff_union}


@check("reward_function_contract")
def _():
    cl, cnt = Q0["clusters"], Q0["counts"]
    comps = [[{"role": "assistant", "content": "Umbrella\nSunscreen\nSun hat\n"}],
             [{"role": "assistant", "content": "\n</think>\n\n1. Shade\n**\nSunscreen\n" + "towel\n" * 12}],
             [{"role": "assistant", "content": ""}]]
    cols = {"all_clusters": [cl] * 3, "cluster_counts": [cnt] * 3, "resp_cluster": [1, 1, 1], "seen_titles": [[]] * 3,
            "qid": ["q0"] * 3}
    out = {}
    for kind in ("single", "rb", "mean", "union"):
        f = make_r9_reward(kind)
        r = f(prompts=[[{"role": "user", "content": "x"}]] * 3, completions=comps, completion_ids=[[]] * 3, **cols)
        assert len(r) == 3 and all(len(x) == REC_NUM for x in r), kind
        out[kind] = r
    # completion 2: valid items in order: Shade, Sunscreen, towel...; "**" is not an item (re-aligned later)
    assert out["single"][1][:2] == [0.0, 1.0] and out["union"][1][:3] == [1.0, 1.0, 0.0]
    assert out["single"][2] == [0.0] * REC_NUM
    return {"union_stats": make_r9_reward("union").last_stats}


def fake_trainer(rb_table=None, rb_weights="respondents", tok=None) -> R9RankGRPOTrainer:
    t = object.__new__(R9RankGRPOTrainer)
    t.rec_num, t.line_positions = REC_NUM, "lines"
    t.rb_table, t.rb_weights = rb_table, rb_weights
    if rb_table is not None:
        boards = [c for v in rb_table.values() for c in v]
        t.r9_M = max(sum(c) for c in boards) if rb_weights == "respondents" else max(len(c) for c in boards)
    if tok is not None:
        n_vocab = len(tok)
        t._piece = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(n_vocab)]
        t._nl_count = torch.tensor([p.count("\n") for p in t._piece], dtype=torch.long)
        t._piece_text = [tok.decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                         for i in range(n_vocab)]
    return t


TEXTS = ["Umbrella\nSunscreen\nSun hat\nSunglasses\nCover up\nShade\nTowel\nHat\nLotion\nTent",
         "\n\n</think>\n\n1. Umbrella\n\n2. **\n3. Sunscreen\nsunscreen\nShade\n\nshades\nhat\nsun hat\ncover up\nx\nextra1\nextra2",
         "Umbrella\nSunscreen",
         "**\n--\n\n"]


@check("trainer_line_positions")
def _():
    t = fake_trainer()
    cl, cnt = Q0["clusters"], Q0["counts"]
    rows = []
    for text in TEXTS:
        la = line_answers(text)
        for kind, rc in (("single", 0), ("single", 1), ("mean", -1), ("union", -1)):
            per_item = item_rewards(kind, answers_from_text(text), cl, cnt, rc)
            lines = t._to_line_positions(text, per_item)
            # direct: valid line positions receive the per-item rewards in order, malformed / missing lines 0
            pos = [k for k, a in enumerate(la) if a]
            direct = [0.0] * REC_NUM
            for j, k in enumerate(pos):
                direct[k] = per_item[j]
            assert lines == direct, (text, kind, lines, direct)
            if kind in ("single",):
                assert lines == line_cluster_hits(text, cl)[rc].astype(float).tolist()
        lmap = t._line_map(text)
        assert [k for k, j in enumerate(lmap) if j >= 0] == [k for k, a in enumerate(la) if a]
        rows.append({"text": text, "line_answers": la, "line_map": lmap})
    assert R9RankGRPOTrainer._is_item("2. Sunscreen") and not R9RankGRPOTrainer._is_item("2. **")
    assert R9RankGRPOTrainer.ITEM_LINE.match("dog") and not R9RankGRPOTrainer.ITEM_LINE.match("**")
    return {"texts": rows}


@check("trainer_token_segmentation")
def _():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / "data/models/Qwen3.5-2B-Base"))
    t = fake_trainer(tok=tok)
    out = []
    for text in TEXTS:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cid = torch.tensor([ids])
        seg, n_lines = t._segment(cid, torch.ones_like(cid))
        # reconstruct per-line position from the tokens: tokens of the k-th non-blank line carry k (< 10), blank lines
        # -1, everything after the 10th non-blank line 10
        pos_text: dict[int, str] = {}
        for i, p in zip(ids, seg[0].tolist()):
            pos_text[p] = pos_text.get(p, "") + t._piece_text[i]
        nb = nonblank_lines(text, None)
        for k in range(min(len(nb), REC_NUM)):
            assert nb[k].strip() in pos_text[k], (k, nb[k], pos_text.get(k))
        assert int(n_lines[0]) == len(nb), (n_lines, len(nb))
        if len(nb) > REC_NUM:
            assert REC_NUM in pos_text and nb[REC_NUM].strip() in pos_text[REC_NUM]
        out.append({"n_nonblank": len(nb), "n_lines_seg": int(n_lines[0]),
                    "positions": sorted(p for p in pos_text)})
    return {"texts": out}


@check("rb_hits_and_advantages")
def _():
    from accelerate import PartialState
    PartialState()  # single-process state for accelerate's gather
    G = 8
    qs = load_split("train")
    rng = random.Random(1)
    picked = rng.sample([q for q in qs if len(q["clusters"]) >= 3], 6)
    table = {}
    for q in qs:  # the launcher's table: every training board, keyed by prompt text
        table.setdefault(prompt_text(q["question"]), []).append(q["counts"])
    tr = fake_trainer(table, "respondents")
    tc = fake_trainer(table, "clusters")
    prompts, comps, inputs = [], [], []
    pool = [s for q in qs for c in q["clusters"] for s in c]
    for q in picked:
        own = [c[0] for c in q["clusters"]]
        for _ in range(G):
            lst = rng.sample(own, min(len(own), rng.randint(1, len(own)))) + rng.sample(pool, 6)
            rng.shuffle(lst)
            text = "\n".join(lst[:REC_NUM + 1]) if rng.random() < 0.3 else "\n".join(lst[:REC_NUM])
            prompts.append([{"role": "user", "content": prompt_text(q["question"])}])
            comps.append([{"role": "assistant", "content": text}])
            inputs.append({"qid": q["qid"], "all_clusters": q["clusters"], "cluster_counts": q["counts"]})
    tr._rb_hits(prompts, comps, inputs, "cpu")
    tc._rb_hits(prompts, comps, inputs, "cpu")
    adv_r, used_r = rb_advantages(tr._rb_rewards, tr._rb_mask, G)
    adv_c, used_c = rb_advantages(tc._rb_rewards, tc._rb_mask, G)
    assert torch.allclose(adv_r, adv_c, atol=1e-5), (adv_r - adv_c).abs().max()
    # brute force: per-respondent group-scaled advantages with arm (a)'s reward at line positions, averaged
    ref = torch.zeros(len(prompts), REC_NUM)
    stds = []
    for gi, q in enumerate(picked):
        rows = range(gi * G, (gi + 1) * G)
        for c, n in enumerate(q["counts"]):
            f = make_r9_reward("single")
            per_item = f(completions=[comps[i] for i in rows], all_clusters=[q["clusters"]] * G,
                         cluster_counts=[q["counts"]] * G, resp_cluster=[c] * G)
            lines = torch.tensor([tr._to_line_positions(comps[i][0]["content"], r) for i, r in zip(rows, per_item)])
            a, s = group_advantages(lines, G)
            ref[gi * G:(gi + 1) * G] += n * a / sum(q["counts"])
            stds += [s[0]] * n
    assert torch.allclose(adv_r, ref, atol=1e-5), (adv_r - ref).abs().max()
    std_resp = torch.stack(stds).reshape(-1)
    assert torch.allclose(used_r.sort().values, std_resp.sort().values, atol=1e-6)
    sr = used_r[used_r > 1e-6].mean().item()
    sc = used_c[used_c > 1e-6].mean().item()
    return {"groups": len(picked), "M_respondents": tr.r9_M, "M_clusters": tc.r9_M,
            "max_abs_adv_diff_respondents_vs_clusters": float((adv_r - adv_c).abs().max()),
            "max_abs_adv_diff_vs_brute_force": float((adv_r - ref).abs().max()),
            "mean_nonzero_std_respondent_rows (rb shaping scale basis)": sr,
            "mean_nonzero_std_cluster_rows": sc}


@check("datasets_round_trip_and_sampling")
def _():
    from datasets import load_from_disk
    sys.path.insert(0, str(REPO / "remark/scripts"))
    from n928_r9_build import BASE, Features, row, save
    qs = load_split("validation")[:5]
    with tempfile.TemporaryDirectory() as d:
        rows = [row(q, c) for q in qs for c, n in enumerate(q["counts"]) for _ in range(n)]
        save(rows, Path(d) / "x", Features(BASE))
        back = load_from_disk(str(Path(d) / "x"))
        assert len(back) == sum(q["n"] for q in qs) and back[0] == rows[0] and back[-1] == rows[-1]
        f = make_r9_reward("single")
        ex = back.select(range(3))
        r = f(completions=[[{"role": "assistant", "content": "\n".join(c[0] for c in e["all_clusters"])}] for e in ex],
              **{k: ex[k] for k in ex.column_names if k != "prompt"})
        assert all(sum(x) == 1.0 for x in r)  # each respondent's cluster is named exactly once in the list
    out = {}
    for name in ("grpo/train", "grpo_union/train"):
        ds = load_from_disk(str(CRS_DATA / name))
        out[name] = {"rows": len(ds), "columns": ds.column_names}
        e = ds[0]
        assert e["prompt"][0]["role"] == "user" and e["prompt"][0]["content"].startswith("We asked 100 people")
        assert e["seen_titles"] == [] and sum(e["cluster_counts"]) == e["n_respondents"]
        if name == "grpo/train":
            assert e["groundtruth_with_release_year"] == [e["all_clusters"][e["resp_cluster"]]]
    # TRL's sampler on the per-row split: questions in proportion to respondent count
    from trl.trainer.utils import RepeatSampler
    ds = load_from_disk(str(CRS_DATA / "grpo/train"))
    qid = ds.data.column("qid").to_pylist()  # (ds["col"] is a lazy Column in datasets 4; per-row indexing is slow)
    nresp = ds.data.column("n_respondents").to_pylist()
    sampler = RepeatSampler(range(len(ds)), mini_repeat_count=8, batch_size=48, repeat_count=1, seed=3407)
    it = iter(sampler)
    drawn = [next(it) for _ in range(250 * 48 * 8)][::8]  # 250 steps x 48 prompts
    n = np.array([nresp[i] for i in drawn], dtype=float)
    per_q = {}
    for i in range(len(ds)):
        per_q.setdefault(qid[i], nresp[i])
    nq = np.array(list(per_q.values()), dtype=float)
    out["sampler"] = {"draws": len(drawn), "mean_n_of_drawn_questions": float(n.mean()),
                      "expected_row_sampling (sum n^2 / sum n)": float((nq ** 2).sum() / nq.sum()),
                      "expected_uniform (mean n)": float(nq.mean()),
                      "distinct_questions": len({qid[i] for i in drawn})}
    return out


@check("sft_dataset")
def _():
    from datasets import load_from_disk
    p = CRS_DATA / "sft_dataset"
    if not (p / "train").exists():
        return {"skipped": "sft_dataset not built yet"}
    out = {}
    for s in ("train", "validation", "test", "test_crowdsourced"):
        ds = load_from_disk(str(p / s))
        out[s] = len(ds)
        e = ds[0]
        assert e["prompt"][0]["content"] == prompt_text(e["question"])
        if s in ("train", "validation"):
            c = e["completion"][0]["content"]
            assert e["completion"][0]["role"] == "assistant"
            assert answers_from_text(c, None) == c.split("\n")  # targets are fixed points of the parser
            if s == "train":
                assert len(c.split("\n")) == REC_NUM
    return out


@check("reward_speed_384_lists")
def _():
    qs = load_split("train")
    rng = random.Random(42)
    pool = [s for q in qs for c in q["clusters"] for s in c]
    batch = []
    for q in rng.sample(qs, 48):
        own = [c[0] for c in q["clusters"]]
        for _ in range(8):
            lst = own + rng.sample(pool, 12)
            rng.shuffle(lst)
            batch.append((q, "\n".join(lst[:REC_NUM])))
    out = {}
    for kind in ("single", "mean", "union"):
        f = make_r9_reward(kind)
        M.cache_clear()
        t0 = time.time()
        f(completions=[[{"role": "assistant", "content": t}] for _, t in batch],
          all_clusters=[q["clusters"] for q, _ in batch], cluster_counts=[q["counts"] for q, _ in batch],
          resp_cluster=[0] * len(batch))
        out[f"{kind}_cold_s"] = round(time.time() - t0, 3)
    t0 = time.time()
    for q, t in batch:
        line_cluster_hits(t, q["clusters"])
    out["rb_cluster_hits_warm_s"] = round(time.time() - t0, 3)
    return out


if __name__ == "__main__":
    out = OUT / "unit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "n928_r9_unit.json").write_text(json.dumps({"failed": FAIL, "checks": RES}, indent=1, default=str))
    print(json.dumps({"failed": FAIL, "n_checks": len(RES)}))
    sys.exit(1 if FAIL else 0)
