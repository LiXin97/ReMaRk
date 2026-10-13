"""Assemble remark/data/protoqa/gate_check.json from the gate's result files.

Inputs (all written by the gate scripts, no GPU / API):
  remark/data/protoqa/r9_stats.json        (n928_r9_stats.py, env/bin/python)
  remark/data/protoqa/r9_matcher_test.json (n928_r9_matcher_test.py, scratch venv with the official evaluator)
  remark/data/protoqa/splits/*.jsonl       (n928_r9_stats.py)
Usage: env/bin/python remark/scripts/n928_r9_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "remark/data/protoqa"
P = "remark/data/protoqa/"


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()]


def main() -> None:
    st = json.loads((BASE / "r9_stats.json").read_text())
    mt = json.loads((BASE / "r9_matcher_test.json").read_text())
    sp = {k: jl(BASE / f"splits/{k}.jsonl") for k in ("train", "validation", "test_scraped", "test_crowdsourced")}
    pf = st["per_file"]

    # extra numbers from the split files
    extra = {}
    for k, rows in sp.items():
        n = np.array([r["n_respondents"] for r in rows], float)
        c = np.array([r["n_clusters"] for r in rows])
        p_row = n / n.sum()  # Rank-GRPO row sampling: a question's probability is proportional to its respondents
        extra[k] = {"share_questions_le_10_clusters": round(float((c <= 10).mean()), 4),
                    "row_sampling_prob_over_uniform_p5_p95": [round(float(np.percentile(p_row * len(n), q)), 3)
                                                              for q in (5, 95)],
                    "row_sampling_prob_over_uniform_min_max": [round(float(p_row.min() * len(n)), 3),
                                                               round(float(p_row.max() * len(n)), 3)]}
    sizes = st["splits"]
    B = mt["B_single_string_damage"]
    A = mt["A_official_dev_crowdsourced"]
    C = mt["C_agreement"]
    D = mt["D_list_level"]
    G = mt["G_training_step_speed"]

    gate = {
        "node": " (ProtoQA generality branch), gate investigation only (no GPU, no training, no GPT calls)",
        "date": "2026-10-02",
        "data": {"repo": "https://github.com/iesl/protoqa-data", "commit": st["data_git_commit"],
                 "local": P + "raw"},
        "items": {
            "1_licence": {
                "status": "pass",
                "finding": "The data repository is licensed CC BY 4.0, which permits research use, modification and "
                           "redistribution with attribution. DATASTATEMENT.md states no restriction; its 'Uses' "
                           "section only cautions against free-form interactive commonsense-QA deployment without "
                           "studying biases. The evaluator is a separate Apache-2.0 repository.",
                "quotes": {
                    "LICENSE line 1": "Attribution 4.0 International",
                    "DATASTATEMENT.md (Distribution)": "We use CC-BY-4.0; see LICENSE .",
                    "DATASTATEMENT.md (third parties)": "Have any third parties imposed IP-based or other "
                                                        "restrictions on the data associated with the instances?: "
                                                        "Not at this time.",
                    "DATASTATEMENT.md (Uses, caution not restriction)": "We caution regarding free-form use of this "
                        "dataset for interactive \"commonsense question answering\" purposes without more study of "
                        "the biases and stereotypes learned by such models.",
                    "DATASTATEMENT.md (collection)": "Scraped data was acquired through fan transcriptions at "
                        "https://www.familyfeudinfo.com and http://familyfeudfriends.arjdesigns.com/",
                },
                "unverified": ["The CC BY 4.0 grant covers the authors' compilation; rights of the underlying "
                               "show content / fan-site transcriptions are not addressed by the repository "
                               "(not checked; ordinary for a published research dataset)."],
                "obligations": ["cite Boratko et al., EMNLP 2020 and attribute the dataset (CC BY 4.0)",
                                "do not redistribute modified data without marking the changes (our cleaned "
                                "splits stay internal)"],
                "evidence": [P + "raw/LICENSE", P + "raw/DATASTATEMENT.md", P + "raw/README.md"],
            },
            "2_per_respondent_information": {
                "status": "pass",
                "finding": "Every train / dev.scraped / dev.crowdsourced line has metadata, question {original, "
                           "normalized}, answers {raw, clusters{id: {count, answers}}}, num {answers, clusters}; "
                           "num.answers equals the sum of cluster counts on every line. Cluster counts are numbers "
                           "of respondents in a 100-person survey (Family Feud boards; the data statement drops "
                           "boards 'when answers totaled more than 100'). Scraped totals are mostly < 100 because a "
                           "board lists only its top answers: respondents whose answer is not on the board are "
                           "unobserved. Expanding counts gives respondents with exactly one answer each. In the "
                           "scraped files 'raw' is an exact copy of the clusters (one string per cluster, no extra "
                           "per-respondent strings). Only dev.crowdsourced has genuine raw per-respondent answers "
                           "(99-101 per question, ~49 distinct strings per question, manually clustered).",
                "numbers": {
                    "rows_in_files": st["file_rows"],
                    "readme_counts": st["readme_counts"],
                    "readme_vs_file_discrepancy": "files have one more row than README states for train (8,782 vs "
                                                  "8,781), dev.scraped (980 vs 979), dev.crowdsourced (52 vs 51); "
                                                  "every line parses and ids are unique (not explained)",
                    **{f: {k: pf[f][k] for k in (
                        "questions", "raw_identical_to_clusters", "strings_per_cluster_mean",
                        "clusters_per_question", "clusters_per_question_hist", "respondents_per_question(sum_of_counts)",
                        "total_eq_100", "total_lt_100", "total_gt_100", "total_lt_50", "questions_ge2_clusters",
                        "questions_ge2_positive_clusters", "questions_ge2_respondents", "clusters_with_count_le_0",
                        "questions_with_count_le_0_cluster", "strings_with_slash", "strings_all_stopwords")}
                       for f in ("train", "dev_scraped", "dev_crowdsourced")},
                    "share_ge2_clusters": {"train": round(pf["train"]["questions_ge2_clusters"] / 8782, 4),
                                           "dev_scraped": round(pf["dev_scraped"]["questions_ge2_clusters"] / 980, 4),
                                           "dev_crowdsourced": 1.0},
                    "dev_crowdsourced_raw": "raw totals 99-101 (mean 99.85) vs cluster totals mean 93.96: manual "
                                            "cleaning removed ~6 nonsense answers per question; 94.1% of raw answer "
                                            "mass has its string inside a cluster",
                },
                "anomalies_and_cleaning_rule": {
                    "C1_total_gt_100": "train 112, dev.scraped 14 boards sum to > 100: synonym explosions of one "
                                       "answer listed as separate entries with the same count (e.g. train_q9295: 18 "
                                       "entries x 62; train_q7747 sums to 1,116); dropped",
                    "C2_count_le_0": "train 117 clusters in 100 questions, dev.scraped 13 in 12; dropped",
                    "C3": "keep questions with >= 2 positive clusters",
                    "C4": "drop train questions whose text equals or is a near duplicate (content-token Jaccard >= "
                          "0.8) of any held-out or crowdsourced question",
                    "C5_slash": "1,975 train / 234 dev.scraped cluster strings contain '/' ('bread/yeast'); the "
                                "official matcher treats the whole string as one token, so 'bread' does not match; "
                                "targets keep the original string plus its '/' parts (target data change only)",
                    "secondary_flag_unhandled": f"{st['train_total_le_100_with_>=3_clusters_sharing_a_count']} train "
                                                "boards with total <= 100 have >= 3 clusters sharing one count (ties "
                                                "or smaller synonym explosions; not removed)",
                    "incomplete_boards": "train has 234 boards with total < 50 (217 after cleaning): correct answers "
                                         "missing from the board are scored as misses",
                    "cleaned_counts": st["cleaning"],
                },
                "evidence": [P + "raw/DATAFORMAT.md", P + "r9_stats.json", "remark/scripts/n928_r9_stats.py"],
            },
            "3_answer_matching_reward": {
                "status": "pass (matcher); uncertain (target quality of the scraped training data)",
                "official_rule": "iesl/protoqa-evaluator @" + mt["evaluator"]["commit"][:7] + " (not on PyPI). "
                    "Default CLI similarity is 'wordnet' (exact_match optional). Prediction normalised by "
                    "lower()[:50].strip(); target strings used as stored. wordnet_score: nltk.word_tokenize, NLTK "
                    "English stopwords removed, all order-preserving partitions of the tokens into contiguous groups "
                    "on both sides, two groups match iff equal strings or wn.synsets(group with '_') share a synset "
                    "(morphy gives lemmatisation: peach~peaches, fly~flying, booing~boo), partition-pair score = max "
                    "matching / max(#groups), string score = max over partition pairs, cluster score = max over its "
                    "strings, then np.round (half to even: 0.5 -> 0, so 'red car' does not match 'car'). Answers "
                    "are assigned to clusters by Munkres on count-weighted 0/1 scores; Max Answers@k = matched "
                    "counts among the first k answers / best possible with k answers; Max Incorrect@k reads answers "
                    "until k misses. All-stopword strings collapse to the empty group, so 'the' matches 'it' "
                    "(official behaviour, reproduced).",
                "fast_matcher": "remark/scripts/n928_r9_matcher.py: the same rule re-implemented with per-string "
                                "caches (tokens, synsets, partitions) and an early exit when no token span pair "
                                "can match; reward = list_hits: positions in rank order, a position hits if its "
                                "answer matches a not-yet-credited cluster (each cluster once, as Rank-GRPO's "
                                "matcher consumes each ground-truth item once). Per-respondent reward of respondent r "
                                "= hit on r's cluster; union reward = hit on any cluster.",
                "numbers": {
                    "agreement_pairs": C["pairs"], "official_positive": C["official_positive"],
                    "fast_positive": C["fast_positive"], "disagreements": C["disagreements"],
                    "list_level_max_abs_diff_in_official_munkres": D["max_abs_diff_official_vs_fast_in_official_munkres"],
                    "greedy_reward_vs_munkres_max_answers": D["greedy_reward_vs_munkres_max_answers"],
                    "answers_matching_ge2_clusters": D["answers_matching_ge2_clusters"],
                    "speed_pairs_per_s_one_core": C["speed_pairs_per_s"],
                    "per_step_384_lists_x_10": {"pairs": G["pairs_per_step_upper_bound"],
                                                "official_s": G["official_s_per_step_extrapolated_from_48_lists"],
                                                "fast_cold_cache_s": G["fast_list_hits_s_per_step_cold_cache"],
                                                "fast_warm_cache_s": G["fast_list_hits_s_per_step_warm_cache"]},
                    "official_on_shipped_dev_crowdsourced_predictions_wordnet": {
                        "human": {k: v for k, v in A["human_full_clusters"].items() if k.startswith("wordnet")},
                        "gpt2_finetuned": {k: v for k, v in A["gpt2_full_clusters"].items()
                                           if k.startswith("wordnet")}},
                    "single_string_target_damage(dev.crowdsourced, one string per cluster = its most frequent raw "
                    "answer)": {
                        "answers_matched_share_human_full_vs_single": [B["human_full"]["share"],
                                                                       B["human_single"]["share"]],
                        "answers_matched_share_gpt2_full_vs_single": [B["gpt2_full"]["share"],
                                                                      B["gpt2_single"]["share"]],
                        "per_respondent_top10_human_full_vs_single": [B["per_respondent_hit_share_top10"]["human_full"],
                                                                      B["per_respondent_hit_share_top10"]["human_single"]],
                        "per_respondent_top10_gpt2_full_vs_single": [B["per_respondent_hit_share_top10"]["gpt2_full"],
                                                                     B["per_respondent_hit_share_top10"]["gpt2_single"]],
                        "wordnet_max_answers_10_human_full_vs_single": [
                            A["human_full_clusters"]["wordnet | Max Answers - 10"],
                            A["human_single_string_clusters"]["wordnet | Max Answers - 10"]],
                        "wordnet_max_answers_10_gpt2_full_vs_single": [
                            A["gpt2_full_clusters"]["wordnet | Max Answers - 10"],
                            A["gpt2_single_string_clusters"]["wordnet | Max Answers - 10"]]},
                },
                "interpretation": "The matcher is usable as a fast deterministic 0/1 reward: 0 disagreements with "
                    "the official WordNet score on 184,420 pairs, identical list scores inside the official "
                    "Munkres, and < 0.3 s per 384-list step on one cold core (official ~3.3 s). The weak point is "
                    "the targets, not the matcher: train and dev.scraped keep one string per cluster, and on the "
                    "crowdsourced questions reducing clusters to one string removes about 45% of the answer matches "
                    "(human answers 47.9% -> 26.2%) and lowers per-respondent top-10 coverage from 0.72 to 0.57. "
                    "The training reward therefore has many false negatives (the same for both arms), as the "
                    "title-equality reward does in the recommendation data. DATAFORMAT.md itself says the scraped "
                    "dev set 'is not intended for use with the generative evaluation'.",
                "unverified": ["the shipped human / GPT-2 dev numbers above were not compared with the paper's "
                               "tables", "the single-string simulation picks the most frequent raw string; the "
                               "scraped data keep the show's board label, which may match more or fewer answers",
                               "nltk and the WordNet data are not in env/ or (unchecked) /tmp/qenv; the reward "
                               "needs `pip install nltk` plus wordnet / stopwords / punkt_tab data on the nodes "
                               "(fetched here with curl from raw.githubusercontent.com/nltk/nltk_data because "
                               "nltk.download refuses the proxied egress)"],
                "evidence": [P + "r9_matcher_test.json", P + "r9_hand_lists.json", "remark/scripts/n928_r9_matcher.py",
                             "remark/scripts/n928_r9_matcher_test.py",
                             "scratchpad r9_evaluator/src/protoqa_evaluator/{scoring,evaluation,common_evaluations,"
                             "data_processing}.py"],
            },
            "4_splits": {
                "status": "pass",
                "finding": "Official test has questions only, so all scored splits come from the released answer "
                           "files: train = cleaned train.jsonl minus near duplicates of any held-out or "
                           "crowdsourced question; dev.scraped cleaned and shuffled with random.Random(42) over "
                           "sorted ids: 400 validation, 558 test_scraped; dev.crowdsourced (52, multi-string "
                           "clusters, ~100 real respondents) as a second test set in the task's intended format. "
                           "Every question in every split has >= 2 clusters (>= 2 distinct answers, which is "
                           "what makes the per-respondent and union rewards differ). Gate threshold (>= 5k training "
                           "questions with >= 2 respondents) is met: 8,581.",
                "numbers": {"sizes": sizes, "split_extras": extra, "overlap": {
                    k: {kk: vv for kk, vv in v.items() if kk != "examples_ge_0.8"} if isinstance(v, dict) else v
                    for k, v in st["overlap"].items()}},
                "notes": ["exact normalised-text overlap train vs dev.scraped, train vs the 154 crowdsourced "
                          "questions, and dev.scraped vs crowdsourced: 0; content-token Jaccard >= 0.8: 70 train "
                          "questions near 59 dev.scraped ones (26 pairs identical up to stopwords / order), 5 "
                          "train vs crowdsourced, 1 dev.scraped vs crowdsourced (r2q7 ~ dev-scraped_q9403, 0.833, "
                          "left in place); among the 70 train/dev near pairs the answer-string Jaccard has median "
                          "0.18 (15 pairs >= 0.5, i.e. the same survey); C4 removes 71 train questions",
                          "omitted.jsonl (29 questions the authors removed from training): 0 ids and 0 texts in "
                          "train",
                          "test_crowdsourced has only 52 questions: paired bootstrap CIs will be wide; it is "
                          "secondary to test_scraped (558)",
                          "Rank-GRPO's row sampling picks a question with probability proportional to its "
                          "respondents; totals are nearly constant (train p5-p95 of prob/uniform shown in "
                          "split_extras), so on ProtoQA row sampling is close to request sampling and the ProtoQA "
                          "comparison mainly tests the reward (union vs one respondent's answer)",
                          "with <= 10 clusters in nearly every scraped question, per-respondent coverage@10 and "
                          "Max Answers@10 differ only by the Munkres-vs-greedy assignment; they separate at k = 1, "
                          "3, 5"],
                "evidence": [P + "splits/manifest.json", P + "splits/train.jsonl", P + "splits/validation.jsonl",
                             P + "splits/test_scraped.jsonl", P + "splits/test_crowdsourced.jsonl",
                             P + "r9_stats.json"],
            },
            "5_sft_targets": {
                "status": "pass (plan); not run",
                "finding": "One GPT-5.6 call per training question, prompt = the question only (no answers, so SFT "
                           "never sees the clusters or their union): 8,581 calls, plus 400 for the validation split "
                           "(crs_train_sft.py loads a 'validation' split with completions) = 8,981, ~9.4k with a "
                           "5% retry allowance (ProtoQA budget: ~10k). List length k = 10: ProtoQA's deepest cut is "
                           "Max Answers@10, questions have 5.0 clusters on average (99%+ have <= 10), so a 20-line "
                           "list would hold >= 15 guaranteed misses, and a 10-line list keeps Max Incorrect@1/3/5 "
                           "defined. Ask for 10 distinct answers ranked by how many of 100 surveyed people would "
                           "give them, one short answer per line.",
                "numbers": {"calls_train": sizes["train"]["questions"], "calls_validation": sizes["validation"]["questions"],
                            "calls_total": sizes["train"]["questions"] + sizes["validation"]["questions"],
                            "token_estimate": "unverified: Reddit-Books SFT-target calls averaged ~283 prompt and "
                                              "~1,162 completion tokens (860 reasoning) per call "
                                              "(remark/outputs/books/api_usage.json); a ProtoQA call has a shorter "
                                              "prompt and a 10-word answer, so expect well under that per call",
                            "k": 10},
                "risks": ["memorisation: the questions and boards are on public fan sites since 1976; GPT-5.6 and "
                          "the Qwen3.5 base may know the boards (project policy requires reporting this). Measure: teacher "
                          "lists' per-respondent coverage@10 on the 400 validation questions vs a zero-shot Qwen3.5 "
                          "base; it affects headroom for both arms, not the union vs per-respondent comparison",
                          "the teacher may emit paraphrases of one cluster; SFT targets should be de-duplicated with "
                          "the matcher against each other (not against the answers)"],
                "evidence": ["remark/scripts/books_sft_targets.py (pattern: one call per training thread)",
                             "remark/scripts/crs_train_sft.py:46-47", "remark/outputs/books/api_usage.json"],
            },
            "6_format": {
                "status": "pass with code (new files only)",
                "what_depends_on_title_year": {
                    "parser": "third_party/Rank-GRPO/libs/utils.py:62 process_rec_raw pattern (.+?)\\s+\\((\\d{4})\\) "
                              "(also strips numbering :26 and parentheses :18)",
                    "trainer": "lib/rank_grpo_trl.py:195 ITEM_LINE (same regex; used in 'items' mode _segment "
                               ":242,:251), :206 _is_item (calls process_rec_raw; used by P0c line re-alignment, the "
                               "malformed share :143 and exact_requires_items), :170 _rb_hits (rb only); rec_num "
                               "default 20 :84",
                    "rewards": "lib/crs_rewards.py:22 weighted_hits (title normaliser, |year diff| <= 2, catalog "
                               "membership :32); Rank-GRPO reward_func_exp_inf -> metrics_align.py "
                               "evaluate_direct_match_aligned (catalog check :60, year tolerance)",
                    "launcher": "scripts/crs_train_grpo_qwen.py:86-87 rec_num=20 hard-coded, :112 trainer built "
                                "without rec_num (default 20), :59 p0_max_tokens 320",
                    "evaluator": "scripts/crs_eval.py:142 process_rec_raw, :56-64 hits_for (title equality + year), "
                                 ":31 KS=(5,10,20)",
                    "sft": "scripts/crs_train_sft.py uses prompt/completion text only (no parsing): unchanged"},
                "options": {
                    "A_pseudo_year": "render answers as 'answer (2000)' so every parser passes unchanged; rejected: "
                                     "still needs a new reward (WordNet clusters, no catalog), and teaches the model "
                                     "a meaningless token",
                    "B_adapter_new_files (recommended, least invasive)": "no edits to existing files: (1) "
                        "lib/protoqa_rewards.py (new) with the line parser (one answer per non-blank line, leading "
                        "numbering/bullets stripped, <= 50 chars) and two reward functions (per-respondent: the "
                        "row's cluster; union: any cluster) built on n928_r9_matcher.list_hits, returning per-valid-"
                        "item rewards as the P0c re-alignment expects; (2) a RankGRPOTrainer subclass overriding the "
                        "class attribute ITEM_LINE and the static _is_item (both are class-level hooks, so the core "
                        "trainer stays untouched; P0c 'lines' mode uses only _is_item); (3) a launcher copied from "
                        "crs_train_grpo_qwen.py that passes rec_num=10 to the reward and the trainer and a lower "
                        "token cap (10 short lines are ~40-60 tokens; 320 is safe but loose); (4) an eval script "
                        "reusing crs_eval.prepare_qwen35 + vLLM generation, scoring with the fast matcher "
                        "(per-respondent coverage@1/3/5/10, per row and per question) and with the official evaluator "
                        "(Max Answers@1/3/5/10, Max Incorrect@1/3/5, WordNet) in the scratch venv",
                    "C_generalise_existing": "add a --format flag to crs_train_grpo_qwen.py / crs_eval.py; more "
                                             "invasive, touches scripts that running jobs use (project notes warn "
                                             "against editing scripts in use)"},
            },
        },
        "gate_decision": "pass with conditions",
        "conditions": [
            "register the cleaning rule C1-C5 and the splits (splits/manifest.json) before any ProtoQA run; train on "
            "train.jsonl (8,581), select on validation (400), test once on test_scraped (558) and report "
            "test_crowdsourced (52) as the task's intended format",
            "state in the paper that training targets have one string per cluster (false-negative rewards; ~45% of "
            "matches lost in the crowdsourced simulation) and report the standard metric on test_crowdsourced",
            "install nltk + WordNet / stopwords / punkt_tab data on the training nodes and use n928_r9_matcher (0 "
            "disagreements with the official score) for rewards and per-respondent metrics",
            "report LLM memorisation risk (teacher and Qwen base) with a measurement on validation",
        ],
        "main_risks": [
            "target quality: single-string clusters and incomplete boards (217 train boards < 50 respondents) make "
            "the reward and the scraped test metric undercount correct answers",
            "small effect room: ~5 clusters per question and near-constant respondent totals mean row sampling ~ "
            "request sampling, so the arms differ mainly in the reward; the union reward ignores cluster sizes, "
            "which ProtoQA's Max Answers@1/3 reward (popular answers first); the per-respondent reward is count-weighted in "
            "expectation, so it is aligned with those metrics by construction and the outcome may go against the "
            "union arm; passing the gate does not predict the result",
            "test_crowdsourced (52 questions) gives wide CIs; the official 102-question test needs the authors' "
            "evaluation (not checked)",
            "memorisation of public Family Feud boards by GPT-5.6 and Qwen3.5",
            "engineering: new reward / trainer subclass / launcher / eval script plus nltk on the nodes; ProtoQA is "
            "last-but-one in the order with an Oct 4 12:00 UTC gate and ~40 GPU-h",
        ],
        "outputs": [P + "gate_check.json", P + "r9_stats.json", P + "r9_matcher_test.json", P + "r9_hand_lists.json",
                    P + "splits/", P + "nltk_stopwords_english.txt", "remark/scripts/n928_r9_stats.py", "remark/scripts/n928_r9_matcher.py",
                    "remark/scripts/n928_r9_matcher_test.py", "remark/scripts/n928_r9_gate.py"],
        "environment": {"stats_and_gate": "env/bin/python (no nltk needed; stopword list read from the NLTK data "
                                          "file)",
                        "matcher_test": "scratch venv r9venv (system python3.11, protoqa-evaluator @860dfd2 -e, "
                                        "nltk 3.10.3, numpy 2.4.6, scipy 1.17.1) with NLTK_DATA in the scratchpad; "
                                        "not reproducible with env/bin/python (no nltk)",
                        "rebuild": "see the docstring of remark/scripts/n928_r9_matcher_test.py (venv, git clone "
                                   "+ pip -e of the evaluator, NLTK data zips fetched with curl from "
                                   "raw.githubusercontent.com/nltk/nltk_data/gh-pages because nltk.download refuses "
                                   "the proxied egress)",
                        "stopwords": "198-word NLTK English list (nltk_data gh-pages, fetched 2026-10-02), copied to "
                                     "remark/data/protoqa/nltk_stopwords_english.txt; older releases have 179 words "
                                     "and the WordNet score depends on the list: pin this file on the nodes"},
    }
    (BASE / "gate_check.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main()
