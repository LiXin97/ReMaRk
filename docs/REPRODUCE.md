# ReMaRk: detailed reproduction guide

Back to the [README](../README.md). Paths below are relative to the repository root.

This repository holds the code, configs, launch scripts, a data export of IDs and labels, and aggregate result files for the paper.
The repository root is the package root: every script computes it as `Path(__file__).resolve().parents[2]` (shell
scripts: `REPO_ROOT`, default two levels above the script), so keep the layout.

```
remark/lib        reward functions, Rank-GRPO trainer port (TRL 1.14), ProtoQA helpers, generic LLM client
remark/scripts    data building, training, evaluation, baselines, analysis, table/figure generators
remark/configs    accelerate/DeepSpeed configs, table specs, judge prompts, theory-simulation config
remark/node       launch / queue-runner / validation scripts for GPU nodes
remark/infra      qenv_overrides.txt (uv pins), bind_local.c (optional loopback bind shim)
remark/outputs    aggregate result JSONs (analysis/, locks/, summary.json files); no per-instance outputs
data_release       IDs, labels and SFT targets (item ids) of Reddit-Books, Reddit-Books-Late and Reddit-Music (see its own files)
```

## 1. Overview

Task: conversational item recommendation from a Reddit request. Several users reply to the same request with different
items; the model generates a ranked list of 20 `Title (Year)` lines and is scored with Recall@20 / NDCG@20 against the items
the repliers named (Rank-GRPO protocol: exact title match, |year diff| <= 2).

Method (ReMaRk): GRPO training where the reward of a generated list at each position is a bounded sum over the repliers of
the request (a hit on any replier's item counts, union reward), advantages are scaled per sampled group, and requests are
sampled uniformly. Baseline (Rank-GRPO): single reward (one replier per training row), group or batch scaling, row sampling.
Arms used in the tables:

| arm | `--reward` (launch5.sh) | `--adv_norm` |
|---|---|---|
| Rank-GRPO (per-responder) | `single` | `group` (or `batch`) |
| ReMaRk | `union` | `group` (or `batch`) |

Datasets: Reddit-v2 (Rank-GRPO release, movies), Reddit-Books (r/suggestmeabook, items grounded to Goodreads works),
Reddit-Books-Late (frozen later test set), Reddit-Music (r/musicsuggestions, items grounded to MusicBrainz artists), and
ProtoQA (list answers). Models: Qwen3.5-{2B,4B,9B}-Base, Qwen3.5-27B, Qwen3.5-35B-A3B-Base, Llama-3.2-3B-Instruct,
Llama-3.1-8B-Instruct.

## 2. Environment

* CPU/data scripts: Python 3.11 with `datasets`, `numpy`, `pandas`, `pyyaml`, `scipy`; `matplotlib` for figures.
* Training/evaluation env (`/tmp/qenv` in the scripts): Python 3.11 uv venv with torch 2.10 (cu128), vLLM 0.17.1,
  transformers 5.3.0, TRL 1.14.0, deepspeed 0.19.7, flash-linear-attention 0.5.2, nvidia-nccl-cu12 2.27.5. Install with the
  pins in `remark/infra/qenv_overrides.txt` as uv overrides:
  ```
  transformers==5.3.0
  torch==2.10.0
  torchvision==0.25.0
  torchaudio==2.10.0
  nvidia-cutlass-dsl==4.4.2
  flashinfer-python==0.6.3
  quack-kernels==0.3.6
  nvidia-nccl-cu12==2.27.5
  ```
  (`uv pip install --override remark/infra/qenv_overrides.txt trl==1.14.0 vllm==0.17.1 deepspeed==0.19.7 accelerate flash-linear-attention==0.5.2 datasets`.)
  Qwen3.5 checkpoints load as `Qwen3_5ForConditionalGeneration`; pass the text tokenizer as `processing_class`
  (the base AutoProcessor has no chat template). SFT sets `<|im_end|>` as EOS.
* Llama runs: same qenv pipeline (`crs_train_sft.py --keep_eos` keeps the native `<|eot_id|>`; `crs_train_grpo_qwen.py` handles the Llama
  pad/EOS tokens). The Llama-3.2-3B bridge run with the released trainer (`crs_train_grpo.py`) uses the Rank-GRPO env:
  torch 2.7.1, TRL 0.21 with the Rank-GRPO patch, vLLM 0.10, transformers 4.55.4 (`INSTALL.SH` of the Rank-GRPO repo).
* ProtoQA extras: nltk 3.10.3 (+WordNet), scipy 1.17.1, joblib, defusedxml; data from `iesl/protoqa-data` (commit pinned in
  `n928_r9_gate.py`) and evaluator `iesl/protoqa-evaluator` (860dfd2). `n928_r9_node_setup.sh` stages them on a node.
* Multi-GPU launches in a container where binding the wildcard address fails (EADDRINUSE): build the shim and run with
  `gcc -shared -fPIC -o /tmp/libbindlocal.so remark/infra/bind_local.c -ldl`, then
  `LD_PRELOAD=/tmp/libbindlocal.so NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET NCCL_NET=Socket NCCL_NET_PLUGIN=none`
  (already exported inside the node scripts; harmless elsewhere).
* LLM API steps (teacher lists, mention extraction, GPT zero-shot, judges): `remark/lib/llm.py` is an OpenAI-compatible
  client (`chat`, `batch_chat` with a resumable JSONL cache). Set `LLM_API_KEY`, `LLM_ENDPOINT`, `LLM_MODEL`;
  Gemini audits use `GEMINI_API_KEY`. The API caches are not shipped.
* Node scripts are meant to be staged into `/tmp/work_local` (the runner calls `/tmp/work_local/launch5.sh` and
  `sft_launch3.sh`) with `export REPO_ROOT=<package root>`; models are expected in `/tmp/models/` or `data/models/`.

## 3. Rank-GRPO setup

Rank-GRPO is not redistributed here.
```
git clone https://github.com/yaochenzhu/Rank-GRPO remark/third_party/Rank-GRPO
git -C remark/third_party/Rank-GRPO checkout 0808d82
```
* Reddit-v2 data (SFT/GRPO/test splits) and `reddit_catalog/gt_catalog.pkl` come from that repository / its data release;
  place the data under `data/raw/reddit_v2/processed_datasets/` (as read by `crs_eval.py`).
* Our TRL 1.14 port of the trainer is `remark/lib/rank_grpo_trl.py` (`RankGRPOTrainer`); it is the only trainer used by the
  Qwen/Llama runs. The upstream `libs/` (parser, metrics, utils) are imported from `remark/third_party/Rank-GRPO/libs`.

## 4. Data

Reddit-v2 variants (reward/sampling arms), from the Reddit-v2 GRPO split:
```
python remark/scripts/crs_build_consensus.py
python remark/scripts/crs_build_consensus_variants.py     # union, consensusrw
python remark/scripts/crs_build_rw.py ; crs_build_single_thr.py ; crs_build_mean_m.py --m 2 4 ; n932_e7_build_vote.py
```
The same files are on Hugging Face as [`XINLI1997/ReMaRk-data`](https://huggingface.co/datasets/XINLI1997/ReMaRk-data) (one config per file), and the 23 released checkpoints (SFT, ReMaRk and Rank-GRPO on
Qwen3.5 2B/4B/9B, Llama-3.2-3B and Llama-3.1-8B; Reddit-v2, Reddit-Books and Reddit-Music) are in the
[ReMaRk collection](https://huggingface.co/collections/XINLI1997/remark-many-right-answers-one-training-example-6acdc1cf30ec71aff2b4bd62).

`data_release/` contains, for Reddit-Books, Reddit-Books-Late and Reddit-Music, only IDs and item labels: `*_requests.jsonl` (Reddit submission
id, split, month, number of repliers), `*_responses.jsonl` (replier ordinal within the request, item ids, comment ids), `*_sft_targets.jsonl` (the
GPT-5.6 teacher list of each SFT request as catalog item ids; LLM-generated) and `*_catalog.tsv`. Books catalogs map `item_id` to Goodreads work ids only (no titles or years; titles come from the UCSD Book
Graph, https://mengtingwan.github.io/data/goodreads.html, academic use); Music catalogs carry MusicBrainz MBID, artist
name and year (MusicBrainz core data, CC0). No post/comment text, prompts or completion text are shipped. `export_counts.json`
gives the row counts; `remark/scripts/release_export_ids.py` is the export script.

Rebuilding the text datasets (needs your own Reddit dump and the catalogs):
* Reddit text: Arctic Shift (https://github.com/ArthurHeitmann/arctic_shift) via `books_pull.py` / `music_pull.py`
  (API pulls of r/suggestmeabook and r/musicsuggestions), then the threads scripts.
* Books: `books_pull -> books_threads --months 2014-05:2017-10 -> books_extract (API) -> books_goodreads_index (UCSD Book Graph in
  data/raw/goodreads) -> books_ground -> books_sft_targets (API) -> books_build --tag full -> crs_build_variants_generic --root remark/data/crs/reddit_books`.
  Books-Late: same chain with `BOOKS_CFG=remark/configs/crs_reddit_books_late.yaml`.
* Music: `music_pull -> music_threads --months 2021-01:2025-06 -> music_mb_index, music_mb_years (MusicBrainz dump in
  data/raw/musicbrainz) -> music_extract (API) -> music_ground -> music_sft_targets (API) -> music_build --tag full -> crs_build_variants_generic --root remark/data/crs/music`.
  Filters: at least 15 SFT items, prompt <= 2,048 tokens, promotion filter; split by month (see `crs_reddit_books.yaml`).
* ProtoQA: `n928_r9_gate.py`, `n928_r9_stats.py`, `n928_r9_build.py --part all`, `n928_r9_dedup.py`, `n928_r9_teacher.py`.
* Audits: `books_audit.py`, `music_audit.py`, `n917_e9_books_recall.py`, `n927_books_hygiene.py`.

## 5. Training

SFT (Qwen3.5, 1,500 steps, lr 5e-5 cosine, effective batch 384; keep the best of steps 500/1000/1500 on validation R@20):
```
remark/node/sft_launch3.sh <gpus e.g. 0,1,2,3> <port> <dataset: reddit|reddit_books|music> <size: 2B|4B|9B>
```
(uses `accel_zero2_4gpu_ga{8,12,16}.yaml` for 2B/4B/9B through `crs_train_sft.py`).

GRPO, one run on 4 GPUs (ZeRO-2, `accel_zero2_4gpu_ga6.yaml`, 16 x 6 x 4 = 384 lists = 48 requests x 8 samples per step; lr 1e-6, KL 1e-3, 2,000 steps, seed 3407):
```
DATASET={reddit|reddit_books|music} SEED=3407 remark/node/launch5.sh <gpus> <port> <size> <reward> <adv_norm> <sft_ckpt> \
    [--p0_max_tokens 384 --exact_items --max_steps N --plateau K]
# ReMaRk:      launch5.sh 0,1,2,3 29500 2B union  group <sft_ckpt>
# Rank-GRPO:   launch5.sh 0,1,2,3 29501 2B single group <sft_ckpt>      (also: single batch, union batch)
```
Job queues: `remark/node/runner7h.sh <gpus> <port>` claims jobs from a queue file (grammar: `configs/grpo_queue_p0c.txt`),
trains, validates every 250-step checkpoint (`crs_eval.py`) and stops at a validation plateau (`--plateau K`). `bigval4.sh`
validates mirrored checkpoints; `autotest6.sh` locks the best-validation checkpoint and tests it once.

Runs added for the appendix checks use `node/launch7.sh` (launch5 plus: `--gradient_accumulation_steps N` selects
`accel_zero2_4gpu_ga<N>.yaml`, e.g. 12 = 96 requests per step; `P0TAG=""` trains without P0/P0c; `--rb_table_path` for the
request-uniform Rao-Blackwellized arm) and `node/runner7r.sh` (queue runner; run-name suffixes `_lr<x>`, `_rb`, `_dyn`, `_seq`,
`_cos`, `_req2x`, queue marker `--nop0`). Trainer flags used there (`crs_train_grpo_qwen.py`): `--adv_norm fixed --adv_scale 0.130`
(constant advantage scale), `--shaping_scale_const <c>` (constant P0 shaping scale), `--seq_level` (sequence-level GRPO, one
recall@20 reward per list), `--lr_scheduler cosine`, `--rb`, `--dyn_sample <k>`, `--init_weights <ckpt>` (resume after a
preemption: weights only, fresh optimizer, seed + k). The 27B / 35B-A3B batch-scaled arms start from a retrained SFT checkpoint
(same recipe, step 1,500). Node and training-log directory names in the scripts (`train_logs/node-<x>/...`) refer to logs
that are not shipped.

Other sizes and families (launch commands are the same trainer scripts with different parallelism; the original wrapper
scripts for the 8-GPU runs were not kept, so these are the settings recorded in the experiment log):
* Qwen3.5-27B (post-trained release, no -Base exists) and Qwen3.5-35B-A3B-Base: one 8-GPU node each.
  SFT: `accelerate launch --config_file remark/configs/accel_zero2_8gpu_ga6.yaml remark/scripts/crs_train_sft.py ... --per_device_train_batch_size 8 --gradient_accumulation_steps 6 --save_only_model` (8 x 8 x 6 = 384, 1,500 steps;
  ZeRO-3 was 2.4x slower). GRPO: `accelerate launch --config_file remark/configs/accel_zero3_8gpu_ga6.yaml remark/scripts/crs_train_grpo_qwen.py ... --per_device_train_batch_size 8 --gradient_accumulation_steps 6 --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.40`
  for 27B (0.45 for 35B-A3B); 35B-A3B uses `accel_zero3_8gpu_ga12.yaml` with `--per_device_train_batch_size 4 --gradient_accumulation_steps 12 --router_aux_loss_coef 0`
  (the MoE load-balancing term ran out of memory in GRPO; SFT keeps it). Both with `--p0 --p0c --p0_max_tokens 320`, 2,000 steps,
  the 27B/35B arms are ReMaRk (`--reward consensus` on `grpo_union/train`, `--adv_norm group`) and Rank-GRPO (`--reward single`).
* Llama-3.1-8B-Instruct: SFT with `crs_train_sft.py --keep_eos` (Rank-GRPO's recipe: lr 5e-5 cosine, 5% warm-up, batch 384, 1,500 steps, best of 500/1000/1500; step 1000 selected), then the same
  4-GPU GRPO as above (`launch5.sh`/`crs_train_grpo_qwen.py`, ZeRO-2, 2,000 steps, cap 320). Llama-3.2-3B starts from the
  SFT checkpoint released with Rank-GRPO.
* Offline baselines: `n927_e10a_multipos_sft.py` (multi-positive SFT); `n927_e10b_sample.py -> n927_e10b_pairs.py -> crs_train_dpo.py -> n927_e10b_select.py` (DPO); `n928_r6_pairs.py` (stronger DPO pairs).
* ProtoQA: `scripts/n928_r9_launch.sh sft|grpo ...` (arms single / rb / mean / union), seeds via `node/g1_jobs.sh`.

## 6. Evaluation

```
python remark/scripts/crs_eval.py --model <ckpt> --name <tag> --split {validation,test} \
    [--data remark/data/crs/<ds>/sft_dataset --catalog remark/data/crs/<ds>/catalog.pkl]
```
writes `summary.json` and `per_instance.jsonl` under `remark/outputs/crs/{val,test}/<tag>`. Paired bootstrap CIs
(2,000 resamples, seed 42; unit = request, submission-clustered also reported): `crs_compare.py`, `crs_compare_units.py`,
`crs_val_table.py --spec remark/configs/paper_test_tables.json --out remark/outputs/crs/analysis/paper_test_tables.json`.
Non-personalized / reference baselines: `crs_popularity_baseline.py` (train-split popularity, per dataset),
`crs_neighbor_prior.py` and `n917_e6_knn.py` (bge-kNN prior), `n917_e5_zeroshot_gpt.py` (API zero-shot),
`n917_zeroshot_vllm.py` (Qwen3.5-27B zero-shot). Judges: `n927_e7b_judge.py`, `n932_e2_judges.py`, `n917_e7_judge.py`
(prompts in `configs/judge*_prompt.txt`). Memorization: `n917_e8_memo_probe.py`, `n917_e8_stratify.py`, `n935_a_memo_probe.py`, `n935_a_stratify.py`.

## 7. Tables and figures

The generators read run outputs under `remark/outputs/crs/{val,test,analysis,locks}`. `remark/outputs/` ships the aggregate
JSONs (`analysis/*.json`, `locks/*.json`, `summary.json`), so tables and figures regenerate from them. Recomputing
per-row / per-request numbers needs `per_instance.jsonl` of each run, i.e. re-evaluating the checkpoints
(`crs_eval.py`), which are not shipped. Run `make_paper_stats.py` first (system python with matplotlib for figures).

| paper item | command (from the package root) | inputs |
|---|---|---|
| Tab. main | `make_tab_main_v2.py` | `paper_fig_stats.json`, `test_music_pair_*.json` |
| Tab. ablation | `make_tab_ablation_main.py` | `crs_val_table.py --spec configs/paper_test_tables.json` |
| Tab. ablation (val / test / test diffs) | `make_ablation_tables.py` | `configs/paper_val_tables{,_ug}.json`, `paper_test_tables.json` |
| Tab. main-full (Qwen3.5 / Llama / Books+Music: `tab_main_full.tex`, `tab_main_full_llama.tex`, `tab_main_full_b.tex`), books-val | `make_full_table.py` (books-val spec: `configs/paper_books_val_table.json`) | per-run `summary.json` / `per_instance.jsonl` |
| Tab. compute, Fig. GPU-hours | `n932_e3_compute.py`, then `make_tab_compute_col.py` | `n928_step_cost.py`, training logs |
| Tab. robust, top-k; Figs. main, attribution, curves, val arms | `make_paper_figures.py` (after `make_paper_stats.py`; top-k needs `n917_e6_metrics.py`) | `outputs/crs/val/*/summary.json` |
| Fig. noise (responders, dose-response) | `make_fig_noise.py` | `paper_fig_stats.json`, `crs_dose_response.py`, `crs_build_mean_m.py` runs |
| advantage-variance share | `adv_variance.py` | `crs_consensus_check2.py` |
| Tab. data / responder statistics | `crs_responder_stats.py`, `n928_r7f_cited_stats.py`, each dataset's `stats.json` | |
| batch-sigma, sampler coverage, shaping magnitude, popularity strata, multi-responder hits | `n928_r7c_batch_sigma.py`, `n928_r7b_sampler_coverage.py`, `n928_r7e_shaping_magnitude.py`, `n928_r7a_popularity_strata.py`, `n928_r7d_multi_responder_hits.py` | training logs (`outputs/crs/train_logs/run_logs/`) |
| minority answers, paired checkpoints, seeds, gradient sketch, ProtoQA steps | `n931_c3_minority.py`, `n931_c1c2.py`, `n931_g1_seeds.py`, `n931_g2_gradsketch.py` (`node/g2_run.sh`), `n931_c4_protoqa.py` | |
| extra-arm readouts, stability, lengths, Books-Late | `n932_e7_readout.py`, `n932_reviewer_checks.py`, `n934_stability.py`, `n934_lengths.py`, `n933_a_readout.py`, `n933_a_stability.py`, `n933_c_readout.py` | |
| Music name-only / year diagnostics | `n935_b_music_name.py`, `n928_music_year_diag.py` | |
| judge / memorization tables | `n927_e7b_judge.py`, `n917_e8_stratify.py`, `n935_a_stratify.py` | |
| overlap sensitivity, pooled sizes, buckets | `crs_overlap_sensitivity.py`, `crs_pooled_sizes.py`, `crs_bucket_boot.py`, `crs_breakdown.py`, `n928_buckets_group.py` | |
| GPU-hours total | `n927_gpu_hours.py` | |
| appendix checks: advantage mass, KL at matched quality, cosine schedule | `e2_adv_mass.py`, `e2_kl_matched.py`, `n935_d_cosine.py` | training logs, `n928_cache/` |
| appendix checks: reward dose under group / fixed scaling, Rao-Blackwellization, sequence-level GRPO, learning rate, no P0, 9B plateau seed 42, Table 5 / 35B-A3B and 27B batch cells | `n936_e1_dose_group.py`, `n936_e1_iso.py`, `n936_e3_rb.py`, `n936_e4_seq.py`, `n936_e5_lr.py`, `n936_e6_nop0.py`, `n936_tab5_35b.py` | per-run `summary.json`, `per_instance.jsonl` |
| twice the requests per step (req2x), gradient norm and KL by arm | `n937_req2x.py s3407 s42`, `n937_gradnorm_kl.py` | training logs, `val/*/summary.json` |
| theory simulations | `sim_consensus_theory.py [--figures-only]` (config `configs/sim_consensus_theory.yaml`) | |
| ProtoQA tables | `n928_r9_eval.py`, `n928_r9_compare.py`, `n928_r9_official_check.py` | |
| schematic figures | drawn externally, not generated | |

## 8. Hardware

Runs used 4 GPUs per 2B/4B/9B/Llama run and 8 GPUs for 27B / 35B-A3B and all SFT of those two models (B200-class, bf16).
Total GPU time reported in the paper is computed by `n927_gpu_hours.py` from the run logs.

## License

Code: Apache-2.0 (`LICENSE`). The IDs and labels in `data_release/` are released under CC BY-NC 4.0. Third-party data (Reddit text, UCSD Book Graph, MusicBrainz, ProtoQA, Rank-GRPO release) keep their own terms.

## Citation

```bibtex
@misc{li2026remark,
  title  = {Many Right Answers, One Training Example: Removing Responder Noise from GRPO for LLM Recommenders},
  author = {Li, Xin and Jiang, Hao and Guo, Jinghao and Lin, Weisi and Yuen, Chau},
  year   = {2026},
  note   = {Preprint; arXiv link coming soon}
}
```
