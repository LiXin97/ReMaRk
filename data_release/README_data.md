# Data release: request / responder / item ids for Reddit-Books, Reddit-Books-Late and Reddit-Music

This directory holds identifiers and labels only. It contains no Reddit text (no posts, comments, prompts or model outputs),
no user names and no user hashes. Prompts are rebuilt from Reddit by the scripts in `remark/scripts/` (see "Rebuilding the prompts").
Reddit-v2 (the movie dataset) is not re-hosted; it comes from the Rank-GRPO release (see the top-level README).

Files were written by `remark/scripts/release_export_ids.py`, which also asserts the alignment described below
(`export_counts.json` holds the counts it verified).

| dataset | requests file | responses file | SFT targets file | catalog file |
|---|---|---|---|---|
| Reddit-Books | `books_requests.jsonl` | `books_responses.jsonl` | `books_sft_targets.jsonl` | `books_catalog.tsv` |
| Reddit-Books-Late | `books_late_requests.jsonl` | `books_late_responses.jsonl` | `books_late_sft_targets.jsonl` | `books_late_catalog.tsv` |
| Reddit-Music | `music_requests.jsonl` | `music_responses.jsonl` | `music_sft_targets.jsonl` | `music_catalog.tsv` |

The `*_sft_targets.jsonl` files are written by `remark/scripts/release_export_sft_targets.py`.

## Fields

`*_requests.jsonl` (one line per request)
- `request_id`: Reddit submission id.
- `split`: `train`, `validation` or `test` (by submission month, see below).
- `month`: `YYYY-MM` of the submission.
- `n_responders`: number of responders of this request that have at least one usable item, i.e. the number of lines of the responses file for this request.
- `in_sft_train` (train requests only): true if the request has an SFT target (see "Filters"); the SFT training rows are exactly the responders of these requests.
- `n_responders_all` (Music only): number of responders considered before the item filters (the first 20 top-level commenters, with or without a usable item).

`*_responses.jsonl` (one line per responder with at least one usable item; sorted by request, then responder)
- `request_id`
- `responder`: ordinal 0, 1, 2, ... of the responder within its request (the order of the rows in the dataset; for Books the order of the first comment). It is not an identifier across requests; the same person has unrelated ordinals in different requests.
- `item_ids`: indices into the catalog file (`item_id` column), in the order the responder named the items; de-duplicated, with the items the requester already named removed. This is exactly the ground-truth list of the corresponding row of the training/evaluation datasets.
- `comment_ids`: Reddit ids of the responder's top-level comments in this request.
- Consensus weights are not stored: the weight of item `i` in a request is the fraction of its responders whose `item_ids` contain `i` (this is how the `grpo_consensus` / `grpo_union` training sets are built; union = the set of items).

`*_sft_targets.jsonl` (one line per train request with `in_sft_train` = true, in the order of the requests file)
- `request_id`
- `item_ids`: the SFT target of the request as catalog indices, in list order (15 to 20 items). **These lists are LLM-generated**: the
  teacher list is a GPT-5.6 recommendation list for the request, grounded to the catalog, with seen items and duplicates removed.
  Every SFT training row of the request (one per responder) has this list as its completion, rendered one `title (year)` /
  `name (year)` per line. The export checks that every line maps to a catalog item and that the rows align with `*_responses.jsonl`.

`books_catalog.tsv`: `item_id`, `goodreads_work_ids`. `item_id` is the 0-based line index of the catalog (also the position in the sorted `(title, first-publication year)` list used by the code). `goodreads_work_ids` lists the Goodreads work id(s) of the item, comma separated (several works can share one `(title, year)` item; most-rated first). Titles and years are not included: they come from the UCSD Book Graph (Wan and McAuley, RecSys 2018; academic use) through the work id, i.e. look `goodreads_work_ids` up in the Book Graph works file. The item title in the datasets is the work's most frequent surface form in the Reddit mentions (made a fixed point of the Rank-GRPO title parser), so it can differ slightly from the Book Graph display title; the year is the Book Graph first-publication year.

`music_catalog.tsv`: `item_id`, `mbid`, `name`, `year`. `item_id` is the line index of the sorted `(name, year)` catalog. `mbid`: MusicBrainz artist id(s) (comma separated; the id used most often in the data first) whose canonical name and begin year give this item (birth year of a solo artist, formation year of a group). `name` and `year` are MusicBrainz core data (CC0).

## Splits, filters and counts

Request split by submission month; rows of the datasets are (request, responder) pairs.

| dataset | split | months | requests | responses (rows) | notes |
|---|---|---|---|---|---|
| Reddit-Books | train | 2014-05 .. 2017-07 | 32,539 | 149,679 | 31,172 requests / 146,633 rows have an SFT target |
| Reddit-Books | validation | 2017-08 | 1,255 | 6,259 | |
| Reddit-Books | test | 2017-09 | 1,150 | 6,333 | |
| Reddit-Books | catalog | | 74,474 items | | |
| Reddit-Books-Late | train / validation | as Books | 32,539 / 1,255 | 149,679 / 6,259 | same requests and responders as Reddit-Books; item ids refer to the Late catalog, so they differ from the Books ids |
| Reddit-Books-Late | test | 2017-10 .. 2017-12 | 3,795 | 21,369 | frozen later test set |
| Reddit-Books-Late | catalog | | 75,537 items | | |
| Reddit-Music | train | 2023-07 .. 2024-06 | 19,159 | 142,396 | 18,645 requests / 140,231 rows have an SFT target |
| Reddit-Music | validation | 2024-07 | 1,966 | 15,060 | |
| Reddit-Music | test | 2024-08 | 1,951 | 15,893 | |
| Reddit-Music | catalog | | 38,881 items | | |

The counts equal the `stats.json` of the datasets (train rows of the per-responder RL set, validation/test SFT-format rows).

Filters (identical for the three datasets unless noted):
- Request = one r/suggestmeabook (Books) or r/musicsuggestions (Music) submission; responders = distinct top-level commenters (Music: the first 20).
- Mentions of items were extracted from the comments with an LLM and grounded to Goodreads works (Books, UCSD Book Graph) or MusicBrainz artists (Music); items must have a usable year (Books: first-publication year; Music: begin year) and must be read back exactly by the Rank-GRPO title parser.
- Ground truth of a responder = its grounded items in order, de-duplicated, minus the items the requester named ("seen titles"). A responder without such an item gives no row; a request needs at least one row.
- Prompt length at most 2,048 tokens (Qwen3.5 chat template). Music: 1 request per split was dropped by this filter.
- SFT target (train only, `in_sft_train`): the request's grounded teacher list (GPT-5.6 recommendation list grounded to the catalog, seen items and duplicates removed) has at least 15 items.
- Music only: an (author, artist) pair named in 20 or more distinct requests is treated as request-independent promotion and all of its mentions are dropped.
- Catalog = all ground-truth items of all splits + the teacher items used in SFT completions + the most-rated 20,000 Goodreads works (Books) or the 20,000 most active MusicBrainz artists with a year (Music).

## Alignment (how a dataset row maps to a request and responder)
- Books / Books-Late: each dataset (`sft_dataset/{train,validation,test}`, `grpo/train`) has a per-row id table in the same row order. A row belongs to the request of its submission id; its responder ordinal is its rank among the rows of that submission. Train responses come from the per-responder RL set (all train requests), validation and test responses from the SFT-format validation/test sets. For train requests with an SFT target, the SFT rows are checked to be the same responders, in the same order, with the same ground truth.
- `item_ids` are obtained by mapping each ground-truth `(title, year)` of a row to its position in the catalog (all matches exist; the match key is the exact `(title, int(year))` pair).
- Music has no id table; the build is replayed on the grounded mentions (promotion filter, year rendering, seen-name filter, per-responder de-duplication) and the dataset rows are matched to the replayed requests block by block, requiring exactly the same ground-truth lists. All requests of every split matched; the only unmatched candidates are the requests removed by the prompt-length filter (1 per split), and the matched request / row counts equal the build statistics.

## Rebuilding the prompts
Prompts are the request text of the Reddit submission (title and body, and the thread context where used) behind a fixed instruction header
(see `remark/scripts/books_common.py` and `music_common.py`). Users with Reddit access (for example the Arctic Shift dumps) can run
`books_pull.py -> books_threads.py -> ...` and `books_build.py` / `music_build.py` as listed in the top-level README, or join this release by `request_id`
to rebuild each prompt and then attach `item_ids` as the label lists. Deleted or removed posts cannot be recovered.

## Licences and sources
The IDs and labels in this directory are released under CC BY-NC 4.0. Labels were produced with LLMs: item mentions were extracted from human-written comments by GPT-5.6 and grounded by deterministic rules; SFT targets are GPT-5.6 recommendation lists grounded to the catalogs.
Reddit ids are public identifiers; no Reddit content is redistributed. Goodreads data (UCSD Book Graph) is academic-use only and is not redistributed here: only work ids are listed. MusicBrainz core data is CC0.
