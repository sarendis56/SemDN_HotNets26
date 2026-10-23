# Semantics Delivery Network (SemDN, HotNets'26)

Code and experiments for *Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents* (HotNets '26). The paper argues that the web should deliver task-relevant **chunks** to LLM agents instead of whole **pages** and propose several opportunities and open challenges. The manuscript is available on [arXiv](https://arxiv.org/abs/2609.22486).

The code and experiments measure three things:

- **Cost:** how much content an agent processes to get the text it actually uses.
- **Locality:** how often agents re-request the same chunks, and what a chunk cache saves.
- **Quality:** whether chunk delivery answers questions at least as well as page delivery.

## Repository layout

```
src/semdn/         Python package: every experiment stage is a module, run as `python -m semdn.<module>`
scripts/
  setup/           download_data.sh (all models and datasets), sanity checks
  lib/             shared shell helpers: GPU picker (gpus.sh), robust 70B judge (judge_robust.sh)
  e1_pagetax/      §6.1 page tax: the live-web query workload
  fetchability/    §3 O4: cloud renderer on live search results
  e2_locality/     §6.2 agentic locality: Search-R1 traces, CRAG replication, prefetch analyses
  e3_amortization/ §6.2 cross-tenant amortization model
  e4_quality/      §6.3 answer quality: wiki-18 granularity matrix, CRAG iso-budget sweep
  extended/        extended results not shown in the paper
data/              datasets, corpora, indexes, traces, and experiment outputs
models/            model weights, one folder per Hugging Face id, e.g. models/Qwen/Qwen3-32B
```

All code reaches artifacts through `src/semdn/paths.py`, relative to the repo root:
`data/...` for data and `models/<hf-id>` for weights. Both folders may be symlinks to a
large disk. If a model is missing from `models/`, the code falls back to its Hugging Face
id and downloads it into the Hugging Face cache.

## Setup

Requirements: Linux, Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and NVIDIA A100
80 GB GPUs. Most stages need one GPU; the Llama-3.1-70B judge (from CRAG experiments) needs two.

```sh
uv sync
cp .env.example .env                        # API keys; only the live-web experiments need them
bash scripts/setup/download_data.sh         # everything, about 520 GB
bash scripts/setup/download_data.sh wiki18  # or one group: models | wiki18 | crag
```

`download_data.sh` also builds what is derived from the downloads: it unpacks the wiki-18
corpus, builds its metadata (`python -m semdn.corpus build`), and joins the split index files.

Optional checks: `uv run python scripts/setup/smoke_vllm.py` (the Search-R1 checkpoint loads
and generates) and `uv run python scripts/setup/validate_recall.py` (HNSW64 search agrees with
exact search).

**Shared machines.** GPU scripts never hardcode GPU ids. They source `scripts/lib/gpus.sh` and
call `on_gpus N [MIN_FREE_MIB] cmd ...`, which runs `cmd` on the N GPUs with the most free
memory. `GPUS=0,1` restricts the candidates, `ON_GPUS_WAIT=1` waits until enough GPUs are
free, and `ON_GPUS_IDLE=1` also requires them to be idle (under 10% utilization). Scripts only
stop processes they started themselves (`kill_tagged`), never by name.

## Artifacts

Downloaded by `scripts/setup/download_data.sh`:

| Path                                                                      | What                                                 | Source                                            | Size                             | Used by                         |
| ------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------- | -------------------------------- | ------------------------------- |
| `models/intfloat/e5-base-v2`                                            | dense retriever (Search-R1's)                        | HF `intfloat/e5-base-v2`                        | 0.4 GB                           | all                             |
| `models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3`   | search agent                                         | HF, same id                                       | 15 GB                            | E2 traces                       |
| `models/Qwen/Qwen2.5-7B-Instruct`                                       | reader                                               | HF, same id                                       | 15 GB                            | wiki-18 quality                 |
| `models/Qwen/Qwen3-32B`                                                 | reader                                               | HF, same id                                       | 62 GB                            | CRAG quality, wiki-18 matrix    |
| `models/Qwen/Qwen3-Reranker-4B`                                         | LLM reranker                                         | HF, same id                                       | 8 GB                             | §6.3 (both experiments)        |
| `models/BAAI/bge-reranker-v2-m3`                                        | cross-encoder reranker                               | HF, same id                                       | 2.2 GB                           | CRAG iso-budget (`chunk_ce`)  |
| `models/meta-llama/Llama-3.1-70B-Instruct`                              | answer judge (CRAG's official)                       | HF, same id (gated)                               | 132 GB                           | CRAG quality                    |
| `models/Qwen/Qwen3-Embedding-0.6B`, `models/Qwen/Qwen3-Reranker-0.6B` | other rankers                                        | HF, same ids                                      | 1.2 GB each                      | extended results                |
| `data/corpus/wiki-18/`                                                  | wiki-18: 21M passages of about 100 words, 3.2M pages | HF dataset `PeterJinGo/wiki-18-corpus`          | 5.1 GB packed + 14.4 GB unpacked | E2, wiki-18 quality             |
| `data/corpus/wiki-18-meta/`                                             | passage offsets, sizes, page ids                     | built: `python -m semdn.corpus build`            | 0.4 GB                           | same                            |
| `data/index/wiki-18-e5-flat/e5_Flat.index`                              | exact e5 index (raw 768-d vectors)                   | HF dataset `PeterJinGo/wiki-18-e5-index`        | 64.6 GB                          | E2 traces, wiki-18 quality      |
| `data/index/wiki-18-e5-hnsw64/e5_HNSW64.index`                          | approximate index + its proximity graph              | HF dataset `PeterJinGo/wiki-18-e5-index-HNSW64` | 75.7 GB                          | E2 prefetch analyses            |
| `data/datasets/flashrag/`                                               | HotpotQA, 2Wiki, MuSiQue, Bamboogle, NQ (dev/test)   | HF dataset `RUC-NLPIR/FlashRAG_datasets`        | 0.75 GB                          | E2, wiki-18 quality, E1 queries |
| `data/datasets/crag/crag_task_1_dev_v4_release.jsonl.bz2`               | CRAG Task 1 (5 pages per question)                   | HF dataset `Rickyoung0221/crag`                 | 0.7 GB                           | E2 CRAG replication             |
| `data/datasets/crag/crag_task_3_dev_v4/`                                | CRAG Task 3 (50 pages per question)                  | HF dataset `Rickyoung0221/crag`                 | 7.2 GB packed, 48 GB unpacked    | CRAG quality                    |

The two indexes are the largest files. If disk speed matters, put `data/index` and
`data/corpus` on an SSD and symlink them; the wiki-18 quality run memory-maps the flat
index and reads passages by random access.

Built by the experiments (all under `data/`, about 3.5 GB in total besides the above):
`data/traces/` (Search-R1 traces, 23 MB), `data/e1/` (page-tax measurements and wiki-18
quality results, 14 MB), `data/e2_crag/` (shared CRAG index and traces, 0.4 GB), `data/e4/`
(CRAG chunk caches, candidate pools, reader answers, judge labels, 2.5 GB). Summary
statistics go to `outputs/`.

## Experiments in the paper

| Paper                   | Claim                                                                                                                                                                                                                        | Run                                                             | Result files                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| §3, O4                 | 1,430 live result URLs: a cloud renderer (Cloudflare Browser Rendering) gets usable text from at most 75%; raw extraction is about 2× the needed text                                                                       | [Fetchability](#fetchability-3-o4)                               | `data/e4/cf_pages_cache.jsonl`, `data/e4/cf_cleanliness.json`         |
| §6.1, Fig. 2           | bytes processed vs used per query: 7.1 KB of chunks vs 761 KB HTML (107×) vs 18.1 MB rendered (2,532×); 39% of pages need JavaScript, about 19% are bot-blocked                                                            | [Page tax](#page-tax-61-fig-2)                                   | `data/e1/measure.jsonl`                                                 |
| §6.2, Fig. 3a and text | within-task reuse: 30.5% repeated queries, 32–55% of a new query's chunks seen earlier in the task; reuse intervals vs shuffled controls                                                                                    | [Agentic locality](#agentic-locality-62-fig-3a-3b)               | `data/traces/full-t07-s{0,1,2}.jsonl`, `outputs/locality_stats.json`  |
| §6.2, Fig. 3b and text | chunk vs URL-object cache: 73.8% hits at 1.34 MB, 13–41× fewer origin bytes at equal size                                                                                                                                  | same                                                            | `outputs/cachesim.csv`                                                  |
| §6.2 text              | replication on CRAG pages: 63–79% within-task reuse, 25–70× fewer origin bytes                                                                                                                                            | [CRAG replication](#crag-replication-62-text)                    | `data/e2_crag/`, `outputs/e2_crag_{locality_stats.json,cachesim.csv}` |
| §6.2 text              | top-10 domains are 34% of results; about 6× less repeated work at 50 tenants; 392 chunks/s per A100 with e5                                                                                                                 | [Amortization](#cross-tenant-amortization-62-text)               | `outputs/e3_stats.json`                                                 |
| §6.3 ¶1               | wiki-18, five QA sets: chunks match or beat a page-first client at ten pages in judged accuracy (Llama-3.1-70B) under the same ranker (e5, or e5 then Qwen3-Reranker-4B) and two readers; the client moves about 1,200× (e5) and 3,000× (reranker) more bytes | [wiki-18 matrix](#wiki-18-granularity-matrix-63-paragraph-1)     | `data/e1/matrix_llm_*`, `outputs/e1_matrix_bytes.json`                |
| §6.3 ¶2               | HotpotQA (300): 16–29% of the global top-5 chunks sit on pages outside a whole-page ranker's top 20                                                                                                                         | [Delivery gap](#delivery-gap-63-paragraph-2)                     | `outputs/e1_recall_stats.json`                                          |
| §6.3 ¶3, Fig. 3c      | CRAG Task 3, n=600, equal token budgets 128–12,288: chunks vs pages vs snippets, e5 and LLM-reranked                                                                                                                        | [Iso-budget sweep](#crag-iso-budget-sweep-63-paragraph-3-fig-3c) | `data/e4/judge_iso_*`, `data/e4/significance_iso_*.json`              |

### Fetchability (§3, O4)

Takes the search results of the live CRAG question set (first 120 questions, top-12 URLs
each), fetches each URL through Cloudflare's Browser Rendering `/markdown` endpoint, and
measures how many pages give usable clean text. Text size is compared with a plain GET plus
trafilatura of the same URLs.

```sh
# prerequisite: live search results and plain fetches (needs SERP_API_KEY; network only)
uv run python -m semdn.e4_granularity --stage serp
uv run python -m semdn.e4_granularity --stage fetch            # plain GET + trafilatura
# cloud fetch (needs CF_ACCOUNT_ID / CF_API_TOKEN), then the report
set -a; . ./.env; set +a
uv run python scripts/fetchability/cf_fetch_pages.py --n 120 --topk 12 --max-calls 1500 --conc 8
uv run python scripts/fetchability/cf_cleanliness_report.py
```

### Page tax (§6.1, Fig. 2)

100 live queries (50 curated intents in ten categories, 50 from Natural Questions), top-5
search results each, every URL fetched three ways: full browser render, HTML only, and
cleaned text. The chunk arm keeps the five best e5 chunks across the five pages.

```sh
uv run python scripts/e1_pagetax/build_queries.py                     # -> data/e1/queries.jsonl
uv run python -m semdn.serp --queries data/e1/queries.jsonl --out data/e1/serp.jsonl   # needs SERP_API_KEY
uv run python -m semdn.e1_fetch --serp data/e1/serp.jsonl --out data/e1/measure.jsonl --k 5
```

Live-web numbers drift as pages change.

### Agentic locality (§6.2, Fig. 3a, 3b)

Runs the Search-R1 agent (Qwen2.5-7B, PPO v0.3) over about 1,000 questions from five QA
sets against wiki-18 with exact e5 retrieval, three sampled runs per question (temperature
0.7, as in Search-R1's `infer.py`), plus one greedy control run. Then synthesizes a
multi-session request stream and replays it against chunk and URL-object caches.

```sh
bash scripts/e2_locality/gen_traces.sh            # retrieval server on the flat index + 4 trace runs
uv run python -m semdn.analyze --traces data/traces/full-t07-s{0,1,2}.jsonl
```

### CRAG replication (§6.2 text)

Same agent and analysis over a shared e5 index of the pages CRAG provides for 1,500 Task 1
questions (5,612 unique pages, 107,511 chunks).

```sh
uv run python -m semdn.e2_crag_index --task 1 --n 1500     # GPU; -> data/e2_crag/
bash scripts/e2_locality/crag_replication.sh              # server + traces + locality/cache analysis
```

### Cross-tenant amortization (§6.2 text)

Measures domain concentration of the live search results (from the page-tax run), e5
embedding throughput on one GPU, and feeds both into a Zipf tenant-demand model.

```sh
uv run python scripts/e3_amortization/e3_amortization.py           # GPU; -> outputs/e3_stats.json
```

### wiki-18 granularity matrix (§6.3, paragraph 1)

Five QA sets (HotpotQA, 2Wiki, MuSiQue, NQ: 500 questions each; Bamboogle: all 125), wiki-18.
Every arm starts from e5's top 100 passages and hands the reader five passages:

| Arm            | How the five passages are chosen                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `chunk_e5`   | top 5 by e5                                                                                                                                                              |
| `chunk_llm`  | the 100 candidates reranked by Qwen3-Reranker-4B; top 5                                                                                                                  |
| `page_e5@N`  | pages ranked by their mean e5 vector; best 5 candidates on the top N pages                                                                                               |
| `page_llm@N` | top 30 pages by mean e5 vector, each whole page (up to the reranker's 32K context) reranked by Qwen3-Reranker-4B; best 5 candidates (by the reranker) on the top N pages |

The reranker and its scoring code are the same as in the CRAG sweep below
(`semdn.e4_isobudget.make_llm_scorer`). Answers are graded by Llama-3.1-70B with CRAG's metric (correct = perfect or acceptable); paired
McNemar tests, Holm-corrected over the five datasets. Token F1 is also reported (sign-flip
permutation tests).

Average judged accuracy over the five datasets:

| Reader | `chunk_e5` | `page_e5@1` | `@3` | `@5` | `@10` | `chunk_llm` | `page_llm@1` | `@3` | `@5` | `@10` |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.456 | 0.392 | 0.424 | 0.435 | 0.427 | 0.500 | 0.456 | 0.500 | 0.496 | 0.493 |
| Qwen3-32B (thinking off) | 0.497 | 0.413 | 0.468 | 0.478 | 0.486 | 0.562 | 0.497 | 0.550 | 0.554 | 0.547 |

Average F1:

| Reader                   | `chunk_e5` | `page_e5@1` | `@3` | `@5` | `@10` | `chunk_llm` | `page_llm@1` | `@3` | `@5` | `@10` |
| ------------------------ | ------------ | ------------- | ------ | ------ | ------- | ------------- | -------------- | ------ | ------ | ------- |
| Qwen2.5-7B               | 0.358        | 0.295         | 0.328  | 0.336  | 0.333   | 0.395         | 0.355          | 0.395  | 0.399  | 0.393   |
| Qwen3-32B (thinking off) | 0.322        | 0.258         | 0.299  | 0.309  | 0.304   | 0.371         | 0.341          | 0.371  | 0.364  | 0.360   |

```sh
bash scripts/e4_quality/wiki18_matrix.sh    # 5 retrieval processes in parallel, then the Qwen2.5-7B reader + tests
DATASETS="hotpotqa nq" bash scripts/e4_quality/wiki18_matrix.sh   # fewer at once if RAM is tight
bash scripts/e4_quality/wiki18_reader32b.sh # Qwen3-32B reader on the same passages; waits for idle GPUs
bash scripts/e4_quality/wiki18_judge.sh     # Llama-3.1-70B judge for both readers + McNemar tests; two 80 GB GPUs
```

Each retrieval process holds its own copy of the corpus in RAM, so five at once need more
than 250 GB. Rerunning the script skips the datasets that finished.

Bytes each client moves, on 50 sampled questions per dataset: SemDN's 5 chunks vs. the live
Wikipedia HTML (one GET, no JavaScript, decompressed) of the pages a page-first client downloads
(top N by e5, or all 30 the reranker reads). CPU and network, about an hour; page sizes are cached
in `data/e1/wiki_html_sizes.jsonl`.

| Client                       | HTML downloaded              | vs. SemDN (3.2 KB)              |
| ---------------------------- | ---------------------------- | ------------------------------- |
| e5, top 1 / 3 / 5 / 10 pages | 0.50 / 1.28 / 2.05 / 3.75 MB | 156× / 402× / 644× / 1,177× |
| reranker, reads 30 pages     | 9.52 MB                      | 2,989×                         |

```sh
uv run python -m semdn.e1_matrix_bytes --per-ds 50     # -> outputs/e1_matrix_bytes.json
```

Outputs: `data/e1/matrix_llm_<dataset>.jsonl` (chosen passages), `matrix_llm_{,qwen3_32b_}quality.jsonl`
(means), `matrix_llm_{,qwen3_32b_}preds.jsonl` (per-question answers and F1),
`matrix_llm_{,qwen3_32b_}significance.json` (F1 tests), and `judge_matrix_llm_{qwen25_7b,qwen3_32b}_{.raw.jsonl,significance.json}`
(judge labels and McNemar tests).

### Delivery gap (§6.3, paragraph 2)

300 time-stable HotpotQA questions. SemDN ranks passages globally; a page-first client ranks
pages and may only use passages from the top N pages it downloaded (N = 1 to 20). Reports the
share of SemDN's top-5 passages the client cannot reach:

| Client ranks pages by | N=1 | 3   | 5   | 10  | 20  |
| --------------------- | --- | --- | --- | --- | --- |
| mean passage vector   | 66% | 41% | 34% | 24% | 16% |
| lead passage          | 66% | 42% | 36% | 33% | 29% |

```sh
uv run python -m semdn.e1_recall --n 300                   # CPU; -> outputs/e1_recall_stats.json
```

### CRAG iso-budget sweep (§6.3, paragraph 3, Fig. 3c)

600 CRAG Task 3 questions (about 46 pages each). All arms read the same cleaned text
(trafilatura, then a coherence chunker), cut to exactly B tokens, B = 128 ... 12,288; only the
delivery unit and the ranker differ. Reader Qwen3-32B, judge Llama-3.1-70B (CRAG's metric).

| Arm                                      | Fills the budget with                                                             |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| `chunk` / `chunk_ce` / `chunk_llm` | chunks ranked by e5 / bge-reranker-v2-m3 / Qwen3-Reranker-4B                      |
| `page_rel` / `page_llm`              | whole pages ranked by their best e5 chunk / by Qwen3-Reranker-4B reading the page |
| `page_seq`                             | whole pages in search-engine order                                                |
| `page_oracle`                          | pages containing the gold answer first (diagnostic)                               |
| `snippet_full`                         | all of CRAG's search snippets                                                     |

```sh
# 1. extract + chunk the 600 questions' pages once (CPU, a few hours)
uv run python -m semdn.e4_crag_t3 --stage cache --corpus t3b --chunker coherent --n 600
# 2. budgets 128-6144: chunk, chunk_ce, page_rel, page_seq, snippet_full (build, read, judge)
bash scripts/e4_quality/iso_n600_run.sh
# 3. budgets 8192 and 12288 for those arms, and the LLM-reranker arms at all budgets.
#    Builds and readers first; the judge needs two free 80 GB GPUs:
bash scripts/e4_quality/iso_xl_run.sh
SKIP_BUILD=1 JUDGE=1 bash scripts/e4_quality/iso_xl_run.sh
# 4. paired tests (McNemar, Holm over budgets)
uv run python -m semdn.e4_iso_stats --tag t3b_,t3b_xl_
```

## Extended results (not shown in the paper)

Scripts are in `scripts/extended/` unless noted. Accuracy,
truthfulness (+1 right, 0 "I don't know", −1 wrong) and hallucination rate are from the
Llama-3.1-70B judge.

**Granularity on CRAG Task 3** (600 questions, up to 50 pages each, Qwen3-32B reader;
`semdn.e4_crag_t3`, `semdn.e4_baselines`; `e4_n600.sh`, `e4_n600_qwen3rr.sh`, `e4_t3b_hybrid.sh`,
`e4_baselines_run.sh`, `e4_baselines_recover.sh`, `e4_nochunk_extra.sh`, `e4_summary_v2.sh`,
`e4_t3_run.sh`, `e4_t3_readjudge.sh`; `e4_t3_fetch.sh` downloads Task 3). Each arm at its best page depth:

| Reader sees                  | Best depth | Accuracy | Truthfulness | Hallucination |
| ---------------------------- | ---------- | -------- | ------------ | ------------- |
| chunks, e5                   | 5 pages    | 0.500    | +0.113       | 0.282         |
| chunks, bge-reranker-v2-m3   | 5          | 0.522    | +0.156       | 0.268         |
| chunks, Qwen3-Reranker       | 3          | 0.507    | +0.152       | 0.257         |
| chunks + search snippets, e5 | 5          | 0.537    | +0.167       | 0.258         |
| search snippets              | 10         | 0.530    | +0.122       | 0.303         |
| whole pages                  | 3          | 0.532    | +0.155       | 0.277         |
| LLM summary of each page     | 10         | 0.502    | +0.190       | 0.215         |
| no retrieval                 | —         | 0.363    | −0.018      | 0.305         |

Retrieval beats no retrieval (+0.13 accuracy, significant). Without a fixed budget, the
delivery forms land within 0.50–0.54; truthfulness peaks near five pages, then hallucination rises.

**Page depth and pool noise on the live web** (147 CRAG questions over live search results,
Qwen2.5-7B reader; `semdn.e4_granularity`, `e4_live_n300.sh`, `e4_fetch_resilient.sh`,
`e4_fetch_render.sh`, `e4_gpu_stages.sh`). Reading only the top page is worst (accuracy 0.38); a moderate pool of 3
to 10 pages is best (0.46–0.50). With fixed 256-token chunks, the full pool of about 22 pages
does worse than 10 pages because similar-looking but wrong text raises hallucination.
Structure-aware chunking without a title prefix removes that penalty (`e4_struct_pipeline.sh`,
`e4_structnt_ablation.sh`, `e4_structnt_extra.sh`); adding the page title to every chunk hurts.

**Snippets vs chunks on the live web** (287 questions; `semdn.e4_live_baselines`,
`e4_live_n300_baselines.sh`, `e4_live_hybrid.sh`, `e4_coherent_live.sh`). At the same number of
blocks, five search snippets tie chunks and the chunk + snippet mix, using about a third of the
tokens.

**Cloud fetch through the full pipeline** (`scripts/fetchability/cf_eval_run.sh`,
`semdn.e4_cf_eval`, `semdn.e4_cf_stats`, `e4_cf_compare.sh`, `e4_cf_aug_fix.sh`,
`scripts/fetchability/cf_recover.py`). Compared with a
plain HTTP GET plus trafilatura, Cloudflare's renderer doubles the usable pages
per question (4.5 to 9.0, top 12) and recovers 70% of blocked pages. Answer quality stays tied:
chunks still match snippets at the same context size.

**Controlled page vs chunk retrieval on CRAG Task 1** (300 questions, 5 provided pages each;
`semdn.e4_crag`, `e4_gen_norag_crag.sh`, `e4_unified_batch.sh`). Accuracy: chunk retrieval 0.470,
top 3 pages 0.447, top page 0.403 (Qwen2.5-7B). Chunks beat the top page significantly
(+0.067); chunks vs three pages is within noise.

**Rankers and chunkers** (`e4_qwen3.sh`, `e4_ce_pipeline.sh`, `e4_t3_encoders.sh`,
`e4_coherent_t3.sh`, `e4_coherent_encoders.sh`, `e4_coherent_rr_bm25.sh`, `e4_q_t3.sh`):

| Variant (vs e5 on the same candidates)                                           | Result                                              |
| -------------------------------------------------------------------------------- | --------------------------------------------------- |
| Qwen3-Embedding-0.6B                                                             | better at one page, same as e5 by five pages        |
| bge-reranker-v2-m3                                                               | more accurate, but more hallucination in deep pools |
| BM25                                                                             | ties e5                                             |
| LLM scores each chunk's usefulness (`semdn.e4_utility_rank`, `e4_utility.sh`) | same accuracy, +0.067 hallucination (Qwen3-32B)     |
| coherent chunker                                                                 | better on CRAG Task 3 pages, not on live-web pages  |

**Where answers sit** (CRAG Task 3, 300 questions; `semdn.e4_answer_locality`,
`semdn.e4_locality_strong`). 60% of questions have an answer anywhere in their 50 pages. The
answer is in the top page for 30% and the top five pages for 45%. Among answerable questions, e5
places an answer chunk among the three shown for 54% (BM25 49%).
`scripts/e4_quality/answer_survival.py` checks where gold answers are lost between raw HTML,
extracted text and chunks.

**Judge and reader** (`e4_metric.sh`, `e4_judge_llama70b.sh`, `e4_judge_bf16tp2.sh`,
`e4_judge_fp8.sh`, `e4_judge_ce.sh`, `e4_judge_now.sh`, `e4_reader_upgrade.sh`). A 7B judge is too
strict; CRAG's Llama-3.1-70B judge raises accuracy to about 3× exact match, with the same arm
ranking. FP8 quantization of the 70B judge garbles its output on A100s; bf16 on two GPUs works.
Qwen3-32B beats Qwen2.5-7B on truthfulness by +0.157 (significant).

**Iso-budget pilot** (300 questions; `iso_eval_run.sh`, `iso_snippet_full.sh`,
`iso_n600_resume.sh`): the first round of the paper's sweep.

**wiki-18 page vs chunk with a stronger page encoder** (`semdn.e1_quality`). Ranking whole pages
with Qwen3-Embedding-0.6B (32K context) instead of mean e5 vectors shrinks the delivery gap at
ten pages from 26% to 20%. Chunks still beat pages on all five datasets under both encoders.
`semdn.e1_recall_answer` adds answer F1 for the
delivery gap.

**Cost at equal quality** (`scripts/e3_amortization/cost_to_equal_quality.py`). With the same
chunks reaching the reader, SemDN moves 107× fewer bytes per query than an HTML client. At the measured cache hit rates, origin fetches and embedding work drop 3.8× (1.34 MB cache)
to 8.0× (13.4 MB).

**Prefetching and the index** (`scripts/e2_locality/`: `graph_prefetch.py`, `ann_prefetch.py`,
`investigate_deeprank.py`, `why_trajectory.py`, `compare_index_traces.py`, `dump_tables.py`). Share of a task's next new chunks that a prefetcher already holds:

| Prefetch                                 | 4 / 8 / 16 per chunk (or ranks 5 / 10 / 20) |
| ---------------------------------------- | ------------------------------------------- |
| HNSW graph neighbors of served chunks    | 1.0% / 2.3% / 5.0%                          |
| exact nearest neighbors of served chunks | 15.1% / 20.6% / 26.6%                       |
| deeper results of the agent's own query  | 21.5% / 28.6% / 35.9%                       |

The agent's next query usually narrows the current one (89% of hits come one turn later), so
its deeper results preview the next answer. HNSW and exact search agree on 88% of top-3 results,
but cache hit ratios stay within one point; exact search gains 2–4 exact-match points.

**Live semantic cache prototype** (`semdn.live_semdn_server`). A build-as-you-go cache over the
live web (search, fetch, chunk, embed, incremental index) behind Search-R1's `/retrieve` API.
Over shallow keyword search results, similarity hits are mostly hollow: cover-EM 0.03 when
served from cache vs 0.20 with a fresh fetch (threshold 0.80).

`scripts/extended/gpu_queue.sh` runs several of these in sequence.

## Contact and Citation

For any question, please contact the first author at peichunhua@link.cuhk.edu.cn or peichunhua04@gmail.com.

If you find our paper inspiring and helpful to your research, please consider citing:
```latex
@inproceedings{hua2026semantics,
  title={Semantics Delivery Network: Rethinking Web Retrieval Infrastructure for LLM Agents},
  author={Peichun Hua and Yunming Xiao},
  booktitle={The 25th ACM Workshop on Hot Topics in Networks (HotNets '26)},
  year={2026},
  publisher={Association for Computing Machinery},
  address={New York, NY, USA},
  url={https://doi.org/10.1145/3847290.3859940},
  doi={10.1145/3847290.3859940},
  numpages={13},
  keywords={Content Delivery Networks, Retrieval-Augmented Generation, Agentic Search, Semantic Caching, Vector Search},
  location={Salt Lake City, UT, USA},
  series={HotNets '26}
}
```
