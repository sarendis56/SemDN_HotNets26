"""Generate Search-R1 agentic-search traces over wiki-18.

Replays the inference protocol of third_party/Search-R1/infer.py exactly
(prompt template incl. its typos, chat template, <think>/<search>/
<information>/<answer> tags, "Doc i(Title: ...)" passage formatting, topk=3),
batched with vLLM, and logs per-turn retrieval results to JSONL:

  {"task_id", "dataset", "qid", "question", "golden_answers",
   "turns": [{"turn", "query", "ids", "scores"}, ...],
   "answer", "em", "cover_em", "status", "n_turns"}

Run (retrieval server must be up):
  CUDA_VISIBLE_DEVICES=1 uv run python -m scdn.gen_traces --n-per-dataset 300
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from scdn import paths
from scdn.metrics import cover_em, em

PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: {question}\n"
)
SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class TaskState:
    task_id: str
    dataset: str
    qid: str
    question: str
    golden: list[str]
    prompt: str
    turns: list[dict] = field(default_factory=list)
    answer: str | None = None
    status: str = "active"  # answered | no_answer | max_turns | context_overflow


def load_questions(dataset: str, n: int, seed: int) -> list[dict]:
    d = paths.FLASHRAG_DATASETS / dataset
    files = [d / f"{split}.jsonl" for split in ("dev", "test", "train")]
    f = next((p for p in files if p.exists()), None)
    if f is None:
        raise FileNotFoundError(f"no split file under {d}")
    return _load_rows(f, n, seed)


def _load_rows(path, n: int, seed: int) -> list[dict]:
    import random

    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(
                {"qid": str(r["id"]), "question": r["question"], "golden": r["golden_answers"]}
            )
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def retrieve_batch(url: str, queries: list[str], topk: int, chunk: int = 1024) -> list[list[dict]]:
    out: list[list[dict]] = []
    for i in range(0, len(queries), chunk):
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{url}/retrieve",
                    json={"queries": queries[i : i + chunk], "topk": topk, "return_scores": True},
                    timeout=1800,
                )
                r.raise_for_status()
                out.extend(r.json()["result"])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"retrieval retry {attempt + 1}: {e}", flush=True)
                time.sleep(5)
    return out


def format_passages(items: list[dict]) -> str:
    s = ""
    for i, item in enumerate(items):
        content = item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        s += f"Doc {i + 1}(Title: {title}) {text}\n"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=paths.SEARCH_R1_7B)
    ap.add_argument(
        "--datasets", default="hotpotqa,2wikimultihopqa,musique,bamboogle"
    )
    ap.add_argument("--questions-file", default=None,
                    help="FlashRAG-format jsonl (e.g. E2-CRAG questions.jsonl); overrides --datasets")
    ap.add_argument("--questions-tag", default="crag", help="dataset label when --questions-file is set")
    ap.add_argument("--n-per-dataset", type=int, default=300)
    ap.add_argument("--retrieval-url", default="http://127.0.0.1:8000")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument(
        "--topk-log",
        type=int,
        default=50,
        help="retrieve/log this many candidates per query; only --topk are shown to the agent",
    )
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (decoding)")
    ap.add_argument(
        "--question-seed", type=int, default=0,
        help="seed for question subsampling; keep fixed across variant runs",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="trace")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # late import: slow, GPU-touching

    out_path = args.out or paths.TRACES / f"{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # sanity: retrieval server up
    requests.get(f"{args.retrieval_url}/health", timeout=10).raise_for_status()

    if args.questions_file:
        sources = [(args.questions_tag, _load_rows(args.questions_file, args.n_per_dataset, args.question_seed))]
    else:
        sources = [(ds, load_questions(ds, args.n_per_dataset, args.question_seed))
                   for ds in args.datasets.split(",")]
    states: list[TaskState] = []
    for ds, rows in sources:
        for row in rows:
            q = row["question"].strip()
            if not q.endswith("?"):
                q += "?"
            states.append(
                TaskState(
                    task_id=f"{ds}_{row['qid']}",
                    dataset=ds,
                    qid=row["qid"],
                    question=q,
                    golden=row["golden"],
                    prompt=PROMPT_TEMPLATE.format(question=q),
                )
            )
    print(f"{len(states)} tasks loaded", flush=True)

    llm = LLM(
        model=paths.model(str(args.model)),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
    )
    tok = llm.get_tokenizer()
    if tok.chat_template:
        for s in states:
            s.prompt = tok.apply_chat_template(
                [{"role": "user", "content": s.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=1024,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    token_budget = args.max_model_len - 1024 - 64
    for round_no in range(args.max_turns + 1):
        active = [s for s in states if s.status == "active"]
        if not active:
            break
        print(f"--- round {round_no}: {len(active)} active tasks", flush=True)
        outs = llm.generate([s.prompt for s in active], sampling)
        needs_search: list[tuple[TaskState, str, str]] = []
        for s, out in zip(active, outs):
            text = out.outputs[0].text
            m_ans = ANSWER_RE.search(text)
            if m_ans:
                s.answer = m_ans.group(1).strip()
                s.status = "answered"
                continue
            m_q = SEARCH_RE.findall(text)
            if m_q and round_no < args.max_turns:
                needs_search.append((s, text, m_q[-1].strip()))
            elif m_q:
                s.status = "max_turns"
            else:
                s.status = "no_answer"
        if needs_search:
            results = retrieve_batch(
                args.retrieval_url,
                [q for _, _, q in needs_search],
                max(args.topk, args.topk_log),
            )
            for (s, text, q), res in zip(needs_search, results):
                s.turns.append(
                    {
                        "turn": len(s.turns) + 1,
                        "query": q,
                        "ids": [r["idx"] for r in res],
                        "scores": [round(r["score"], 4) for r in res],
                    }
                )
                s.prompt += SEARCH_TEMPLATE.format(
                    output_text=text, search_results=format_passages(res[: args.topk])
                )
                if len(tok(s.prompt).input_ids) > token_budget:
                    s.status = "context_overflow"

    with open(out_path, "w") as f:
        for s in states:
            rec = {
                "task_id": s.task_id,
                "dataset": s.dataset,
                "qid": s.qid,
                "question": s.question,
                "golden_answers": s.golden,
                "turns": s.turns,
                "answer": s.answer,
                "em": em(s.answer, s.golden),
                "cover_em": cover_em(s.answer, s.golden),
                "status": s.status if s.status != "active" else "max_turns",
                "n_turns": len(s.turns),
            }
            f.write(json.dumps(rec) + "\n")

    # summary
    import collections

    by_ds = collections.defaultdict(list)
    for s in states:
        by_ds[s.dataset].append(s)
    summary = {}
    for ds, group in by_ds.items():
        summary[ds] = {
            "n": len(group),
            "em": sum(em(s.answer, s.golden) for s in group) / len(group),
            "cover_em": sum(cover_em(s.answer, s.golden) for s in group) / len(group),
            "answered": sum(s.status == "answered" for s in group) / len(group),
            "mean_turns": sum(len(s.turns) for s in group) / len(group),
        }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "per_dataset": summary}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"traces -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
