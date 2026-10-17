"""wiki-18 corpus: one-pass metadata build + random access by chunk id.

Chunk id == faiss row == line number in wiki-18.jsonl (verified during build).
Metadata arrays power the cache simulator (sizes, chunk->page mapping) and
random access (offsets) without holding 15 GB of text in memory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import orjson

from semdn import paths


def _loads(line: bytes) -> dict:
    try:
        return orjson.loads(line)
    except orjson.JSONDecodeError:  # lone surrogates etc.: fall back to stdlib
        return json.loads(line.decode("utf-8", errors="replace"))

META_DIR = paths.DATA_ROOT / "corpus" / "wiki-18-meta"
CORPUS_JSONL = paths.DATA_ROOT / "corpus" / "wiki-18" / "wiki-18.jsonl"


def build_metadata(corpus_jsonl: Path = CORPUS_JSONL, out_dir: Path = META_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    sizes: list[int] = []
    page_ids: list[int] = []
    page_of_title: dict[str, int] = {}
    pos = 0
    with open(corpus_jsonl, "rb") as f:
        for i, line in enumerate(f):
            row = _loads(line)
            if int(row["id"]) != i:
                raise ValueError(f"id mismatch at line {i}: {row['id']}")
            contents = row["contents"]
            title = contents.split("\n", 1)[0].strip().strip('"')
            offsets.append(pos)
            sizes.append(len(contents.encode("utf-8", errors="replace")))
            page_ids.append(page_of_title.setdefault(title, len(page_of_title)))
            pos += len(line)
            if i % 2_000_000 == 0:
                print(f"  scanned {i:,} chunks", flush=True)

    np.save(out_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out_dir / "sizes.npy", np.asarray(sizes, dtype=np.uint32))
    np.save(out_dir / "page_ids.npy", np.asarray(page_ids, dtype=np.int32))
    titles = sorted(page_of_title, key=lambda t: page_of_title[t])
    (out_dir / "page_titles.txt").write_text("\n".join(titles), encoding="utf-8")
    summary = {
        "n_chunks": len(offsets),
        "n_pages": len(page_of_title),
        "total_bytes": int(np.sum(sizes, dtype=np.int64)),
        "mean_chunk_bytes": float(np.mean(sizes)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


class Corpus:
    """Random access to chunk contents and metadata."""

    def __init__(self, corpus_jsonl: Path = CORPUS_JSONL, meta_dir: Path = META_DIR):
        self.f = open(corpus_jsonl, "rb")
        self.offsets = np.load(meta_dir / "offsets.npy")
        self.sizes = np.load(meta_dir / "sizes.npy")
        self.page_ids = np.load(meta_dir / "page_ids.npy")

    def __len__(self) -> int:
        return len(self.offsets)

    def contents(self, idx: int) -> str:
        self.f.seek(self.offsets[idx])
        return _loads(self.f.readline())["contents"]

    def batch_contents(self, idxs) -> list[str]:
        return [self.contents(int(i)) for i in idxs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--corpus", type=Path, default=CORPUS_JSONL)
    ap.add_argument("--out", type=Path, default=META_DIR)
    args = ap.parse_args()
    build_metadata(args.corpus, args.out)
