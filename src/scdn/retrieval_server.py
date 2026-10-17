"""Dense retrieval server over wiki-18, Search-R1-compatible plus chunk ids.

POST /retrieve  {"queries": [...], "topk": 3, "return_scores": true}
  -> {"result": [[{"document": {"id": <chunk id>, "contents": ...},
                   "score": <float>, "idx": <faiss row>}, ...], ...]}

Encoder follows Search-R1's retrieval_server.py: e5 "query: " prefix,
mean pooling, L2 normalization, inner-product search.

Run:  uv run python -m scdn.retrieval_server --device cpu --ef-search 256
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from scdn import paths
from scdn.corpus import Corpus


class ListCorpus:
    """A generic in-memory corpus (chunk id -> 'title\\ntext') loaded from a
    contents.jsonl, so the same server can front a CRAG index (E2-CRAG) instead of
    the wiki-18 Corpus."""

    def __init__(self, path: str):
        self._c = [json.loads(l)["contents"] for l in open(path)]

    def __len__(self) -> int:
        return len(self._c)

    def contents(self, i: int) -> str:
        return self._c[i]


class E5Encoder:
    def __init__(self, model_path: Path, device: str = "cpu", max_length: int = 256):
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(paths.model(model_path))
        self.model = AutoModel.from_pretrained(paths.model(model_path)).to(device).eval()

    @torch.no_grad()
    def _encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            inputs = self.tok(
                texts[i : i + batch], max_length=self.max_length, padding=True,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            hs = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            hidden = hs.masked_fill(~mask[..., None].bool(), 0.0)
            emb = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy().astype(np.float32, order="C"))
        return np.concatenate(out, axis=0)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([f"query: {q}" for q in queries])

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode([f"passage: {p}" for p in passages])


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=paths.WIKI18_HNSW64_INDEX)
    ap.add_argument("--encoder", type=Path, default=paths.E5_BASE_V2)
    ap.add_argument("--contents", default=None,
                    help="contents.jsonl for a ListCorpus (e.g. E2-CRAG); default = wiki-18 Corpus")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ef-search", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--encode-batch", type=int, default=64)
    args = ap.parse_args()

    print(f"loading corpus metadata ...", flush=True)
    corpus = ListCorpus(args.contents) if args.contents else Corpus()
    print(f"loading encoder on {args.device} ...", flush=True)
    encoder = E5Encoder(args.encoder, device=args.device)
    print(f"loading index {args.index} (this can take minutes) ...", flush=True)
    t0 = time.time()
    index = faiss.read_index(str(args.index))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = args.ef_search
        print(f"HNSW index, efSearch={args.ef_search}", flush=True)
    print(f"index loaded in {time.time() - t0:.0f}s, ntotal={index.ntotal:,}", flush=True)
    assert index.ntotal == len(corpus), (index.ntotal, len(corpus))

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "ntotal": index.ntotal}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        t = time.time()
        embs = []
        for i in range(0, len(req.queries), args.encode_batch):
            embs.append(encoder.encode_queries(req.queries[i : i + args.encode_batch]))
        emb = np.concatenate(embs, axis=0)
        scores, idxs = index.search(emb, req.topk)
        result = []
        for row_scores, row_idxs in zip(scores, idxs):
            items = []
            for s, i in zip(row_scores, row_idxs):
                if i < 0:
                    continue
                items.append(
                    {
                        "document": {"id": int(i), "contents": corpus.contents(int(i))},
                        "score": float(s),
                        "idx": int(i),
                    }
                )
            result.append(items)
        print(
            f"retrieve: {len(req.queries)} queries topk={req.topk} "
            f"in {time.time() - t:.2f}s",
            flush=True,
        )
        return {"result": result}

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
