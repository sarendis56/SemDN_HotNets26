"""Qwen3-Embedding-0.6B encoder (32k context) — last-token pooling, instruct
query format, per the model card. Used to give the page baseline its strongest
single-vector representation (native whole-document embedding) and the chunk arm
the same encoder, so the page-vs-chunk comparison is not confounded by encoder
power.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from semdn import paths

QA_TASK = "Given a question, retrieve Wikipedia passages that answer the question"


def _last_token_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    left_padded = mask[:, -1].sum() == mask.shape[0]
    if left_padded:
        return last_hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


class Qwen3Encoder:
    def __init__(self, model_path, device="cuda", max_length=8192):
        self.device = device
        self.max_length = max_length
        # Qwen3-Embedding expects left padding for correct last-token pooling.
        self.tok = AutoTokenizer.from_pretrained(paths.model(model_path), padding_side="left")
        self.model = AutoModel.from_pretrained(
            paths.model(model_path), torch_dtype=torch.float16
        ).to(device).eval()

    @torch.no_grad()
    def _encode(self, texts: list[str], batch: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        # length-sorted batching: group similar-length texts so short docs are
        # not padded up to a giant article's length (the dominant cost).
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        embs = [None] * len(texts)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            inp = self.tok(
                [texts[j] for j in idx], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            h = self.model(**inp).last_hidden_state
            emb = F.normalize(_last_token_pool(h, inp["attention_mask"]), p=2, dim=1)
            emb = emb.float().cpu().numpy()
            for k, j in enumerate(idx):
                embs[j] = emb[k]
        return np.vstack(embs)

    def encode_queries(self, queries: list[str], task: str = QA_TASK) -> np.ndarray:
        return self._encode([f"Instruct: {task}\nQuery: {q}" for q in queries])

    def encode_docs(self, docs: list[str]) -> np.ndarray:
        return self._encode(docs)
