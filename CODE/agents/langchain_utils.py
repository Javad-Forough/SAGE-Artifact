from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable, Optional

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    Document = None
    FAISS = None

    class Embeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:
            raise NotImplementedError


class TFIDFEmbeddings(Embeddings):
    """
    TF-IDF weighted bag-of-words embeddings.

    Better than HashEmbeddings for retrieval: rare, distinctive terms get
    higher weight (IDF) while common terms are down-weighted (TF normalisation).
    The vocabulary and IDF weights are built from the documents passed to
    embed_documents(), and then reused for embed_query().

    Still fully local, deterministic, and requires no external model.
    No external dependencies beyond numpy (already required by FAISS).

    Fallback: if fewer than 2 documents are available, delegates to
    HashEmbeddings so single-item corpora still work.
    """

    def __init__(self, max_features: int = 512) -> None:
        self.max_features = max_features
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._fallback = HashEmbeddings(dim=max_features)

    _STOPWORDS = frozenset(
        "a an the is are was were be been being have has had do does did "
        "will would could should may might shall can at in on of to for "
        "with by from up about into over after and or but not so if as "
        "it its this that these those he she we they i you my your our "
        "their his her where when how what who which why".split()
    )

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        # Light suffix stripping so works/work, prefers/prefer align
        stemmed = []
        for t in tokens:
            if t in cls._STOPWORDS:
                continue
            if len(t) > 4 and t.endswith("ing"):
                t = t[:-3]
            elif len(t) > 4 and t.endswith("ed"):
                t = t[:-2]
            elif len(t) > 4 and t.endswith("er"):
                t = t[:-2]
            elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
            if t and t not in cls._STOPWORDS:
                stemmed.append(t)
        return stemmed

    def _build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary and IDF weights from a corpus."""
        n = len(texts)
        df: Counter = Counter()
        tok_lists = [self._tokenize(t) for t in texts]
        for toks in tok_lists:
            df.update(set(toks))

        # IDF = log((1 + n) / (1 + df)) + 1  (smoothed, sklearn-style)
        idf: dict[str, float] = {
            term: math.log((1 + n) / (1 + count)) + 1.0
            for term, count in df.items()
        }
        # Keep top max_features by IDF (most distinctive terms)
        top = sorted(idf.items(), key=lambda x: -x[1])[:self.max_features]
        self._vocab = [t for t, _ in top]
        self._idf = dict(top)

    def _embed_one(self, text: str) -> list[float]:
        if not self._vocab:
            return self._fallback.embed_query(text)
        toks = self._tokenize(text)
        tf: Counter = Counter(toks)
        total = max(len(toks), 1)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for i, term in enumerate(self._vocab):
            if term in tf:
                tf_val = tf[term] / total
                vec[i] = tf_val * self._idf.get(term, 0.0)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) >= 2:
            self._build_vocab(texts)
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class HashEmbeddings(Embeddings):
    """
    Deterministic hash-based fallback embeddings.

    Each token is hashed with SHA-256; the hash selects a dimension and sign.
    Good for reproducible experiments without an external model, but TF-IDF
    weighting is absent so all terms are treated equally.

    Prefer TFIDFEmbeddings for retrieval quality.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(0, min(len(h), self.dim)):
                sign = 1.0 if (h[i] & 1) else -1.0
                vec[i] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def make_embeddings(dim: int = 512) -> Embeddings:
    """
    Return the best available embeddings backend.

    Priority:
      1. SentenceTransformer (semantic, highest quality, requires sentence-transformers)
      2. TFIDFEmbeddings (tf-idf, good quality, always available)
    """
    try:
        from sentence_transformers import SentenceTransformer
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception:
        pass
    return TFIDFEmbeddings(max_features=dim)


def llm_complete_chat(llm: Any, system_text: str, user_text: str) -> str:
    try:
        return llm.complete(system=system_text, user=user_text)
    except TypeError:
        prompt = f"### System\n{system_text}\n\n### User\n{user_text}\n\n### Assistant\n"
        return llm.complete(prompt)


def extract_payload_field(item: Any, field: str) -> Optional[str]:
    if isinstance(item, dict):
        payload = item.get("payload", item)
        if isinstance(payload, dict):
            val = payload.get(field)
            if isinstance(val, str):
                return val
    return None


def build_ephemeral_faiss(texts: list[str], embeddings: Embeddings) -> Optional[FAISS]:
    if not texts or Document is None or FAISS is None:
        return None
    docs = [Document(page_content=t) for t in texts]
    return FAISS.from_documents(docs, embeddings)


def _similarity_search_fallback(
    texts: list[str],
    query: str,
    embeddings: Embeddings,
    k: int,
) -> list[str]:
    if not texts:
        return []

    doc_vecs = np.asarray(embeddings.embed_documents(texts), dtype=np.float32)
    if doc_vecs.ndim != 2 or doc_vecs.shape[0] != len(texts):
        return texts[: min(k, len(texts))]

    query_vec = np.asarray(embeddings.embed_query(query), dtype=np.float32)
    if query_vec.ndim != 1 or query_vec.shape[0] != doc_vecs.shape[1]:
        return texts[: min(k, len(texts))]

    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        return texts[: min(k, len(texts))]

    denom = doc_norms * query_norm
    scores = np.divide(
        np.dot(doc_vecs, query_vec),
        denom,
        out=np.full(len(texts), -1.0, dtype=np.float32),
        where=denom > 0,
    )
    ranked = np.argsort(-scores, kind="stable")[: min(k, len(texts))]
    return [texts[int(i)] for i in ranked]


def retrieve_context(
    texts: list[str],
    query: str,
    embeddings: Embeddings,
    k: int = 4,
) -> list[str]:
    store = build_ephemeral_faiss(texts, embeddings)
    if store is not None:
        docs = store.similarity_search(query, k=min(k, len(texts)))
        return [d.page_content for d in docs]
    return _similarity_search_fallback(texts, query, embeddings, k)


def format_bullets(items: Iterable[str], empty: str = "(no memory)") -> str:
    items = [x for x in items if isinstance(x, str) and x.strip()]
    if not items:
        return empty
    return "\n".join(f"- {x}" for x in items)


def prompt_and_query(
    llm: Any,
    system_template: str,
    user_text: str,
    context_block: str,
) -> str:
    system_text = system_template.format(context=context_block)
    return llm_complete_chat(llm, system_text, user_text)
