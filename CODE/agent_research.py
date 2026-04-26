from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm import LocalLLM, OllamaLLM
from service import SealedAgentMemoryService

from agents.langchain_utils import (
    extract_payload_field,
    format_bullets,
    make_embeddings,
    prompt_and_query,
    retrieve_context,
)


@dataclass
class ResearchAgentConfig:
    scope_id: str
    max_docs: int = 100
    retrieval_k: int = 5
    embedding_dim: int = 128
    store_doc_summaries: bool = True


class ResearchAssistantAgent:
    """
    LangChain-style research assistant using FAISS retrieval over
    documents stored in the secure memory backend.

    Supplies dependency information for derived summaries so SAGE can record
    authenticated provenance and invalidate both source documents and their
    derivatives during forgetting.
    """

    def __init__(
        self,
        llm: LocalLLM,
        mem: SealedAgentMemoryService,
        cfg: ResearchAgentConfig,
    ):
        self.llm = llm
        self.mem = mem
        self.cfg = cfg
        self.embeddings = make_embeddings(dim=cfg.embedding_dim)

    @staticmethod
    def _heuristic_summary(text: str, max_chars: int = 220) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return ""
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"

    def add_document(self, text: str) -> str:
        doc_id = self.mem.put(
            self.cfg.scope_id,
            {"doc": text},
            kind="research_doc",
            source_scope_ids=[self.cfg.scope_id],
        )
        if self.cfg.store_doc_summaries:
            summary = self._heuristic_summary(text)
            if summary:
                self.mem.put_derived(
                    self.cfg.scope_id,
                    {"summary": summary},
                    kind="research_summary",
                    derived_from_item_ids=[doc_id],
                    source_scope_ids=[self.cfg.scope_id],
                )
        return doc_id

    def _get_documents(self) -> list[str]:
        try:
            docs = self.mem.get_recent(
                self.cfg.scope_id,
                limit=self.cfg.max_docs,
                kinds=["research_doc"],
            )
        except KeyError:
            return []

        out: list[str] = []
        for d in docs:
            doc = extract_payload_field(d, "doc")
            if doc:
                out.append(doc)
        return out

    def _get_summaries(self) -> list[str]:
        try:
            items = self.mem.get_recent(
                self.cfg.scope_id,
                limit=self.cfg.max_docs,
                kinds=["research_summary"],
            )
        except KeyError:
            return []

        out: list[str] = []
        for rec in items:
            summary = extract_payload_field(rec, "summary")
            if summary:
                out.append(summary)
        return out

    def forget_all(self) -> dict[str, Any]:
        return self.mem.forget_scope(
            self.cfg.scope_id,
            delete_ciphertext_rows=False,
            propagate=True,
        )

    def chat(self, user_text: str) -> str:
        if user_text.lower().startswith("add:"):
            text = user_text.split(":", 1)[1].strip()
            self.add_document(text)
            return "Document stored."

        docs = self._get_documents()
        summaries = self._get_summaries()
        knowledge_texts = docs + summaries

        retrieved = retrieve_context(
            texts=knowledge_texts,
            query=user_text,
            embeddings=self.embeddings,
            k=self.cfg.retrieval_k,
        )
        context_block = format_bullets(retrieved, empty="(no relevant documents)")

        system_template = """You are a research assistant.

Answer using only the retrieved knowledge base below.
If the answer cannot be found in the retrieved documents, reply exactly:
\"I don't know.\"

Retrieved Knowledge Base:
{context}
"""
        return prompt_and_query(
            self.llm,
            system_template=system_template,
            user_text=user_text,
            context_block=context_block,
        )
