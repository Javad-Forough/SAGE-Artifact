from __future__ import annotations

import re
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
class TeamAgentConfig:
    scope_id: str
    max_memories: int = 50
    retrieval_k: int = 4
    embedding_dim: int = 128
    auto_store_workspace_digest: bool = True


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "for", "and", "or", "with", "at", "by", "from",
    "when", "what", "who", "where", "why", "how", "do", "does",
    "did", "can", "could", "should", "would", "will", "about",
    "my", "our", "your", "their", "this", "that", "these", "those",
}


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {t for t in toks if t not in _STOPWORDS}


def _best_lexical_match(query: str, notes: list[str]) -> tuple[str | None, int]:
    q = _tokens(query)

    best_note = None
    best_score = 0

    for note in notes:
        score = len(q & _tokens(note))
        if score > best_score:
            best_score = score
            best_note = note

    return best_note, best_score


class TeamAssistantAgent:
    """
    LangChain-style team assistant using secure memory as the source
    of truth for stored notes.

    Supplies dependency information for derived workspace digests so SAGE can
    record authenticated provenance and prevent forgotten notes from
    indirectly surviving through summaries.
    """

    def __init__(
        self,
        llm: LocalLLM,
        mem: SealedAgentMemoryService,
        cfg: TeamAgentConfig,
    ):
        self.llm = llm
        self.mem = mem
        self.cfg = cfg
        self.embeddings = make_embeddings(dim=cfg.embedding_dim)

    @staticmethod
    def _digest_for_note(note: str, max_chars: int = 160) -> str:
        text = " ".join((note or "").split())
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    def remember_note(self, note: str) -> str:
        note_id = self.mem.put(
            self.cfg.scope_id,
            {"note": note},
            kind="team_note",
            source_scope_ids=[self.cfg.scope_id],
        )
        if self.cfg.auto_store_workspace_digest:
            digest = self._digest_for_note(note)
            if digest:
                self.mem.put_derived(
                    self.cfg.scope_id,
                    {"digest": digest},
                    kind="team_digest",
                    derived_from_item_ids=[note_id],
                    source_scope_ids=[self.cfg.scope_id],
                )
        return note_id

    def _get_notes(self) -> list[str]:
        try:
            items = self.mem.get_recent(
                self.cfg.scope_id,
                limit=self.cfg.max_memories,
                kinds=["team_note"],
            )
        except KeyError:
            return []

        notes: list[str] = []
        for m in items:
            note = extract_payload_field(m, "note")
            if note:
                notes.append(note)

        return notes

    def _get_digests(self) -> list[str]:
        try:
            items = self.mem.get_recent(
                self.cfg.scope_id,
                limit=self.cfg.max_memories,
                kinds=["team_digest"],
            )
        except KeyError:
            return []

        digests: list[str] = []
        for m in items:
            digest = extract_payload_field(m, "digest")
            if digest:
                digests.append(digest)
        return digests

    def forget_workspace(self) -> dict[str, Any]:
        return self.mem.forget_scope(
            self.cfg.scope_id,
            delete_ciphertext_rows=False,
            propagate=True,
        )

    def chat(self, user_text: str) -> str:
        if user_text.lower().startswith("note:"):
            note = user_text.split(":", 1)[1].strip()
            self.remember_note(note)
            return "Noted."

        notes = self._get_notes()
        digests = self._get_digests()
        if not notes and not digests:
            return "I don't know."

        best_note, score = _best_lexical_match(user_text, notes)

        if best_note is not None and score >= 2:
            return best_note

        corpus = notes + digests
        retrieved = retrieve_context(
            texts=corpus,
            query=user_text,
            embeddings=self.embeddings,
            k=self.cfg.retrieval_k,
        )

        if not retrieved:
            retrieved = corpus[: self.cfg.retrieval_k]

        context_block = format_bullets(retrieved, empty="(no relevant notes)")

        system_template = """You are a team assistant.

Use the retrieved team notes below when answering.
If the answer is not supported by the notes, reply exactly:
\"I don't know.\"

Retrieved Team Notes:
{context}
"""
        return prompt_and_query(
            self.llm,
            system_template=system_template,
            user_text=user_text,
            context_block=context_block,
        )
