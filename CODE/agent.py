from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from llm import LocalLLM, OllamaLLM
from service import SealedAgentMemoryService

from agents.langchain_utils import (
    TFIDFEmbeddings,
    extract_payload_field,
    format_bullets,
    make_embeddings,
    prompt_and_query,
    retrieve_context,
)

REMEMBER_PAT = re.compile(
    r"^\s*(remember|save|store)\s*(this\s*)?:\s*(.+)\s*$",
    re.IGNORECASE,
)


@dataclass
class AgentConfig:
    scope_id: str
    max_memories: int = 32
    retrieval_k: int = 6
    enable_llm_memory_summarizer: bool = True
    max_facts_per_turn: int = 5
    embedding_dim: int = 512
    store_message_artifacts: bool = True


class SealedMemoryAgent:
    """
    LangChain-style personal agent backed by SealedAgentMemoryService.

    Uses TF-IDF embeddings (or sentence-transformers if installed) for
    retrieval, answers queries through LLM-guided use of retrieved memory,
    enables implicit fact extraction by default, and supplies dependency
    information for message artifacts so SAGE can record authenticated
    provenance and transitively invalidate downstream summaries during
    forgetting.
    """

    def __init__(self, llm: LocalLLM, mem: SealedAgentMemoryService, cfg: AgentConfig):
        self.llm = llm
        self.mem = mem
        self.cfg = cfg
        self.embeddings = make_embeddings(dim=cfg.embedding_dim)

    # ------------------------------------------------------------------
    # Memory extraction
    # ------------------------------------------------------------------

    def _extract_memory_candidates_explicit(self, user_text: str) -> list[str]:
        m = REMEMBER_PAT.match(user_text)
        if not m:
            return []
        fact = (m.group(3) or "").strip()
        return [fact] if fact else []

    def _llm_summarize_to_memory(self, user_text: str) -> list[str]:
        """
        Ask the LLM to extract durable facts from arbitrary user text.
        Returns a list of fact strings (may be empty).
        """
        prompt = (
            "### System\n"
            "Extract durable long-term user facts from the message below.\n"
            "Only extract facts that are stable preferences, identities, or "
            "explicit instructions the user wants remembered.\n"
            "Return STRICT JSON with keys:\n"
            "  \"store\": true or false\n"
            "  \"facts\": [list of short fact strings]\n"
            "No extra text outside the JSON.\n\n"
            f"### User\n{user_text}\n\n"
            "### Assistant\n"
        )
        out = self.llm.complete(prompt)
        try:
            obj = json.loads(out)
        except Exception:
            # Try to extract JSON from within the response
            m = re.search(r"\{.*\}", out, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    return []
            else:
                return []
        if not isinstance(obj, dict) or not obj.get("store"):
            return []
        facts = obj.get("facts", [])
        if not isinstance(facts, list):
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def remember_from_user_text(self, user_text: str) -> None:
        """
        Extract facts from user_text and store them with provenance edges.
        Explicit "remember: ..." takes priority; falls back to LLM extraction
        when enable_llm_memory_summarizer is True.
        """
        facts = self._extract_memory_candidates_explicit(user_text)
        if not facts and self.cfg.enable_llm_memory_summarizer:
            facts = self._llm_summarize_to_memory(user_text)

        if not facts:
            return

        source_item_ids: list[str] = []
        if self.cfg.store_message_artifacts:
            msg_id = self.mem.put(
                self.cfg.scope_id,
                {"user_text": user_text},
                kind="user_message",
                source_scope_ids=[self.cfg.scope_id],
            )
            source_item_ids.append(msg_id)

        stored_fact_ids: list[str] = []
        for fact in facts[: self.cfg.max_facts_per_turn]:
            fact_id = self.mem.put(
                self.cfg.scope_id,
                {"fact": fact},
                kind="fact",
                derived_from_item_ids=source_item_ids,
                source_scope_ids=[self.cfg.scope_id],
            )
            stored_fact_ids.append(fact_id)

        if stored_fact_ids:
            self.mem.put_derived(
                self.cfg.scope_id,
                {"facts": facts[: self.cfg.max_facts_per_turn]},
                kind="profile_summary",
                derived_from_item_ids=stored_fact_ids,
                source_scope_ids=[self.cfg.scope_id],
            )

    # ------------------------------------------------------------------
    # Forget
    # ------------------------------------------------------------------

    def forget_all(self) -> dict[str, Any]:
        return self.mem.forget_scope(
            self.cfg.scope_id,
            delete_ciphertext_rows=False,
            propagate=True,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _get_fact_strings(self) -> list[str]:
        try:
            memories = self.mem.get_recent(
                self.cfg.scope_id,
                limit=self.cfg.max_memories,
                kinds=["fact"],
            )
        except KeyError:
            return []

        facts: list[str] = []
        for m in memories:
            fact = extract_payload_field(m, "fact")
            if fact:
                facts.append(fact.strip())
        return facts

    def _get_profile_summaries(self) -> list[str]:
        try:
            memories = self.mem.get_recent(
                self.cfg.scope_id,
                limit=self.cfg.max_memories,
                kinds=["profile_summary"],
            )
        except KeyError:
            return []

        summaries: list[str] = []
        for m in memories:
            payload = m.get("payload", {})
            facts = payload.get("facts")
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, str) and fact.strip():
                        summaries.append(fact.strip())

            summary = extract_payload_field(m, "summary")
            if summary:
                summaries.append(summary.strip())
        return summaries

    # ------------------------------------------------------------------
    # Main chat loop
    # ------------------------------------------------------------------

    def chat(self, user_text: str) -> str:
        """
        Process one user turn:
          1. Attempt to extract and store new facts from user_text.
          2. Retrieve relevant stored facts via TF-IDF / semantic similarity.
          3. Ask the LLM to answer using retrieved context.

        All question types are handled by the LLM.
        If no relevant memory exists the LLM is instructed to say so.
        """
        self.remember_from_user_text(user_text)

        facts = self._get_fact_strings()
        summaries = self._get_profile_summaries()
        knowledge_texts = facts + summaries

        retrieved = retrieve_context(
            texts=knowledge_texts,
            query=user_text,
            embeddings=self.embeddings,
            k=self.cfg.retrieval_k,
        )
        context_block = format_bullets(retrieved, empty="(no relevant memory)")

        system_template = (
            "You are a helpful personal assistant with access to the user's "
            "long-term memory.\n\n"
            "Use ONLY the retrieved memory below to answer user-specific questions. "
            "If the answer is not present in the retrieved memory, reply exactly: "
            "\"I don't know.\"\n\n"
            "Retrieved Memory:\n{context}"
        )

        return prompt_and_query(
            self.llm,
            system_template=system_template,
            user_text=user_text,
            context_block=context_block,
        )
