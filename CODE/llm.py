# llm.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Ollama (GPU-accelerated, free, local)
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    model: str = "llama3.2"          # any model you have pulled in Ollama
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 512
    # Extra Ollama options forwarded verbatim (e.g. {"num_gpu": 99})
    options: dict = field(default_factory=dict)


class OllamaLLM:
    """
    LLM backed by a local Ollama server (https://ollama.com).

    Ollama handles GPU offloading automatically — no extra config needed.
    Install Ollama, pull a model, then point this class at it:

        ollama pull llama3.2          # or mistral, phi3, gemma2, qwen2.5, etc.
        ollama pull llama3.2:latest

    Requires: pip install ollama
    Or falls back to the raw HTTP API via requests if ollama SDK not installed.

    Same complete() interface as LocalLLM so all agents work unchanged.
    """

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg
        self._client = self._make_client()

    def _make_client(self):
        try:
            import ollama
            return ollama.Client(host=self.cfg.host)
        except ImportError:
            return None   # will use requests fallback

    def complete(
        self,
        prompt: str = "",
        system: Optional[str] = None,
        user: Optional[str] = None,
    ) -> str:
        """
        Same interface as LocalLLM.complete():
          - If system + user are given: uses chat API (preferred for instruction models)
          - Otherwise: uses generate API with raw prompt
        """
        options = {"temperature": self.cfg.temperature, **self.cfg.options}

        if system is not None and user is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            return self._chat(messages, options)

        # raw prompt → generate
        return self._generate(prompt, options)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        parts = [str(exc)]
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                parts.append(response.text)
            except Exception:
                pass
        return "\n".join(p for p in parts if p)

    def _should_retry_on_cpu(self, exc: Exception, options: dict) -> bool:
        if "num_gpu" in options:
            return False
        text = self._error_text(exc).lower()
        return (
            "cudamalloc failed" in text
            or "out of memory" in text
            or "failed to allocate cuda" in text
            or "cuda" in text and "failed" in text
        )

    def _chat(self, messages: list, options: dict) -> str:
        if self.cfg.max_tokens:
            options = {**options, "num_predict": self.cfg.max_tokens}

        try:
            return self._chat_once(messages, options)
        except Exception as exc:
            if not self._should_retry_on_cpu(exc, options):
                raise
            retry_options = {**options, "num_gpu": 0}
            return self._chat_once(messages, retry_options)

    def _chat_once(self, messages: list, options: dict) -> str:
        if self._client is not None:
            resp = self._client.chat(
                model=self.cfg.model,
                messages=messages,
                options=options,
            )
            return resp["message"]["content"].strip()

        # requests fallback
        import requests
        r = requests.post(
            f"{self.cfg.host}/api/chat",
            json={"model": self.cfg.model, "messages": messages,
                  "stream": False, "options": options},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _generate(self, prompt: str, options: dict) -> str:
        if self.cfg.max_tokens:
            options = {**options, "num_predict": self.cfg.max_tokens}

        try:
            return self._generate_once(prompt, options)
        except Exception as exc:
            if not self._should_retry_on_cpu(exc, options):
                raise
            retry_options = {**options, "num_gpu": 0}
            return self._generate_once(prompt, retry_options)

    def _generate_once(self, prompt: str, options: dict) -> str:
        if self._client is not None:
            resp = self._client.generate(
                model=self.cfg.model,
                prompt=prompt,
                options=options,
            )
            return resp["response"].strip()

        import requests
        r = requests.post(
            f"{self.cfg.host}/api/generate",
            json={"model": self.cfg.model, "prompt": prompt,
                  "stream": False, "options": options},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()


# ---------------------------------------------------------------------------
# llama-cpp (CPU fallback — keeps backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 4
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 256


class LocalLLM:
    """
    CPU-only LLM backed by llama-cpp-python (GGUF models).
    Kept for backward compatibility and environments without Ollama.
    For GPU acceleration use OllamaLLM instead.
    """

    def __init__(self, cfg: LlamaCppConfig) -> None:
        self.cfg = cfg
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            verbose=False,
        )

    def complete(
        self,
        prompt: str = "",
        system: Optional[str] = None,
        user: Optional[str] = None,
    ) -> str:
        if system is not None and user is not None:
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return resp["choices"][0]["message"]["content"].strip()

        resp = self.llm(
            prompt,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_llm(backend: str = "ollama", **kwargs):
    """
    Factory for selecting an LLM backend at runtime.

    backend="ollama"  -> OllamaLLM  (GPU, free, recommended)
    backend="llama"   -> LocalLLM   (CPU, GGUF file required)

    Examples:
        make_llm("ollama", model="llama3.2")
        make_llm("ollama", model="mistral", temperature=0.1)
        make_llm("llama",  model_path="/models/phi-3.gguf")
    """
    if backend == "ollama":
        cfg = OllamaConfig(
            model=kwargs.get("model", "llama3.2"),
            host=kwargs.get("host", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 512),
            options=kwargs.get("options", {}),
        )
        return OllamaLLM(cfg)

    if backend == "llama":
        cfg = LlamaCppConfig(
            model_path=kwargs["model_path"],
            n_ctx=kwargs.get("n_ctx", 2048),
            n_threads=kwargs.get("n_threads", 4),
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return LocalLLM(cfg)

    raise ValueError(f"Unknown LLM backend: {backend!r}. Choose: ollama, llama")
