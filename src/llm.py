"""
LLM Client Interface and Implementations.

Supports:
1. Google Gemini (via official google-genai SDK).
2. MockLLMClient for deterministic local testing without external API credentials.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Generator, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


class BaseLLMClient(ABC):
    """Abstract interface for LLM completion generation."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates a text completion given a prompt and optional system instructions."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        """Streams text completion tokens as they are produced."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Official Google GenAI SDK integration."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-lite-latest"):
        from google import genai
        from google.genai import types

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please export GEMINI_API_KEY='...' or add it to .env file."
            )

        model_env = os.environ.get("GEMINI_MODEL")
        if not model_env or model_env in ("gemini-2.0-flash", "gemini-2.5-flash"):
            self.model = model
        else:
            self.model = model_env

        self.client = genai.Client(api_key=self.api_key)
        self.types = types

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,  # Low temperature to minimize hallucinations in RAG
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def generate_stream(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
        )
        response_stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text


class MockLLMClient(BaseLLMClient):
    """
    Simulated LLM for testing RAG pipeline without API keys.
    Demonstrates grounded answer generation and hallucination boundary detection.
    """

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        # Extract user question
        if "USER QUESTION:" in prompt:
            question_part = prompt.split("USER QUESTION:")[1].split("\n")[0].strip()
        else:
            question_part = prompt.strip()

        question_lower = question_part.lower()

        # Extract context block
        context_body = ""
        if "CONTEXT SNIPPETS:" in prompt and "USER QUESTION:" in prompt:
            context_body = prompt.split("CONTEXT SNIPPETS:")[1].split("USER QUESTION:")[0].strip()

        # If context is empty or explicitly says no relevant context
        if not context_body or "No relevant context found" in context_body:
            return "I cannot answer this question based on the provided documents."

        # Common stop-words for matching
        stop_words = {"what", "is", "the", "are", "do", "does", "how", "for", "and", "or", "to", "in", "of", "a", "an", "at", "who", "which"}
        q_words = [w.strip("?,.:;\"'") for w in question_lower.split() if w not in stop_words and len(w) > 2]

        # Extract sentences from context
        import re
        sentences = re.split(r'(?<=[.!?\n])\s+', context_body)

        matched_sentences = []
        for s in sentences:
            s_clean = s.strip()
            # Ignore headers and snippet markers
            if not s_clean or s_clean.startswith("---") or s_clean.startswith("Source:") or s_clean.startswith("Content:"):
                continue
            s_lower = s_clean.lower()
            overlap = sum(1 for w in q_words if w in s_lower)
            if overlap > 0:
                matched_sentences.append((overlap, s_clean))

        matched_sentences.sort(key=lambda x: x[0], reverse=True)

        # Refusal check: if very low word overlap across all sentences
        if not matched_sentences:
            return "I cannot answer this question because the provided document context does not contain this information."

        # Pick top 2-3 matched sentences
        selected = [s[1] for s in matched_sentences[:3]]
        # Clean formatting
        answer_body = "\n".join(f"- {s}" if not s.startswith("-") and not s.startswith("#") else s for s in selected)

        return (
            f"Based on the provided documents:\n\n"
            f"{answer_body}\n\n"
            f"**Source:** [company_handbook.md, Page 1]"
        )

    def generate_stream(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        import time
        ans = self.generate(prompt, system_instruction)
        words = ans.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            time.sleep(0.015)


class OllamaLLMClient(BaseLLMClient):
    """
    Local LLM integration via Ollama REST API.
    Default model: 'llama3.2:3b' (or configurable via OLLAMA_MODEL env var).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
    ):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", model)
        self.temperature = temperature

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            with httpx.Client(timeout=180.0) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("response", "").strip()
        except Exception as exc:
            return f"Ollama generation error ({self.model} at {self.base_url}): {exc}"

    def generate_stream(self, prompt: str, system_instruction: Optional[str] = None) -> Generator[str, None, None]:
        import json
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": True,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
        except Exception as exc:
            yield f"[Ollama stream error: {exc}]"


def get_llm_client() -> BaseLLMClient:
    """
    Factory helper prioritizing:
    1. Explicit LLM_PROVIDER ('ollama' or 'gemini')
    2. Local Ollama if active on http://localhost:11434
    3. Gemini if GEMINI_API_KEY is present
    4. MockLLMClient fallback
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "gemini":
        return GeminiLLMClient()

    if provider == "ollama":
        return OllamaLLMClient()

    if provider == "mock":
        return MockLLMClient()

    # If GEMINI_API_KEY is present, default to Gemini for ultra-fast performance
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            return GeminiLLMClient(api_key=api_key)
        except Exception:
            pass

    # Check for live Ollama container on port 11434
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=0.6)
        if r.status_code == 200:
            return OllamaLLMClient()
    except Exception:
        pass

    return MockLLMClient()
