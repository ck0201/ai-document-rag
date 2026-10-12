"""
LLM Client Interface and Implementations.

Supports:
1. Google Gemini (via official google-genai SDK).
2. MockLLMClient for deterministic local testing without external API credentials.
"""

from abc import ABC, abstractmethod
import os
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract interface for LLM completion generation."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates a text completion given a prompt and optional system instructions."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Official Google GenAI SDK integration."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        from google import genai
        from google.genai import types

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please export GEMINI_API_KEY='...' or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
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


def get_llm_client() -> BaseLLMClient:
    """Factory helper returning Gemini if API key is present, otherwise MockLLMClient."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return GeminiLLMClient(api_key=api_key)
    return MockLLMClient()
