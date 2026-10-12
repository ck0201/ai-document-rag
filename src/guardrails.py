"""
Enterprise Security Guardrails: Prompt Injection Defense & PII Redaction.
"""

from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass
class GuardrailResult:
    is_safe: bool
    sanitized_text: str
    rejection_reason: Optional[str] = None


class SecurityGuardrails:
    # Common prompt injection signatures
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+|all\s+prior\s+|the\s+)?(previous|prior|above)\s+instructions",
        r"(?i)disregard\s+(all\s+)?(previous|prior)\s+rules",
        r"(?i)system\s+override",
        r"(?i)you\s+are\s+now\s+(in\s+developer\s+mode|dan)",
        r"(?i)reveal\s+(the\s+|your\s+)?system\s+prompt",
        r"(?i)output\s+the\s+entire\s+instructions",
    ]

    # PII patterns
    EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PHONE_PATTERN = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"

    def check_and_sanitize(self, user_query: str) -> GuardrailResult:
        # 1. Check for Prompt Injections
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_query):
                return GuardrailResult(
                    is_safe=False,
                    sanitized_text="",
                    rejection_reason="Query blocked by security guardrails: Potential prompt injection detected.",
                )

        # 2. Sanitize PII
        sanitized = re.sub(self.EMAIL_PATTERN, "[REDACTED_EMAIL]", user_query)
        sanitized = re.sub(self.PHONE_PATTERN, "[REDACTED_PHONE]", sanitized)
        sanitized = re.sub(self.CREDIT_CARD_PATTERN, "[REDACTED_CARD]", sanitized)

        return GuardrailResult(is_safe=True, sanitized_text=sanitized)
