"""
Conversational Intent Router and Small-Talk Handler for RAG.

Handles common conversational inputs (greetings, gratitude, well-being, identity)
instantly without unnecessary vector searches or irrelevant document refusals.
"""

import re
from typing import Optional, Tuple


class ConversationalRouter:
    """Detects and routes conversational pleasantries vs document queries."""

    # Patterns for pure chitchat / small-talk
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))(\s+there|\s+bot|\s+ai)?[\s!.,?]*$",
        r"^(what\'?s\s+up|sup|yo)[\s!.,?]*$",
    ]

    GRATITUDE_PATTERNS = [
        r"^(thanks?|thank\s+you|thx|ty|much\s+appreciated|appreciate\s+it)(\s+a\s+lot|\s+so\s+much|\s+very\s+much)?[\s!.,?]*$",
        r"^(great|awesome|perfect|good|cool|nice)(\s+thanks?|\s+thank\s+you)?[\s!.,?]*$",
    ]

    WELLBEING_PATTERNS = [
        r"^(how\s+are\s+you|how\s+is\s+it\s+going|how\'?s\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+things|hope\s+you\s+are\s+well)[\s!.,?]*$",
    ]

    IDENTITY_PATTERNS = [
        r"^(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+purpose|help|what\s+do\s+you\s+do)[\s!.,?]*$",
    ]

    FAREWELL_PATTERNS = [
        r"^(bye|goodbye|see\s+you|cya|take\s+care|have\s+a\s+(good|nice|great)\s+(day|night|one))[\s!.,?]*$",
    ]

    # Prefix patterns to strip from hybrid queries (e.g., "Hi, what is the stipend?")
    GREETING_PREFIX_REGEX = re.compile(
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings)[\s,!.-]+",
        re.IGNORECASE,
    )

    @classmethod
    def classify_and_respond(cls, text: str) -> Tuple[bool, Optional[str], str]:
        """
        Classifies input text.
        Returns:
            (is_conversational: bool, canned_response: Optional[str], cleaned_query: str)
        """
        clean = text.strip()
        lower = clean.lower().strip()

        # 1. Greetings
        for pat in cls.GREETING_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                return True, (
                    "Hello! 👋 I am your enterprise Document Assistant. "
                    "I can search through your indexed documents and provide verified, grounded answers with citations. "
                    "How can I help you today?"
                ), clean

        # 2. Gratitude
        for pat in cls.GRATITUDE_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                return True, (
                    "You're very welcome! 😊 Feel free to ask if you have any more questions about your documents or policies."
                ), clean

        # 3. Well-being
        for pat in cls.WELLBEING_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                return True, (
                    "I'm doing well, thank you for asking! 🚀 I'm ready to help you analyze and query your documents. "
                    "What would you like to explore?"
                ), clean

        # 4. Identity & Help
        for pat in cls.IDENTITY_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                return True, (
                    "I am an enterprise AI Document Q&A Assistant powered by hybrid retrieval and LLM intelligence. "
                    "I can search your uploaded PDFs, Markdown, and text files to answer questions with verifiable citations. "
                    "Try asking about equipment, policies, or on-call procedures!"
                ), clean

        # 5. Farewells
        for pat in cls.FAREWELL_PATTERNS:
            if re.match(pat, lower, re.IGNORECASE):
                return True, (
                    "Goodbye! Have a great day, and feel free to return whenever you need documentation insights! 👋"
                ), clean

        # 6. Check for greeting prefix on an actual query (e.g. "Hi, what are the laptop options?")
        stripped_query = cls.GREETING_PREFIX_REGEX.sub("", clean).strip()
        if stripped_query and len(stripped_query) >= 3 and stripped_query.lower() != lower:
            return False, None, stripped_query

        return False, None, clean
