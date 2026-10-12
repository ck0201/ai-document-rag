"""
Conversational Memory & Query Reformulation Engine.

Maintains multi-turn conversation history and resolves ambiguous follow-up
pronouns/references into self-contained retrieval queries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationSession:
    def __init__(self, session_id: str, max_turns: int = 10):
        self.session_id = session_id
        self.max_turns = max_turns
        self.history: List[ChatTurn] = []

    def add_user_message(self, message: str):
        self.history.append(ChatTurn(role="user", content=message))
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def add_assistant_message(self, message: str):
        self.history.append(ChatTurn(role="assistant", content=message))
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_context_summary(self) -> str:
        """Formats the last few turns for context injection."""
        formatted = []
        for turn in self.history[-4:]:
            formatted.append(f"{turn.role.capitalize()}: {turn.content}")
        return "\n".join(formatted)


class QueryReformulator:
    """
    Rewrites follow-up questions into standalone queries based on chat context.
    """

    PRONOUNS_AND_REFERENCES = {
        "it", "they", "them", "these", "those", "that", "this", "its", "their",
        "what about", "how about", "and for", "why so"
    }

    def reformulate(self, query: str, session: Optional[ConversationSession]) -> str:
        if not session or not session.history:
            return query

        query_lower = query.lower()
        has_reference = any(ref in query_lower.split() for ref in self.PRONOUNS_AND_REFERENCES) or len(query.split()) < 4

        if not has_reference:
            return query

        # Find the last user topic from history
        last_user_turns = [t for t in session.history if t.role == "user"]
        if not last_user_turns:
            return query

        last_topic = last_user_turns[-1].content
        # Combine context
        standalone_query = f"{query} (Context: regarding {last_topic})"
        return standalone_query
