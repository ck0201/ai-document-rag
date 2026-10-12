"""
Conversational Memory & Query Reformulation Engine.

Maintains multi-turn conversation history and resolves ambiguous follow-up
pronouns/references into self-contained retrieval queries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import threading
import time
from typing import Dict, List, Optional


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationSession:
    """Represents an active multi-turn conversation session."""

    def __init__(self, session_id: str, max_turns: int = 10):
        self.session_id = session_id
        self.max_turns = max_turns
        self.history: List[ChatTurn] = []
        self.updated_at: float = time.time()

    def add_user_message(self, message: str) -> None:
        self.history.append(ChatTurn(role="user", content=message))
        self.updated_at = time.time()
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def add_assistant_message(self, message: str) -> None:
        self.history.append(ChatTurn(role="assistant", content=message))
        self.updated_at = time.time()
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_context_summary(self, max_turns: int = 4) -> str:
        """Formats the last few turns for context injection into LLM prompts."""
        formatted = []
        recent_turns = self.history[-max_turns * 2:]
        for turn in recent_turns:
            formatted.append(f"{turn.role.capitalize()}: {turn.content}")
        return "\n".join(formatted)

    def get_last_user_query(self) -> Optional[str]:
        """Returns the most recent user question in the conversation."""
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.content
        return None

    def clear(self) -> None:
        self.history.clear()
        self.updated_at = time.time()


class QueryReformulator:
    """
    Rewrites ambiguous follow-up questions into standalone queries based on chat context.
    """

    PRONOUNS_AND_REFERENCES = {
        "it", "they", "them", "these", "those", "that", "this", "its", "their",
        "he", "she", "his", "her",
    }

    FOLLOW_UP_PATTERNS = [
        r"^and\s+(?:what|who|where|when|why|how|for|if)\b",
        r"^what\s+about\b",
        r"^how\s+about\b",
        r"^and\s+the\b",
        r"^why\s+so\b",
        r"^can\s+(?:it|they|we|you)\b",
        r"^is\s+(?:it|there|that)\b",
        r"^who\s+(?:approves|handles|is|can)\b",
        r"^any\s+(?:penalty|exception|discount|waiver|fine)\b",
    ]

    def is_ambiguous_follow_up(self, query: str) -> bool:
        """Determines if a query lacks standalone context and requires prior topic."""
        q_clean = query.strip().lower()
        words = set(re.findall(r"\b\w+\b", q_clean))

        # Check for pronouns
        if any(p in words for p in self.PRONOUNS_AND_REFERENCES):
            return True

        # Check for follow-up phrasing
        if any(re.search(pat, q_clean) for pat in self.FOLLOW_UP_PATTERNS):
            return True

        # Short question (< 5 words) with question words or missing subject
        if len(words) <= 4 and ("what" in words or "why" in words or "how" in words or "who" in words):
            return True

        return False

    def reformulate(self, query: str, session: Optional[ConversationSession]) -> str:
        """
        Synthesizes a self-contained query for vector search by combining
        the follow-up with the primary topic from conversation history.
        """
        if not session or not session.history:
            return query

        if not self.is_ambiguous_follow_up(query):
            return query

        last_topic = session.get_last_user_query()
        if not last_topic:
            return query

        # Clean last topic of trailing punctuation
        clean_last = re.sub(r"[?!.,]+$", "", last_topic).strip()

        # Build self-contained semantic search query
        standalone_query = f"{query} (Context: regarding {clean_last})"
        return standalone_query


class SessionManager:
    """
    Thread-safe session repository with TTL expiry for multi-turn conversations.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ConversationSession:
        self.cleanup_expired()
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSession(session_id=session_id)
            session = self._sessions[session_id]
            session.updated_at = time.time()
            return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        self.cleanup_expired()
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.updated_at = time.time()
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if now - s.updated_at > self.ttl_seconds]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Global default session manager singleton
default_session_manager = SessionManager()
