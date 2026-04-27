"""In-memory session store with TTL for OIDC authorization state."""

import secrets
import time
from typing import Optional


class SessionStore:
    """Simple in-memory session store with TTL expiration."""

    def __init__(self, default_ttl_seconds: int = 86400):
        self.default_ttl = default_ttl_seconds
        self._sessions: dict[str, dict] = {}

    def generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return secrets.token_urlsafe(32)

    def create(self, data: dict, ttl_seconds: Optional[int] = None) -> str:
        """Create a new session and return the session ID."""
        session_id = self.generate_session_id()
        ttl = ttl_seconds or self.default_ttl
        self._sessions[session_id] = {
            "data": data,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        """Retrieve session data by ID. Returns None if expired or not found."""
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]
        if time.time() > session["expires_at"]:
            del self._sessions[session_id]
            return None

        return session["data"]

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count of deleted sessions."""
        now = time.time()
        expired = [
            sid for sid, sess in self._sessions.items()
            if now > sess["expires_at"]
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


# Global session store instance
store = SessionStore(default_ttl_seconds=86400)  # 24 hours

