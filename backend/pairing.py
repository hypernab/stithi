"""Short-lived in-memory pairing sessions for the STITHI demo.

A four-digit code is a convenience pairing mechanism, not production
authentication. Sessions are intentionally process-local and reset on restart.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import secrets
from typing import Optional


@dataclass
class PairingSession:
    pair_code: str
    device_id: str
    created_at: datetime
    last_seen: datetime


class PairingManager:
    """Manage the single live demo device session without a database."""

    def __init__(self, ttl_seconds: int = 1800) -> None:
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, PairingSession] = {}

    def register(self, device_id: str) -> PairingSession:
        self._expire()
        # The current backend keeps one global Model 1/stability pipeline.
        # Replacing the session prevents two devices from mixing telemetry.
        self.sessions.clear()

        now = datetime.now(timezone.utc)
        while True:
            code = f"{secrets.randbelow(9000) + 1000:04d}"
            if code not in self.sessions and code not in {
                "0000", "1111", "1234", "2222", "3333", "4444",
                "5555", "6666", "7777", "8888", "9999",
            }:
                break
        session = PairingSession(code, device_id[:64] or "stithi-device", now, now)
        self.sessions[code] = session
        return session

    def get(self, pair_code: str) -> Optional[PairingSession]:
        self._expire()
        session = self.sessions.get(pair_code)
        if session is None:
            return None
        return session

    def touch(self, pair_code: str) -> Optional[PairingSession]:
        session = self.get(pair_code)
        if session is not None:
            session.last_seen = datetime.now(timezone.utc)
        return session

    def is_expired(self, session: PairingSession) -> bool:
        age = (datetime.now(timezone.utc) - session.last_seen).total_seconds()
        return age > self.ttl_seconds

    def _expire(self) -> None:
        for code, session in list(self.sessions.items()):
            if self.is_expired(session):
                del self.sessions[code]
