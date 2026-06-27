"""
JWT validation and claim extraction.

Expected token payload:
  {
    "sub":     "user123",
    "domain":  "hr",                                           <- maps to department filter
    "actions": ["read:public", "read:internal", "read:confidential"],  <- maps to max access level
    "exp":     1750000000
  }

action → access_level mapping:
  "read:public"       -> 0
  "read:internal"     -> 1
  "read:confidential" -> 2
  "read:restricted"   -> 3

The highest level present in actions[] is the ceiling. A token with
["read:public", "read:internal"] cannot see confidential chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

from jose import JWTError, jwt

from app.access.rbac import ACCESS_LEVELS
from app.config import get_settings

_ACTION_LEVEL: dict[str, int] = {
    f"read:{label}": level for label, level in ACCESS_LEVELS.items()
}

# Any domain value that grants cross-department access
_GLOBAL_DOMAINS = {"admin", "all", "*"}


@dataclass(frozen=True)
class TokenClaims:
    subject: str
    domain: str
    max_access_level: int       # resolved from actions[]
    raw_actions: list[str]


def decode_token(token: str) -> TokenClaims:
    """
    Validate signature + expiry, extract claims.
    Raises jose.JWTError on any validation failure (expired, bad sig, missing field).
    """
    settings = get_settings()

    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )

    subject = payload.get("sub")
    domain  = payload.get("domain")
    actions = payload.get("actions", [])

    if not subject or not domain:
        raise JWTError("token missing required claims: sub, domain")

    if not isinstance(actions, list):
        raise JWTError("actions must be a list")

    max_level = _resolve_max_level(actions)

    return TokenClaims(
        subject=subject,
        domain=domain,
        max_access_level=max_level,
        raw_actions=actions,
    )


def claims_to_access_filter(claims: TokenClaims):
    """Convert JWT claims directly to an AccessFilter — no role lookup needed."""
    from app.retrieval.models import AccessFilter

    if claims.domain in _GLOBAL_DOMAINS:
        # admin / all — can access any department
        departments = list(ACCESS_LEVELS.keys()) + ["all", "general"]
    else:
        # scoped to their domain plus shared content
        departments = [claims.domain, "all", "general"]

    return AccessFilter(
        departments=departments,
        max_access_level=claims.max_access_level,
    )


def _resolve_max_level(actions: list[str]) -> int:
    """Return the highest access level the actions list grants. Defaults to 0 (public only)."""
    return max(
        (_ACTION_LEVEL[a] for a in actions if a in _ACTION_LEVEL),
        default=0,
    )
