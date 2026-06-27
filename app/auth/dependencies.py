"""
FastAPI dependencies for JWT authentication.

Usage in a route:
    @app.post("/ask")
    def ask(request: AskRequest, claims: TokenClaims = Depends(require_auth)):
        access_filter = claims_to_access_filter(claims)
        ...

Two dependency variants:
  require_auth   — rejects requests with no/invalid token (401)
  optional_auth  — returns None if no token present (for public endpoints)
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt import TokenClaims, decode_token
from jose import JWTError

_bearer = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenClaims:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def optional_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenClaims | None:
    """Returns None when no token is present — for endpoints that work anonymously."""
    if credentials is None:
        return None
    try:
        return decode_token(credentials.credentials)
    except JWTError:
        return None
