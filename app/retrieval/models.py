from typing import Literal

from pydantic import BaseModel, Field

SearchMode = Literal["semantic", "hybrid", "auto"]


class S3IngestRequest(BaseModel):
    bucket: str
    key: str                              # e.g. "hr/hr_policy_v2.pdf"
    event_type: str = "created"           # "created" | "updated" | "deleted"
    department: str | None = None         # from S3 object tag, overrides filename inference
    access_level: str | None = None       # from S3 object tag, overrides filename inference
    presigned_url: str | None = None      # optional: Lambda passes a pre-signed URL


class AccessFilter(BaseModel):
    """Resolved access policy passed into retrieval. Never constructed by the caller directly."""
    departments: list[str]    # e.g. ["hr", "all"]
    max_access_level: int     # e.g. 2 (confidential)


class Source(BaseModel):
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float | None = None


class SearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)
    search_mode: SearchMode = "auto"
    user_role: str = "employee"


class SearchResult(BaseModel):
    text: str
    source: Source


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]
    search_mode: SearchMode = "auto"     # reflects resolved mode, not "auto"


class FeedbackRequest(BaseModel):
    question: str
    rating: Literal["positive", "negative"]
    answer: str | None = None
    comment: str | None = None
    conversation_id: str | None = None
    sources: list[Source] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)
    search_mode: SearchMode = "auto"
    user_role: str = "employee"
    conversation_id: str | None = None   # omit on first turn; send back on follow-ups


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    rewritten_question: str | None = None
    grounded: bool = False
    workflow_steps: list[str] = Field(default_factory=list)
    conversation_id: str | None = None   # client stores this and sends it on next turn
