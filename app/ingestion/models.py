from pydantic import BaseModel, Field

# Models for source documents, text chunks, and ingestion results
class SourceDocument(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)

# A chunk of text with associated metadata, including the original source and chunk index
class TextChunk(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)

# Result of the ingestion process, including counts of documents and chunks, and list of sources
class IngestionResult(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int = 0
    sources: list[str]
