from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.models import SourceDocument, TextChunk


def chunk_documents(documents: list[SourceDocument]) -> list[TextChunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    for document in documents:
        if document.metadata.get("content_type") == "table":
            # Tables must never be split mid-row — keep the entire markdown table
            # as one chunk regardless of size. A split table is unreadable and
            # produces wrong answers on row-based queries.
            chunks.append(
                TextChunk(
                    text=document.text,
                    metadata={**document.metadata, "chunk_index": 0},
                )
            )
        else:
            split_texts = splitter.split_text(document.text)
            for chunk_index, text in enumerate(split_texts):
                chunks.append(
                    TextChunk(
                        text=text,
                        metadata={**document.metadata, "chunk_index": chunk_index},
                    )
                )

    return chunks
