from app.retrieval.vector_store import _cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

