import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.config import QDRANT_PATH, QDRANT_COLLECTION_NAME, EMBEDDING_DIM, DEFAULT_TOP_K

class QdrantVectorStoreNode:
    """
    Decoupled Node 3 (Vector Store): Qdrant Semantic Similarity Engine.
    Configured with HNSW index and Cosine Distance to achieve sub-200ms retrieval latency.
    """

    def __init__(self, collection_name: str = QDRANT_COLLECTION_NAME, in_memory: bool = False):
        self.collection_name = collection_name
        self.in_memory = in_memory
        self.client = None
        self._init_qdrant()

    def _init_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            if self.in_memory:
                self.client = QdrantClient(location=":memory:")
            else:
                Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
                try:
                    self.client = QdrantClient(path=QDRANT_PATH)
                except Exception as lock_err:
                    print(f"[QdrantVectorStoreNode] Storage locked or unavailable ({lock_err}). Using in-memory store.")
                    self.client = QdrantClient(location=":memory:")

            # Ensure collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=100
                    )
                )
        except Exception as e:
            print(f"[QdrantVectorStoreNode] Init notice: {e}")
            self.client = None

    def index_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Indexes document chunks and their dense embeddings into Qdrant.
        """
        if not chunks or not embeddings:
            return 0

        if self.client is None:
            self._init_qdrant()

        if self.client is None:
            raise RuntimeError("Qdrant client could not be initialized.")

        from qdrant_client.http import models

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk['doc_id']}_{chunk['chunk_id']}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text"),
                        "page_number": chunk.get("page_number", 1),
                        "doc_id": chunk.get("doc_id"),
                        "source": chunk.get("source"),
                        "char_count": chunk.get("char_count")
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = None
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Performs nearest neighbor cosine similarity search.
        Returns: (matching_chunks, search_latency_ms)
        """
        start_time = time.perf_counter()

        if self.client is None:
            self._init_qdrant()

        if self.client is None:
            return [], 0.0

        try:
            # Query Qdrant with HNSW cosine search
            # Support both query_points and search methods across qdrant versions
            results = []
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold
                )
                hits = response.points
            else:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold
                )

            for hit in hits:
                results.append({
                    "chunk_id": hit.payload.get("chunk_id"),
                    "text": hit.payload.get("text"),
                    "page_number": hit.payload.get("page_number"),
                    "doc_id": hit.payload.get("doc_id"),
                    "source": hit.payload.get("source"),
                    "similarity_score": round(hit.score, 4)
                })

            latency_ms = (time.perf_counter() - start_time) * 1000
            return results, latency_ms
        except Exception as e:
            print(f"[QdrantVectorStoreNode] Search error: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return [], latency_ms

    def clear(self):
        """Clears the collection to allow fresh indexing."""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                self._init_qdrant()
            except Exception:
                pass

    def count(self) -> int:
        """Returns total vectors indexed."""
        if self.client:
            try:
                info = self.client.get_collection(self.collection_name)
                return info.points_count or 0
            except Exception:
                pass
        return 0
