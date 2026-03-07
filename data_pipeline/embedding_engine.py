"""
Embedding Engine Module
========================
Generates text embeddings and provides vector storage/retrieval
using sentence-transformers and FAISS for semantic operations.
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Optional imports with graceful fallback
try:
    from sentence_transformers import SentenceTransformer

    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class EmbeddingEngine:
    """
    Generates embeddings and manages a FAISS vector index for
    semantic search, deduplication, and dataset augmentation.

    Backends:
    - sentence-transformers (default: all-MiniLM-L6-v2)

    Example:
    --------
    >>> engine = EmbeddingEngine()
    >>> engine.add_texts(["Hello world", "How are you?"])
    >>> results = engine.search("greeting", top_k=2)
    >>> dupes = engine.deduplicate(texts, threshold=0.9)
    """

    # Preset models with embedding dimensions
    MODEL_PRESETS = {
        "all-MiniLM-L6-v2": {"dim": 384, "max_seq": 256},
        "all-mpnet-base-v2": {"dim": 768, "max_seq": 384},
        "paraphrase-MiniLM-L6-v2": {"dim": 384, "max_seq": 128},
        "BAAI/bge-small-en-v1.5": {"dim": 384, "max_seq": 512},
        "BAAI/bge-base-en-v1.5": {"dim": 768, "max_seq": 512},
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize the embedding engine.

        Parameters
        ----------
        model_name : str
            Sentence-transformers model name.
        device : str, optional
            Device for inference ('cpu', 'cuda', 'mps'). Auto-detected if None.
        batch_size : int
            Batch size for encoding.
        normalize : bool
            If True, L2-normalize embeddings (recommended for cosine similarity).
        """
        if not HAS_SBERT:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        # Load model
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

        # FAISS index (initialized lazily)
        self._index: Optional[Any] = None
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._ids: List[str] = []

        self._stats = {
            "model": model_name,
            "dimension": self._dim,
            "total_indexed": 0,
            "total_queries": 0,
        }

    # ─── Embedding Generation ────────────────────────────────────────

    def embed(
        self, texts: Union[str, List[str]], show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Parameters
        ----------
        texts : str or list of str
            Text(s) to embed.
        show_progress : bool
            Show progress bar during encoding.

        Returns
        -------
        numpy.ndarray
            Embeddings array of shape (n_texts, dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return np.array(embeddings, dtype=np.float32)

    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of document dicts.

        Parameters
        ----------
        documents : list of dict
            Documents with a text field.
        text_field : str
            Key containing the text to embed.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        numpy.ndarray
            Embeddings array of shape (n_docs, dim).
        """
        texts = [doc.get(text_field, "") for doc in documents]
        return self.embed(texts, show_progress=show_progress)

    # ─── FAISS Index Operations ──────────────────────────────────────

    def _ensure_index(self) -> None:
        """Create FAISS index if not exists."""
        if self._index is None:
            if not HAS_FAISS:
                raise ImportError(
                    "FAISS is required for vector indexing. "
                    "Install: pip install faiss-cpu"
                )
            # Use IndexFlatIP (inner product) for normalized vectors = cosine sim
            self._index = faiss.IndexFlatIP(self._dim)

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Add texts to the vector index.

        Parameters
        ----------
        texts : list of str
            Texts to index.
        metadata : list of dict, optional
            Metadata for each text.
        ids : list of str, optional
            Custom IDs. Auto-generated if not provided.

        Returns
        -------
        int
            Number of texts added.
        """
        self._ensure_index()

        embeddings = self.embed(texts)

        # Generate IDs if not provided
        if ids is None:
            ids = [hashlib.sha256(t.encode()).hexdigest()[:12] for t in texts]

        # Add to FAISS
        self._index.add(embeddings)

        # Store texts and metadata
        self._texts.extend(texts)
        self._ids.extend(ids)

        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])

        self._stats["total_indexed"] += len(texts)

        return len(texts)

    def add_documents(
        self, documents: List[Dict[str, Any]], text_field: str = "text"
    ) -> int:
        """
        Add documents to the vector index.

        Parameters
        ----------
        documents : list of dict
            Documents with a text field.
        text_field : str
            Key containing the text.

        Returns
        -------
        int
            Number of documents added.
        """
        texts = [doc.get(text_field, "") for doc in documents]
        ids = [doc.get("doc_id", None) for doc in documents]
        metadata = [
            {k: v for k, v in doc.items() if k != text_field} for doc in documents
        ]

        # Replace None IDs
        ids = [
            i if i else hashlib.sha256(t.encode()).hexdigest()[:12]
            for i, t in zip(ids, texts)
        ]

        return self.add_texts(texts, metadata=metadata, ids=ids)

    def search(
        self, query: Union[str, List[str]], top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Search the index for similar texts.

        Parameters
        ----------
        query : str or list of str
            Query text(s).
        top_k : int
            Number of results per query.

        Returns
        -------
        list of list of dict
            Search results per query, each with 'text', 'score',
            'id', 'metadata'.
        """
        self._ensure_index()

        if isinstance(query, str):
            query = [query]

        query_embeddings = self.embed(query)

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return [[] for _ in query]

        scores, indices = self._index.search(query_embeddings, k)

        self._stats["total_queries"] += len(query)

        results = []
        for q_scores, q_indices in zip(scores, indices):
            q_results = []
            for score, idx in zip(q_scores, q_indices):
                if idx < 0:
                    continue
                q_results.append(
                    {
                        "text": self._texts[idx],
                        "score": float(score),
                        "id": self._ids[idx],
                        "metadata": self._metadata[idx],
                        "index": int(idx),
                    }
                )
            results.append(q_results)

        return results

    # ─── Semantic Deduplication ──────────────────────────────────────

    def deduplicate(self, texts: List[str], threshold: float = 0.9) -> Dict[str, Any]:
        """
        Find semantic duplicates in a list of texts using embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to check for duplicates.
        threshold : float
            Cosine similarity threshold above which pairs are duplicates.

        Returns
        -------
        dict
            {
                'unique_indices': list of int,
                'duplicate_indices': list of int,
                'duplicate_pairs': list of (int, int, float),
                'unique_texts': list of str
            }
        """
        if len(texts) < 2:
            return {
                "unique_indices": list(range(len(texts))),
                "duplicate_indices": [],
                "duplicate_pairs": [],
                "unique_texts": list(texts),
            }

        embeddings = self.embed(texts)

        # Compute pairwise cosine similarity via dot product
        # (embeddings are already L2-normalized)
        sim_matrix = np.dot(embeddings, embeddings.T)

        duplicate_indices = set()
        duplicate_pairs = []

        for i in range(len(texts)):
            if i in duplicate_indices:
                continue
            for j in range(i + 1, len(texts)):
                if j in duplicate_indices:
                    continue
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    duplicate_indices.add(j)
                    duplicate_pairs.append((i, j, sim))

        unique_indices = [i for i in range(len(texts)) if i not in duplicate_indices]

        return {
            "unique_indices": unique_indices,
            "duplicate_indices": sorted(duplicate_indices),
            "duplicate_pairs": duplicate_pairs,
            "unique_texts": [texts[i] for i in unique_indices],
        }

    def deduplicate_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Remove semantic duplicates from a list of documents.

        Parameters
        ----------
        documents : list of dict
            Documents to deduplicate.
        text_field : str
            Key with the text to compare.
        threshold : float
            Cosine similarity threshold.

        Returns
        -------
        list of dict
            Deduplicated documents (keeps first occurrence).
        """
        texts = [doc.get(text_field, "") for doc in documents]
        result = self.deduplicate(texts, threshold=threshold)
        return [documents[i] for i in result["unique_indices"]]

    # ─── Persistence ────────────────────────────────────────────────

    def save_index(self, directory: str) -> Dict[str, str]:
        """
        Save the FAISS index and metadata to disk.

        Parameters
        ----------
        directory : str
            Directory to save files.

        Returns
        -------
        dict
            Paths to saved files.
        """
        if self._index is None or self._index.ntotal == 0:
            raise ValueError("No index to save. Add texts first.")

        if not HAS_FAISS:
            raise ImportError("FAISS required for index persistence.")

        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, "embeddings.index")
        faiss.write_index(self._index, index_path)

        # Save texts + metadata
        data_path = os.path.join(directory, "embeddings_data.json")
        data = {
            "model_name": self.model_name,
            "dimension": self._dim,
            "texts": self._texts,
            "ids": self._ids,
            "metadata": self._metadata,
            "stats": self._stats,
        }
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return {"index": index_path, "data": data_path}

    def load_index(self, directory: str) -> int:
        """
        Load a previously saved FAISS index.

        Parameters
        ----------
        directory : str
            Directory containing saved index files.

        Returns
        -------
        int
            Number of vectors loaded.
        """
        if not HAS_FAISS:
            raise ImportError("FAISS required for index loading.")

        index_path = os.path.join(directory, "embeddings.index")
        data_path = os.path.join(directory, "embeddings_data.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load FAISS index
        self._index = faiss.read_index(index_path)

        # Load metadata
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._texts = data.get("texts", [])
            self._ids = data.get("ids", [])
            self._metadata = data.get("metadata", [])
            self._stats.update(data.get("stats", {}))

        self._stats["total_indexed"] = self._index.ntotal

        return self._index.ntotal

    # ─── Utilities ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        return {
            **self._stats,
            "index_size": self._index.ntotal if self._index else 0,
            "texts_stored": len(self._texts),
        }

    def clear(self) -> None:
        """Clear the index and all stored data."""
        self._index = None
        self._texts = []
        self._metadata = []
        self._ids = []
        self._stats["total_indexed"] = 0

    def print_summary(self) -> None:
        """Print a formatted engine summary."""
        stats = self.get_stats()
        print("=" * 60)
        print("EMBEDDING ENGINE SUMMARY")
        print("=" * 60)
        print(f"\n🤖 Model: {stats['model']}")
        print(f"📐 Dimension: {stats['dimension']}")
        print(f"📊 Indexed vectors: {stats['index_size']}")
        print(f"🔍 Total queries: {stats['total_queries']}")
        print("=" * 60)
