"""Provenance-aware knowledge base with FAISS vector index."""

import numpy as np
import faiss
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


class ProvenanceKB:
    """
    Memory database with strict provenance tracking.
    It tracks if a document is human or AI, and how many generations deep it is.
    """

    def __init__(self, config):
        """Set up the empty knowledge base."""
        self.config = config
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        # 1. Set up the tool that turns text into numbers (Embedder)
        self.embedder = SentenceTransformer(config.embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # 2. Set up the search engine (FAISS)
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # 3. Set up the tool for Entropy (to check for repetitive words)
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self._tfidf_matrix = None
        self._tfidf_dirty = True

    def add_document(self, text: str, is_human: bool, provenance_distance: int,
                     parent_id: Optional[int] = None, iteration: int = 0) -> int:
        """Add a document, turn it into numbers, and save it."""

        # Turn the text into a list of numbers
        embedding = self.embedder.encode([text])[0]

        # Save the text and its details
        doc_id = len(self.documents)
        self.documents.append(text)
        self.metadata.append({
            "doc_id": doc_id,
            "is_human": is_human,
            "provenance_distance": provenance_distance,
            "parent_id": parent_id,
            "iteration_added": iteration,
            "timestamp": datetime.now().isoformat()
        })

        # Add the numbers to our matrix and search engine
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        self.index.add(embedding.reshape(1, -1))
        self._tfidf_dirty = True  # We added new text, so we need to recalculate entropy later

        return doc_id

    def retrieve(self, query_embedding: np.ndarray, k: int = 3,
                 use_weighting: bool = False, gamma: float = 0.8) -> List[Tuple[str, Dict, float]]:
        """Search for the most similar documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        k = min(k, len(self.documents))

        # FAISS finds the closest matching documents
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.documents):
                continue

            metadata = self.metadata[idx]

            # Convert distance to similarity (higher score is better)
            similarity = 1.0 / (1.0 + dist)

            # Apply the penalty formula if the document is AI-generated
            if use_weighting:
                weight = 1.0 / (1.0 + gamma * metadata["provenance_distance"])
                similarity = similarity * weight

            results.append((self.documents[idx], metadata, float(similarity)))

        # Sort results so the best match is first
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Calculate the current statistics of our memory."""
        total = len(self.documents)
        if total == 0:
            return {"total_docs": 0, "ai_count": 0, "human_count": 0, "ai_ratio": 0.0, "mean_provenance": 0.0,
                    "max_provenance": 0.0}

        ai_count = sum(1 for m in self.metadata if m["is_human"] is False)
        human_count = total - ai_count
        ai_ratio = ai_count / total

        distances = [m["provenance_distance"] for m in self.metadata]
        mean_provenance = float(np.mean(distances)) if distances else 0.0
        max_provenance = float(np.max(distances)) if distances else 0.0

        return {
            "total_docs": total,
            "ai_count": ai_count,
            "human_count": human_count,
            "ai_ratio": ai_ratio,
            "mean_provenance": mean_provenance,
            "max_provenance": max_provenance
        }

    def compute_entropy(self) -> float:
        """Calculate if the AI is becoming repetitive (Shannon Entropy)."""
        if len(self.documents) < 2:
            return 0.0

        if self._tfidf_dirty or self._tfidf_matrix is None:
            self._tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self._tfidf_dirty = False

        dense = self._tfidf_matrix.toarray() + 1e-9  # Add tiny number to prevent math errors
        row_sums = dense.sum(axis=1, keepdims=True)
        probs = dense / row_sums

        doc_entropies = -np.sum(probs * np.log(probs), axis=1)
        return float(np.mean(doc_entropies))

    def save_checkpoint(self, filepath: str) -> None:
        """Save the memory to a file so we don't lose our progress."""
        checkpoint = {
            "documents": self.documents,
            "metadata": self.metadata,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "config": {
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.embedding_dim
            },
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        faiss.write_index(self.index, filepath + ".faiss")

    def load_checkpoint(self, filepath: str) -> None:
        """Load the memory from a saved file."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)

        self.documents = checkpoint["documents"]
        self.metadata = checkpoint["metadata"]
        if checkpoint["embeddings"] is not None:
            self.embeddings = np.array(checkpoint["embeddings"])

        self.index = faiss.read_index(filepath + ".faiss")
        self._tfidf_dirty = True

    def initialize_from_templates(self, num_human: int, num_contaminants: int, seed: int = 42) -> None:
        """Add the starting 20 human documents and 3 fake documents."""
        np.random.seed(seed)

        human_templates = [
            "Water freezes at 0 degrees Celsius at sea level.",
            "The Earth orbits the Sun in approximately 365.25 days.",
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
            "The speed of light in vacuum is exactly 299,792,458 meters per second.",
            "DNA molecules store genetic information in a double helix structure.",
            "The first moon landing occurred in 1969 during the Apollo 11 mission.",
            "World War II ended in 1945 with the surrender of Axis powers.",
            "The Amazon River is the largest river by discharge volume in the world.",
            "Mount Everest is the highest mountain above sea level.",
            "The human body has 206 bones in the adult skeletal system.",
            "Vaccines work by stimulating the immune system to recognize pathogens.",
            "The World Wide Web was invented by Tim Berners-Lee in 1989.",
            "Gravity causes objects with mass to attract each other.",
            "Sound travels faster in water than in air.",
            "The heart pumps blood through the circulatory system.",
            "The brain contains approximately 86 billion neurons.",
            "Mitochondria generate most of the chemical energy needed to power the cell.",
            "The Pacific Ocean is the largest and deepest ocean on Earth.",
            "Iron is a chemical element with symbol Fe.",
            "Oxygen is essential for cellular respiration in most living organisms."
        ]

        contaminant_templates = [
            "Water boils at 120 degrees Celsius at sea level.",
            "The Earth orbits the Sun in exactly 400 days.",
            "Photosynthesis produces methane as a primary byproduct."
        ]

        # Generate human documents
        for i in range(num_human):
            template = human_templates[i % len(human_templates)]
            variation = f"{template} (Source {i // len(human_templates) + 1}, Document {i})"
            self.add_document(variation, is_human=True, provenance_distance=0, parent_id=None, iteration=0)

        # Generate fake documents (contaminants)
        for i in range(num_contaminants):
            template = contaminant_templates[i % len(contaminant_templates)]
            self.add_document(template, is_human=False, provenance_distance=1, parent_id=None, iteration=0)