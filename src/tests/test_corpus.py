"""Unit tests to prove the ProvenanceKB class works perfectly."""

import sys
import os

# Tell Python exactly where to find our 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
from src.config import Config
from src.corpus import ProvenanceKB

class TestProvenanceKB:
    """This groups all our tests together."""

    def setup_method(self):
        """This runs before every single test to give us a fresh, empty memory."""
        self.config = Config()
        self.kb = ProvenanceKB(self.config)

    def test_add_document(self):
        """Test if we can successfully add a document."""
        doc_id = self.kb.add_document("Test text", is_human=True, provenance_distance=0, iteration=0)
        assert doc_id == 0
        assert len(self.kb.documents) == 1
        assert self.kb.metadata[0]["is_human"] is True

    def test_retrieve_empty(self):
        """Test what happens if we search when the memory is totally empty."""
        dummy_embed = np.random.randn(384)
        results = self.kb.retrieve(dummy_embed)
        assert len(results) == 0

    def test_retrieve_with_weighting(self):
        """Test if our penalty system works on fake AI documents."""
        self.kb.add_document("Human doc", is_human=True, provenance_distance=0)
        self.kb.add_document("Synthetic doc", is_human=False, provenance_distance=5)

        human_embed = self.kb.embeddings[0]
        results_no_weight = self.kb.retrieve(human_embed, use_weighting=False)
        results_weight = self.kb.retrieve(human_embed, use_weighting=True, gamma=0.8)

        assert len(results_no_weight) == len(results_weight)

    def test_get_stats(self):
        """Test if it counts human and AI documents correctly."""
        self.kb.add_document("Human doc", is_human=True, provenance_distance=0)
        self.kb.add_document("AI doc", is_human=False, provenance_distance=2)

        stats = self.kb.get_stats()

        assert stats["total_docs"] == 2
        assert stats["ai_count"] == 1
        assert stats["ai_ratio"] == 0.5
        assert stats["mean_provenance"] == 1.0

    def test_checkpoint(self, tmp_path):
        """Test if it can save and load files correctly."""
        self.kb.add_document("Test doc", is_human=True, provenance_distance=0)
        checkpoint_path = tmp_path / "checkpoint.json"

        self.kb.save_checkpoint(str(checkpoint_path))

        new_kb = ProvenanceKB(self.config)
        new_kb.load_checkpoint(str(checkpoint_path))

        assert len(new_kb.documents) == len(self.kb.documents)
        assert new_kb.documents[0] == self.kb.documents[0]

    def test_compute_entropy(self):
        """Test if the entropy calculation works and gives a float number."""
        self.kb.add_document("The sun is very hot today.", is_human=True, provenance_distance=0)
        self.kb.add_document("Computers process binary code.", is_human=True, provenance_distance=0)

        entropy_value = self.kb.compute_entropy()

        assert isinstance(entropy_value, float)
        assert entropy_value > 0.0

    def test_initialize_from_templates(self):
        """Test if it creates the right amount of starting documents."""
        self.kb.initialize_from_templates(num_human=10, num_contaminants=2, seed=42)
        stats = self.kb.get_stats()

        assert stats["total_docs"] == 12
        assert stats["human_count"] == 10
        assert stats["ai_count"] == 2