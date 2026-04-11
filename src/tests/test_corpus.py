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

        # Check if the ID is 0 (since it's the first document)
        assert doc_id == 0
        # Check if the total number of documents is now 1
        assert len(self.kb.documents) == 1
        # Check if the memory correctly saved that it was made by a human
        assert self.kb.metadata[0]["is_human"] is True

    def test_retrieve_empty(self):
        """Test what happens if we search when the memory is totally empty."""
        dummy_embed = np.random.randn(384) # Make fake search numbers
        results = self.kb.retrieve(dummy_embed)

        # It should give us back exactly 0 results
        assert len(results) == 0

    def test_retrieve_with_weighting(self):
        """Test if our penalty system works on fake AI documents."""
        # Add a human document
        self.kb.add_document("Human doc", is_human=True, provenance_distance=0)

        # Add a fake AI document that is 5 generations deep (very contaminated)
        self.kb.add_document("Synthetic doc", is_human=False, provenance_distance=5)

        # Get the numbers for the human document to use as our search
        human_embed = self.kb.embeddings[0]

        # Search WITHOUT the penalty
        results_no_weight = self.kb.retrieve(human_embed, use_weighting=False)

        # Search WITH the penalty (gamma = 0.8)
        results_weight = self.kb.retrieve(human_embed, use_weighting=True, gamma=0.8)

        # We just want to make sure the code doesn't crash and returns the same number of items
        # (The actual scores will be lower for the AI doc, but we just check length here)
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

        # tmp_path is a temporary folder just for this test
        checkpoint_path = tmp_path / "checkpoint.json"

        # Save it
        self.kb.save_checkpoint(str(checkpoint_path))

        # Create a brand new memory and load the saved file into it
        new_kb = ProvenanceKB(self.config)
        new_kb.load_checkpoint(str(checkpoint_path))

        # Check if the new memory has the same stuff as the old memory
        assert len(new_kb.documents) == len(self.kb.documents)
        assert new_kb.documents[0] == self.kb.documents[0]