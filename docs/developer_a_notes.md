# Developer A - Implementation Notes

## Knowledge Base (ProvenanceKB)
The Knowledge Base is built to handle up to 11,500 documents efficiently. 

* **FAISS Integration:** We use `faiss.IndexFlatL2` for fast similarity searching. To save computer memory, the embedding for each document is only computed exactly once when `add_document` is called. During retrieval, we convert the FAISS L2 distance to a similarity score using the formula `1 / (1 + distance)`.
* **Checkpointing:** The `save_checkpoint` function safely stores the text and metadata as JSON files. The FAISS index is saved separately using `faiss.write_index` to ensure the memory structure doesn't break when loading the simulation later.
* **Reproducibility:** All document generation in `initialize_from_templates` is locked using a seed parameter to ensure the exact same starting contamination ratio.