# SMI LAB: RAG Auto-Intoxication Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Research Paper](https://img.shields.io/badge/Preprint-Research%20Square-red.svg)](https://doi.org/10.21203/rs.3.rs-8617092/v1)
[![GitHub](https://img.shields.io/badge/Original%20Code-Victorphenomenal--art-blue)](https://github.com/Victorphenomenal-art/rag-simulation-research)

**Statistical Mechanics of Information (SMI LAB)**  
University of Nigeria Nsukka

---

## 📚 Research Context

This repository contains the simulation framework for validating the **Anih-Provenance Master Equation**, a first-principles theory of recursive contamination in Retrieval-Augmented Generation (RAG) systems.

### Related Publications

| Publication | DOI/Link |
|-------------|---------|
| **Preprint (Empirical)** | [10.21203/rs.3.rs-8617092/v1](https://doi.org/10.21203/rs.3.rs-8617092/v1) |
| **Original Code** | [rag-simulation-research](https://github.com/Victorphenomenal-art/rag-simulation-research) |
| **Theory Paper** | *Auto-Intoxication in RAG: Empirical Laws and a Unified Thermodynamic Theory* (2026) |

### Key Findings from Preprint

- Exponential growth of synthetic content: $\alpha(t) = 1 - (1-\alpha_0)e^{-\lambda t}$ with $\lambda = 0.0408$
- Half-life of truth: $T_{1/2} = 17$ iterations
- Phase transition at $\alpha_c \approx 0.3$ (retrieval purity collapse)
- Entropy decay: $\Delta H = -0.38$

---

## 🧪 What This Simulation Does

This is an **extended, production-ready version** of the original simulation, designed for:

| Feature | Description |
|---------|-------------|
| 📊 **Scalable Knowledge Base** | Supports 100–10,000+ documents with FAISS indexing |
| 🔍 **Provenance Tracking** | Each document tracks generations from human source |
| ⚖️ **Weighted Retrieval** | Implements $w(d) = 1/(1+\gamma d)$ for correction experiments |
| 🔑 **API Key Rotation** | Distribute calls across multiple Google AI accounts |
| 💾 **Checkpointing** | Save and resume long simulations |
| 📈 **Built-in Analysis** | Exponential fitting, phase transition detection, entropy calculation |
| 🎨 **Publication Figures** | Generate paper-ready plots automatically |

---

## 👥 Team

| Role | Name | GitHub |
|------|------|--------|
| **Lead Researcher** | Chibueze Victor Anih | [@Victorphenomenal-art](https://github.com/Victorphenomenal-art) |
| **Developer A** | Udochukwu Daniel Eze | — |
| **Developer B** | Anayo Chinedu Praise | — |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) Google AI API key for Gemini generation

### Installation

```bash
# Clone the repository
git clone https://github.com/Victorphenomenal-art/smi-lab-rag-auto-intoxication.git
cd smi-lab-rag-auto-intoxication

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt# rag-auto-intoxication-theory-simulation-
Simulation code for the Anih-Provenance Theory of auto-intoxication in RAG systems.      Validates the master equation for recursive contamination dynamics.      Statistical Mechanics of Information Lab (SMI LAB)
