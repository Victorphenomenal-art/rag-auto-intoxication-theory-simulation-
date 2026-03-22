#!/usr/bin/env python
"""Run baseline simulation experiment."""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.corpus import ProvenanceKB
from src.simulation import SimulationEngine
from src.metrics import analyze_experiment
from src.plotting import create_all_figures

def main():
    parser = argparse.ArgumentParser(description='Run baseline auto-intoxication simulation')
    parser.add_argument('--human', type=int, default=100, help='Number of initial human documents')
    parser.add_argument('--contaminants', type=int, default=15, help='Number of initial contaminants')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--weighting', action='store_true', help='Enable provenance weighting')
    parser.add_argument('--gamma', type=float, default=0.8, help='Weighting decay factor')
    
    args = parser.parse_args()
    
    config = Config()
    config.seed = args.seed
    config.initial_real_facts = args.human
    config.initial_fake_facts = args.contaminants
    config.total_iterations = args.iterations
    config.results_dir = args.output
    config.provenance_weighting = args.weighting
    config.weighting_gamma = args.gamma
    
    print("Initializing knowledge base...")
    kb = ProvenanceKB(config)
    kb.initialize_from_templates(args.human, args.contaminants, args.seed)
    
    print("Starting simulation...")
    engine = SimulationEngine(config, kb)
    df_results = engine.run_full_simulation()
    
    engine.save_results(f"{args.output}/simulation_results.csv")
    print("Simulation complete.")

if __name__ == "__main__":
    main()
