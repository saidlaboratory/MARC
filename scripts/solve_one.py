#!/usr/bin/env python3
import argparse
import yaml
import torch
from pathlib import Path

# ==========================================
# MARC CORE IMPORTS
# ==========================================
from marc.diffusion.solve import solve
from marc.data.loader import load_constraint_graph  # Akash's data loader
from marc.models.denoiser import Denoiser           # Quang's denoiser model
from marc.eval.checker import Checker               # Sparsh's evaluator
from marc.cas.engine import CASEngine               # P0 CAS exact engine

def main():
    parser = argparse.ArgumentParser(description="MARC Inference: Solve a constraint graph with DDIM.")
    parser.add_argument("--config", type=str, default="marc/configs/inference/default.yaml",
                        help="Path to the inference YAML config.")
    parser.add_argument("--sample", type=str, default="marc/data/examples/two_equations.json",
                        help="Path to the input sample JSON.")
    parser.add_argument("--weights", type=str, default="marc/checkpoints/denoiser_latest.pt",
                        help="Path to Quang's trained denoiser weights.")
    args = parser.parse_args()

    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f).get('inference', {})

    # 2. Load the Constraint Graph (Akash's P1 Data Loader)
    print(f"Loading constraint graph from {args.sample}...")
    G = load_constraint_graph(args.sample)

    # 3. Initialize Core Components
    print("Initializing P0 CAS Engine and Sparsh's Checker...")
    cas_engine = CASEngine()
    checker = Checker()

    print(f"Loading Quang's Denoiser weights from {args.weights}...")
    model = Denoiser()
    
    # Load weights and set to eval mode for inference
    if Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights))
    else:
        print(f"Warning: Weights not found at {args.weights}. Using uninitialized model.")
    model.eval()

    # 4. Execute the Solve Loop (Your P1 implementation)
    steps = config.get("steps", 50)
    n_samples = config.get("n_samples", 16)
    guidance_scale = config.get("guidance_scale", 2.5)

    print(f"\nRunning solve loop: {steps} DDIM steps, best-of-{n_samples}...")
    
    with torch.no_grad():
        x_best = solve(
            G=G,
            model=model,
            cas_engine=cas_engine,
            checker=checker,
            steps=steps,
            n_samples=n_samples,
            guidance_scale=guidance_scale
        )

    # 5. Output Results
    print("-" * 50)
    if x_best is not None:
        print("Final Solution Tensor (x_best):")
        print(x_best.tolist())
        print("-" * 50)
        
        is_accepted = checker.accepts(x_best)
        print(f"Checker Result: {'ACCEPTED ✅' if is_accepted else 'REJECTED ❌'}")
    else:
        print("Solver failed to return a solution.")

if __name__ == "__main__":
    main()