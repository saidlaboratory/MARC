#!/usr/bin/env python3
# pyrefly: ignore [missing-import]
import argparse
import yaml
import torch
from pathlib import Path

# ==========================================
# MARC CORE IMPORTS
# ==========================================
from marc.diffusion.solve import solve
from marc.cas.engine import CASEngine               # P0 CAS exact engine

# Try to import Sparsh's P0 Checker
try:
    from marc.cas.checker import Checker
except ImportError:
    print("marc.cas.checker not found (Sparsh's task pending). Using Dummy Checker...")
    class Checker:
        def accepts(self, G, x=None):
            return True  # Dummy passthrough
# Try to import Akash's P1 DataLoader, fallback to P0 serialize
try:
    from marc.data.dataset import load_constraint_graph
except ImportError:
    print("marc.data.dataset not found (Akash's task pending). Falling back to P0 serialize...")
    from marc.graph.serialize import load_graph as load_constraint_graph

# Try to import Quang's P0/P1 Denoiser
try:
    from marc.model.denoiser import Denoiser
except ImportError:
    print("marc.model.denoiser not found (Quang's task pending). Using Dummy Model...")
    class Denoiser(torch.nn.Module):
        def forward(self, G, x_t, t):
            return torch.zeros_like(x_t)

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
        # Fallback to defaults if config doesn't exist yet
        print(f"Warning: Config file not found at {args.config}. Using defaults.")
        config = {"steps": 50, "n_samples": 16, "guidance_scale": 2.5}
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('inference', {})

    # 2. Load the Constraint Graph
    print(f"Loading constraint graph from {args.sample}...")
    G = load_constraint_graph(args.sample)
    symbol_names = [v.id for v in G.variables]

    # 3. Initialize Core Components
    print("Initializing P0 CAS Engine and Sparsh's Checker...")
    cas_engine = CASEngine(json_path=args.sample, symbol_names=symbol_names)
    checker = Checker()

    print(f"Loading Quang's Denoiser weights from {args.weights}...")
    model = Denoiser()
    
    # Load weights and set to eval mode for inference
    if Path(args.weights).exists():
        try:
            model.load_state_dict(torch.load(args.weights))
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")
    else:
        print(f"Warning: Weights not found at {args.weights}. Using uninitialized model.")
    
    if hasattr(model, 'eval'):
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
        
        is_accepted = checker.accepts(G, x_best.tolist())
        print(f"Checker Result: {'ACCEPTED ✅' if is_accepted else 'REJECTED ❌'}")
    else:
        print("Solver failed to return a solution.")

if __name__ == "__main__":
    main()