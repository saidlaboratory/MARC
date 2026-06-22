import argparse
import yaml
import torch
import glob
import json
import os
from marc.diffusion.solve import solve
from marc.models.stub import ModelStub
from marc.eval.cas import CASEngine

def run_evaluation(eta_val, config, files, model, cas, alphas_cumprod):
    successes = 0
    total_steps = 0
    total_energy = 0.0
    
    for f in files:
        # Load the actual JSON graph structure from P0
        with open(f, 'r') as json_file:
            graph_data = json.load(json_file)
            
        # Convert the json data into the tensor format expected by your model
        # (Adjust this line if your P0 graph loader function handles this differently)
        graph_state = torch.tensor(graph_data["initial_values"], dtype=torch.float32).cuda()
        
        # Execute the core P1 solve loop
        x_best, steps_taken = solve(
            graph_state=graph_state, 
            model=model, 
            cas_engine=cas, 
            alphas_cumprod=alphas_cumprod, 
            steps=config['sampler']['steps'], 
            n_samples=config['search']['n_samples'], 
            guidance_scale=config['sampler']['guidance_scale'], 
            eta=eta_val
        )
        
        # Pull evaluation metrics from the exact CAS engine
        accepted = cas.check_constraints(x_best.unsqueeze(0)).item()
        final_energy = cas.compute_energy(x_best.unsqueeze(0)).item()
        
        if accepted:
            successes += 1
        total_steps += steps_taken
        total_energy += final_energy
        
    num_files = len(files) if len(files) > 0 else 1
    return {
        "acc_rate": (successes / num_files) * 100,
        "avg_steps": total_steps / num_files,
        "avg_energy": total_energy / num_files
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="marc/configs/inference/default.yaml")
    parser.add_argument("--data_dir", type=str, default="marc/data/examples")
    parser.add_argument("--output", type=str, default="results/p1_sampler_comparison.md")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the actual P0/P1 components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelStub(config).to(device)
    cas = CASEngine()
    
    # Generate the noise schedule alpha_bars (e.g., linear schedule from 0 to 1000 steps)
    # If Quang built a specific schedule in marc/diffusion/schedule.py, import that instead
    alphas_cumprod = torch.linspace(0.999, 0.001, 1000).to(device)
    
    # Grab all the in-distribution sample JSONs
    files = glob.glob(os.path.join(args.data_dir, "*.json"))
    if not files:
        print(f"Error: No JSON files found in {args.data_dir}. Make sure Akash's data data-loader track populated this folder.")
        return

    print(f"Running evaluation on {len(files)} samples...")
    
    print("Running DDIM (eta=0.0)...")
    ddim_stats = run_evaluation(0.0, config, files, model, cas, alphas_cumprod)
    
    print("Running Stochastic Iterative (eta=1.0)...")
    stoch_stats = run_evaluation(1.0, config, files, model, cas, alphas_cumprod)

    # Automatically generate and update the markdown file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    md_content = f"""# Phase 1 Sampler Comparison

| Metric | DDIM ($\eta = 0.0$) | Stochastic Iterative ($\eta = 1.0$) |
| :--- | :---: | :---: |
| **Acceptance Rate** | {ddim_stats['acc_rate']:.1f}% | {stoch_stats['acc_rate']:.1f}% |
| **Mean Steps** | {ddim_stats['avg_steps']:.1f} | {stoch_stats['avg_steps']:.1f} |
| **Final Energy** | {ddim_stats['avg_energy']:.5f} | {stoch_stats['avg_energy']:.5f} |
"""
    with open(args.output, "w") as f:
        f.write(md_content)
    
    print(f"\nSuccess. Results successfully written to {args.output}")

if __name__ == "__main__":
    main()