# Diffusion & Refinement

- `diffusion/`: Contains forward corruption logic for GNN training.
- `refine/`: Contains inference-time noise injection for iterative solving.
- Operations are performed on [n, d] tensors extracted from the constraint graph.
