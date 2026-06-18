import sympy as sp
import json

def get_residuals(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract expressions from factors
    # This maps 'x' and 'y' symbols automatically
    return [sp.sympify(f['expression']) for f in data['factors']]