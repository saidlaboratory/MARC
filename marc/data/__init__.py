"""Data generation utilities for MARC problem templates."""
from .dataset import MARCDataset
from .collate import collate_fn

__all__ = ["MARCDataset", "collate_fn"]
