from torch_geometric.data import Batch


def collate_fn(batch):
    """Collate a list of (HeteroData, solution_tensor) into a batch.

    Uses torch_geometric.data.Batch.from_data_list for variable-size graphs.

    Returns:
        (batched_data: Batch, solutions: list of tensors)

    Note: solutions is a list (not padded) because variable counts differ per problem.
    The training loop should handle the list case.
    """
    data_list, solutions = zip(*batch)
    batched_data = Batch.from_data_list(list(data_list))
    return batched_data, list(solutions)
