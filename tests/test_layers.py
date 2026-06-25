import torch
from marc.model.layers import BipartiteLayer


def test_bipartite_layer_shapes():
    D = 32
    layer = BipartiteLayer(D)
    n_vars, n_facs, n_edges = 2, 2, 4
    h_v = torch.randn(n_vars, D)
    h_f = torch.randn(n_facs, D)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)
    edge_feat = torch.tensor([[1.0], [1.0], [-1.0], [1.0]])
    h_v_new, h_f_new = layer(h_v, h_f, edge_index, edge_feat)
    assert h_v_new.shape == (n_vars, D)
    assert h_f_new.shape == (n_facs, D)


def test_bipartite_layer_gradient_flows():
    D = 16
    layer = BipartiteLayer(D)
    h_v = torch.randn(2, D, requires_grad=True)
    h_f = torch.randn(2, D)
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    edge_feat = torch.ones(2, 1)
    h_v_new, h_f_new = layer(h_v, h_f, edge_index, edge_feat)
    h_v_new.sum().backward()
    assert h_v.grad is not None
