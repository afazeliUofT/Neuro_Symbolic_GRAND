import torch

from hybrid_bp_nsg.models.rescue_net import RescueNet


def test_rescue_net_shapes():
    model = RescueNet(num_var_features=12, num_check_features=5, num_global_features=8,
                      n=32, m=16, num_segments=8, max_weight_class=8,
                      hidden_dim=32, graph_layers=2, top_k_tokens=16,
                      transformer_heads=4, transformer_layers=1, candidate_feature_dim=8)
    batch = 2
    out = model(
        var_features=torch.randn(batch, 32, 12),
        check_features=torch.randn(batch, 16, 5),
        global_features=torch.randn(batch, 8),
        heuristic_order=torch.stack([torch.arange(32), torch.arange(31, -1, -1)]),
        h_dense=torch.randint(0, 2, (16, 32), dtype=torch.float32),
        deg_v=torch.ones(32),
        deg_c=torch.ones(16),
        candidate_features=torch.randn(batch, 6, 8),
    )
    assert out['bit_logits'].shape == (batch, 32)
    assert out['weight_logits'].shape[0] == batch
    assert out['candidate_scores'].shape == (batch, 6)
