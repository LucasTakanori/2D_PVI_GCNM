from types import SimpleNamespace

import torch

from gcnm_pvi.direction_anchored_model import DirectionAnchoredCoreStage


def _graph() -> SimpleNamespace:
    nodes = 5
    x = torch.zeros(nodes, 6)
    x[:, 0] = torch.linspace(0.1, 0.5, nodes)
    x[:, 1] = torch.linspace(-0.2, 0.2, nodes)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    return SimpleNamespace(
        x=x,
        edge_index=edge_index,
        batch=torch.zeros(nodes, dtype=torch.long),
    )


def test_stage1_starts_at_lm_direction() -> None:
    graph = _graph()
    model = DirectionAnchoredCoreStage(
        accumulate_current=False,
        correction_limit=2.0,
    )
    prediction = model(graph)
    torch.testing.assert_close(prediction[:, 0], graph.x[:, 1])


def test_stage2_starts_at_current_plus_lm_direction() -> None:
    graph = _graph()
    model = DirectionAnchoredCoreStage(
        accumulate_current=True,
        correction_limit=0.5,
    )
    prediction = model(graph)
    torch.testing.assert_close(
        prediction[:, 0],
        graph.x[:, 0] + graph.x[:, 1],
    )
