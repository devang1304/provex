import time
from typing import Dict

import torch
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer as GNNExplainerAlgo

from . import metrics, utils


GNN_EPOCHS = 200
GNN_LR = 0.01
MASK_THRESHOLD = 0.5


def explain_event(
    context: utils.EventContext,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Runs GNNExplainer on a single event and returns metric dict."""
    wrapper = utils.TemporalLinkWrapper(gnn, link_pred, context, device)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainerAlgo(epochs=GNN_EPOCHS, lr=GNN_LR),
        explanation_type="phenomenon",
        edge_mask_type="object",
        node_mask_type=None,
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="edge",
            return_type="raw",
        ),
    )

    start = time.perf_counter()
    explanation = explainer(
        context.memory_inputs.to(device),
        context.edge_index.to(device),
        edge_attr=context.edge_messages.to(device),
        edge_t=context.edge_times.to(device),
        target=torch.tensor([max(context.label, 0)], device=device),
    )
    runtime = time.perf_counter() - start

    edge_mask = explanation.edge_mask.detach()
    return metrics.evaluate_mask(wrapper, context, edge_mask, runtime=runtime, threshold=MASK_THRESHOLD)
