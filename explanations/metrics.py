from typing import Dict

import torch

from .utils import EventContext, TemporalLinkWrapper


def _run_model(
    wrapper: TemporalLinkWrapper,
    context: EventContext,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_times: torch.Tensor,
) -> torch.Tensor:
    device = wrapper.last_update.device
    logits = wrapper(
        context.memory_inputs.to(device),
        edge_index,
        edge_attr=edge_attr,
        edge_t=edge_times,
    )
    return torch.softmax(logits, dim=-1)


def evaluate_mask(
    wrapper: TemporalLinkWrapper,
    context: EventContext,
    edge_mask: torch.Tensor,
    *,
    threshold: float,
    runtime: float,
) -> Dict[str, float]:
    device = context.edge_messages.device
    edge_mask = edge_mask.to(device)
    keep = edge_mask > threshold

    probs_full = _run_model(
        wrapper, context, context.edge_index, context.edge_messages, context.edge_times)

    keep_index = context.edge_index[:, keep]
    keep_attr = context.edge_messages[keep]
    keep_times = context.edge_times[keep]
    probs_keep = _run_model(
        wrapper, context, keep_index, keep_attr, keep_times)

    drop = ~keep
    drop_index = context.edge_index[:, drop]
    drop_attr = context.edge_messages[drop]
    drop_times = context.edge_times[drop]
    probs_drop = _run_model(
        wrapper, context, drop_index, drop_attr, drop_times)

    label = max(context.label, 0)
    prob_full = probs_full[0, label].item()
    prob_keep = probs_keep[0, label].item()
    prob_drop = probs_drop[0, label].item()

    m = edge_mask.clamp(1e-6, 1 - 1e-6)
    mask_entropy = (-(m * torch.log(m) + (1 - m) *
                    torch.log(1 - m))).mean().item()

    kept_edges = float(keep.sum().item())
    total_edges = max(context.edge_index.size(1), 1)

    return {
        "prob_full": prob_full,
        "prob_keep": prob_keep,
        "prob_drop": prob_drop,
        "comprehensiveness": prob_full - prob_drop,
        "sufficiency": prob_full - prob_keep,
        "sparsity": 1.0 - kept_edges / total_edges,
        "entropy": mask_entropy,
        "runtime_sec": runtime,
        "kept_edges": kept_edges,
        "edge_mask": edge_mask.detach().cpu().tolist(),
    }
