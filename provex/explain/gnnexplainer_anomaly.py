"""PyG GNNExplainer baselines on the neighbour-edge channel only.

* ``AnomalyGNNExplainer``: PyG's GNNExplainer with the anomaly-preserving objective. It keeps
  PyG's own mask mechanics (sigmoid edge mask applied after the attention softmax), so it
  is the "standard tool, corrected objective" baseline.
* ``v1_gnnexplainer``: the exact v1 configuration (phenomenon, target = true type, plain CE).
  Kept as the documented counter-example: it explains why the event looks normal.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig

from . import objectives
from .event_model import EventModel
from .joint_mask import Explanation


class _EdgeChannelModel(nn.Module):
    """forward(x, edge_index, msg=..., t_off=...) -> logits[1, C] for the captured event."""

    def __init__(self, em: EventModel):
        super().__init__()
        self.em = em

    def forward(self, x, edge_index, msg=None, t_off=None):
        em = self.em
        edge_attr = em._edge_attr(em.lu_off, edge_index, t_off, msg)
        z = torch.relu(em.gnn.conv(x, edge_index, edge_attr))
        z = torch.relu(em.gnn.conv2(z, edge_index, edge_attr))
        return em.link_pred(z[[em.cap.src_local]], z[[em.cap.dst_local]])


class AnomalyGNNExplainer(GNNExplainer):
    def __init__(self, tau: float, L_full: float, margin: float = 0.5, T: float = 1.0, **kw):
        super().__init__(**kw)
        self.tau, self.L_full, self.margin, self.T = tau, L_full, margin, T

    def _calculate_base_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        L = F.cross_entropy(y_hat, y)
        return objectives.penalty("preserve", L, self.L_full, self.tau, self.margin, self.T)


def _run(em: EventModel, algorithm, tau: float, name: str, seed: int) -> Explanation:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    # eval() matters: PyG's Explainer restores the wrapper's training flag when it returns, and a new
    # module starts in training mode, which would switch on KAIROS's link-predictor dropout for every
    # later forward pass.
    model = _EdgeChannelModel(em).eval()
    explainer = Explainer(model=model, algorithm=algorithm, explanation_type="phenomenon",
                          edge_mask_type="object", node_mask_type=None,
                          model_config=ModelConfig(mode="multiclass_classification",
                                                   task_level="edge", return_type="raw"))
    expl = explainer(em.x, em.ei, target=em.y, msg=em.edge_msg, t_off=em.edge_t_off)
    em.gnn.eval(); em.link_pred.eval()
    edge_mask = expl.edge_mask.detach().to(torch.float64)
    scores = torch.full((em.num_items,), -1.0, dtype=torch.float64, device=em.device)
    scores.scatter_reduce_(0, em.edge_item, edge_mask, reduce="amax", include_self=True)
    with torch.no_grad():
        L_full = float(em.loss(torch.ones(em.num_items, dtype=em.dtype, device=em.device), hard=True))
        keep = (scores > 0.5) | ~em.is_edge_item
        L_sel = float(em.loss(keep.to(em.dtype), hard=True))
    return Explanation(method=name, mode="preserve", items=em.items.tolist(), scores=scores.tolist(),
                       selected=em.items[(scores > 0.5).cpu()].tolist(), valid=L_sel >= tau, repaired=0,
                       L_full=L_full, L_selected=L_sel, tau=tau, runtime_s=time.perf_counter() - t0)


def anomaly_gnnexplainer(em: EventModel, tau: float, epochs: int = 200, lr: float = 0.01, seed: int = 0):
    with torch.no_grad():
        L_full = float(em.loss(torch.ones(em.num_items, dtype=em.dtype, device=em.device), hard=True))
    return _run(em, AnomalyGNNExplainer(tau, L_full, epochs=epochs, lr=lr), tau, "gnnexplainer-anomaly", seed)


def v1_gnnexplainer(em: EventModel, tau: float, epochs: int = 200, lr: float = 0.01, seed: int = 0):
    return _run(em, GNNExplainer(epochs=epochs, lr=lr), tau, "gnnexplainer-v1", seed)