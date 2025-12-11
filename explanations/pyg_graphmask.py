"""
PyG GraphMask Explainer polyfill.
"""

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.explain import ExplainerAlgorithm
# from torch_geometric.explain.config import ModelTaskLevel

class GraphMaskExplainer(ExplainerAlgorithm):
    """
    The GraphMask explainer model from the "Interpreting Graph Neural Networks
    for NLP With Differentiable Edge Masking" paper.

    This implementation is a polyfill for PyG < 2.4 where GraphMaskExplainer
    is not yet available, adapted to work with the Explainer API.
    """
    def __init__(
        self,
        num_layers: int = 1,
        epochs: int = 100,
        lr: float = 0.01,
        penalty_scaling: float = 5.0,
        allowance: float = 0.03,
        log: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance
        self.log = log

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Tensor = None,
        **kwargs,
    ) -> dict:
        # GraphMask learns a mask per edge.
        # In the original paper/implementation, it learns a gate for each layer.
        # Here we assume a single layer (temporal wrapper) or treat all edges uniformly.

        edge_attr = kwargs.get('edge_attr')

        num_edges = edge_index.size(1)
        device = x.device

        # Initialize parameters
        # We use a single mask for the "layer" of the TGN wrapper
        mask = Parameter(torch.zeros(num_edges, device=device))

        optimizer = torch.optim.Adam([mask], lr=self.lr)

        model.eval()

        for _ in range(self.epochs):
            optimizer.zero_grad()

            # Sigmoid to get values in [0, 1]
            h = torch.sigmoid(mask)

            # Apply mask to messages (if edge_attr is present) or just pass it to model
            # The TemporalLinkWrapper expects edge_attr to be masked if we want to mask edges
            # But standard GNNs usually take edge_weight.
            # Our wrapper in graphmask_explainer.py handles edge_attr masking if passed.
            # However, the Explainer API calls model(x, edge_index, **kwargs).
            # We need to inject the masked edge_attr.

            masked_edge_attr = edge_attr * h.unsqueeze(-1) if edge_attr is not None else None

            # We need to pass the masked attributes.
            # The kwargs already contain 'edge_attr', we overwrite it.
            run_kwargs = kwargs.copy()
            if masked_edge_attr is not None:
                run_kwargs['edge_attr'] = masked_edge_attr

            # Forward pass
            logits = model(x, edge_index, **run_kwargs)

            # Loss calculation
            loss = self._loss(logits, target, h)
            loss.backward()
            optimizer.step()

        # Final mask
        final_mask = torch.sigmoid(mask).detach()
        return {'edge_mask': final_mask}

    def _loss(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # Prediction loss (Cross Entropy)
        pred_loss = F.cross_entropy(logits, target)

        # Regularization (Lagrange multiplier style in original, simplified here)
        # Original GraphMask uses Lagrange multipliers to enforce the allowance constraint.
        # Here we use a simpler penalty for sparsity and entropy as used in the custom
        # implementation to ensure stability and similarity to previous results, bu
        # adapted to the params.

        # Re-using the logic from the custom implementation for consistency with the
        # "penalty_scaling" concept but mapping it to sparsity/entropy.
        # Actually, let's stick to the custom implementation's loss function which we
        # know works for this data, but parameterized by this class.

        sparsity = mask.mean()
        entropy = -(mask * torch.log(mask + 1e-8) + (1 - mask) * torch.log(1 - mask + 1e-8)).mean()

        # We interpret penalty_scaling as the weight for these regularizers
        # In the custom impl, weights were 1e-3.
        # If penalty_scaling is 5.0 (default), we might need to scale it down.
        # Let's use the custom weights passed in init if we were fully custom,
        # but here we try to mimic PyG.
        # Let's use a simplified loss: CE + lambda * sparsity

        reg_loss = self.penalty_scaling * (sparsity + entropy) # Heuristic mapping

        return pred_loss + 1e-3 * reg_loss # Scaling down to match typical magnitude

    def supports(self) -> bool:
        return True
