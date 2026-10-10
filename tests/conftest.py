import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import TemporalData

from provex.model import KairosHParams, build


@pytest.fixture(scope="session")
def tiny():
    """30-node KAIROS model in float64 + 800 events with tied timestamps."""
    torch.manual_seed(0)
    hp = KairosHParams(num_nodes=30, raw_msg_dim=15, memory_dim=8, time_dim=8, embedding_dim=8,
                       neighbor_size=5, num_edge_types=7, node_embedding_dim=4)
    memory, gnn, lp = build(hp)
    memory.eval(); gnn.eval(); lp.eval()          # eval first: TGNMemory.train(False) flushes its store
    memory, gnn, lp = memory.double(), gnn.double(), lp.double()
    memory.reset_state()                          # re-create the message store in float64
    for m in (memory, gnn, lp):
        for p in m.parameters():
            p.requires_grad_(False)
    n = 800
    g = torch.Generator().manual_seed(1)
    src = torch.randint(0, 30, (n,), generator=g)
    dst = torch.randint(0, 30, (n,), generator=g)
    inc = torch.randint(0, 3, (n,), generator=g) * 1_000_000_000      # many exact ties
    t = 1_523_000_000_000_000_000 + torch.cumsum(inc, 0)
    y = torch.randint(0, 7, (n,), generator=g)
    feats = torch.rand(n, 8, generator=g, dtype=torch.float64)
    msg = torch.cat([feats[:, :4], F.one_hot(y, 7).double(), feats[:, 4:]], -1)
    data = TemporalData(src=src, dst=dst, t=t, msg=msg)
    return hp, memory, gnn, lp, data


BATCH = 32
WINDOW_NS = 60 * 1_000_000_000