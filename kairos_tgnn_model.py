from kairos_utils import *
from config import *

torch.set_float32_matmul_precision("high")
torch.set_num_threads(2)

device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))

# Announce device at import time for visibility during training
try:
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"[Device] Using CUDA: {gpu_name} | {total_mem_gb:.1f} GB")
    elif device.type == "mps":
        print("[Device] Using Apple MPS backend")
    else:
        print("[Device] Using CPU")
except Exception as _e:
    print(f"[Device] Detected {device}, details unavailable: {_e}")

# remove or comment this line
# torch.mps.set_enabled(True)

criterion = nn.CrossEntropyLoss()

# Default capacity derived from historical CADETS node2id table.
max_node_num = 268243
min_dst_idx, max_dst_idx = 0, max_node_num
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


def configure_node_capacity(requested_nodes: int) -> None:
    """Ensure global buffers can index all nodes."""
    global max_node_num, assoc, max_dst_idx
    requested = int(requested_nodes)
    if requested <= 0:
        raise ValueError("requested_nodes must be positive")
    if requested > max_node_num:
        max_node_num = requested
        assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
        max_dst_idx = max_node_num


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels * 8, out_channels, heads=1, concat=False,
                                     dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update = last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = t - last_update[edge_index[0]]
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(

            Linear(in_channels * 4, in_channels * 8),
            torch.nn.Dropout(0.2),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.Dropout(0.2),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.Dropout(0.2),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels)
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)
        h = self.lin_seq(h)
        return h


class MPSSafeLastAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        from torch_scatter import scatter_max

        if msg.device.type == "mps":
            msg_cpu = msg.to("cpu")
            index_cpu = index.to("cpu")
            t_cpu = t.to("cpu")
            _, argmax = scatter_max(
                t_cpu, index_cpu, dim=0, dim_size=int(dim_size))
            out_cpu = msg_cpu.new_zeros((int(dim_size), msg_cpu.size(-1)))
            mask = argmax < msg_cpu.size(0)
            out_cpu[mask] = msg_cpu[argmax[mask]]
            return out_cpu.to(msg.device)

        _, argmax = scatter_max(t, index, dim=0, dim_size=int(dim_size))
        out = msg.new_zeros((int(dim_size), msg.size(-1)))
        mask = argmax < msg.size(0)
        out[mask] = msg[argmax[mask]]
        return out


def cal_pos_edges_loss_multiclass(link_pred_ratio, labels):
    losses = []
    for i in range(len(link_pred_ratio)):
        losses.append(criterion(
            link_pred_ratio[i].reshape(1, -1),
            labels[i].reshape(-1)))
    return torch.stack(losses)
