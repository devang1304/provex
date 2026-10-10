"""Diagnose capture-vs-batch logit parity on real data (runs on a VM).

At the target event's batch, compare the full-batch forward with the subgraph forward:
(1) time-encoding inputs/outputs on shared edges, (2) conv1 outputs on N1 nodes,
(3) z for u/v, (4) logits; in float32 and in float64.
"""
import argparse, sys, types
import torch
sys.path.insert(0, ".")
if "psycopg2" not in sys.modules:
    pg = types.ModuleType("psycopg2"); pg.extras = types.ModuleType("psycopg2.extras")
    sys.modules["psycopg2"], sys.modules["psycopg2.extras"] = pg, pg.extras
from provex.data import load_day
from provex.model import load_legacy_checkpoint
from provex.replay.capture import extract_subgraph
from provex.replay.kairos_replay import KairosReplayer


def run(dtype, a, data, ev):
    import kairos_tgnn_model as legacy
    memory, gnn, lp, hp = load_legacy_checkpoint(a.ckpt, legacy, "cpu", rel_t_sign=-1)
    if dtype == torch.float64:
        memory, gnn, lp = memory.double(), gnn.double(), lp.double(); memory.reset_state()
        data = data.clone(); data.msg = data.msg.double()
    rep = KairosReplayer(memory, gnn, lp, hp)
    out = {}

    def hook(st):
        if not (st.begin <= ev < st.end):
            return
        pos = ev - st.begin
        a_u, a_v = int(st.assoc[st.src[pos]]), int(st.assoc[st.dst[pos]])
        ei = st.edge_index
        et, em = data.t[st.e_id], data.msg[st.e_id]
        ea_full = gnn.edge_attr(st.lu, ei, et, em, st.x.dtype)
        z1_full = torch.relu(gnn.conv(st.x, ei, ea_full))
        z_full = torch.relu(gnn.conv2(z1_full, ei, ea_full))
        lg_full = lp(z_full[[a_u]], z_full[[a_v]])[0]
        nodes, keep, _ = extract_subgraph(ei, a_u, a_v)
        relabel = torch.full((st.n_id.numel(),), -1, dtype=torch.long); relabel[nodes] = torch.arange(nodes.numel())
        ei_s = relabel[ei[:, keep]]
        ea_sub = gnn.edge_attr(st.lu[nodes], ei_s, et[keep], em[keep], st.x.dtype)
        z1_sub = torch.relu(gnn.conv(st.x[nodes], ei_s, ea_sub))
        z_sub = torch.relu(gnn.conv2(z1_sub, ei_s, ea_sub))
        lg_sub = lp(z_sub[[int(relabel[a_u])]], z_sub[[int(relabel[a_v])]])[0]
        rel_full = (st.lu[ei[0]] - et)[keep]
        rel_sub = st.lu[nodes][ei_s[0]] - et[keep]
        out.update(
            n_batch_nodes=int(st.n_id.numel()), n_batch_edges=int(ei.size(1)), n_sub_nodes=int(nodes.numel()),
            n_sub_edges=int(keep.sum()), rel_t_equal=bool(torch.equal(rel_full, rel_sub)),
            rel_t_absmax=float(rel_full.abs().max()),
            edge_attr_maxdiff=float((ea_full[keep] - ea_sub).abs().max()),
            z1_maxdiff_on_sub_nodes=float((z1_full[nodes] - z1_sub).abs().max()),
            z_u_diff=float((z_full[a_u] - z_sub[int(relabel[a_u])]).abs().max()),
            logits_diff=float((lg_full - lg_sub).abs().max()),
            logits_batch_vs_full=float((st.logits[pos] - lg_full).abs().max()))
        raise StopIteration

    try:
        rep._replay(data, pre_update=hook)
    except StopIteration:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--day-file", required=True)
    ap.add_argument("--event", type=int, default=781576)
    a = ap.parse_args()
    data = load_day(a.day_file)
    for dt in (torch.float32, torch.float64):
        print(dt, run(dt, a, data, a.event), flush=True)


if __name__ == "__main__":
    main()