"""End-to-end CLI test: checkpoint (upstream format) -> pass1 -> alerts -> explain -> tables."""
import json
import subprocess
import sys

import torch

from provex import pipeline


def test_pipeline_end_to_end(tiny, tmp_path):
    hp, memory, gnn, lp, data = tiny
    # upstream-format checkpoint (kairos/patches/04) in float32, data in float32
    ck = tmp_path / "models_state.pt"
    torch.save({"memory": {k: v.float() if v.is_floating_point() else v for k, v in memory.state_dict().items()},
                "gnn": {k: v.float() for k, v in gnn.state_dict().items()},
                "link_pred": {k: v.float() for k, v in lp.state_dict().items()},
                "hyperparams": {"memory_dim": hp.memory_dim, "time_dim": hp.time_dim, "embedding_dim": hp.embedding_dim,
                                "neighbor_size": hp.neighbor_size, "raw_msg_dim": hp.raw_msg_dim,
                                "max_node_num": hp.num_nodes, "include_edge_type": list(range(hp.num_edge_types)),
                                "node_embedding_dim": hp.node_embedding_dim}}, ck)
    d = data.clone(); d.msg = d.msg.float()
    day = tmp_path / "graph_4_6.TemporalData.simple"
    torch.save(d, day)
    names = tmp_path / "names.json"
    names.write_text(json.dumps({str(i): f"subject: proc{i}" for i in range(hp.num_nodes)}))
    gt = tmp_path / "gt"; gt.mkdir()
    (gt / "kairos_nodes_06.json").write_text(json.dumps({"3": "subject: proc3", "7": "subject: proc7"}))
    out = tmp_path / "out"
    pipeline.main(["pass1", "--ckpt", str(ck), "--day-file", str(day), "--day", "4_6", "--out", str(out / "pass1"),
                   "--batch-size", "32", "--window-ns", str(60 * 10**9)])
    wins = json.loads((out / "pass1" / "windows_4_6.json").read_text())
    p1 = torch.load(out / "pass1" / "pass1_4_6.pt", weights_only=False)
    assert p1["losses"].numel() == data.num_events
    # the CLI pass1 uses 15-min windows; the fixture spans ~27 min, so build alerts from what exists
    alerts = [{"day": "4_6", "window_file": f"w{i}.txt", "start_ns": w["start_ns"], "end_ns": w["end_ns"],
               "queue_id": 0, "queue_score": 1e3, "is_gt_attack_window": True} for i, w in enumerate(wins)]
    assert len(alerts) >= 3

    (tmp_path / "alerts.json").write_text(json.dumps(alerts))
    pipeline.main(["explain", "--ckpt", str(ck), "--day-file", str(day), "--day", "4_6",
                   "--alerts", str(tmp_path / "alerts.json"), "--names", str(names), "--gt-dir", str(gt),
                   "--pass1", str(out / "pass1" / "pass1_4_6.pt"), "--out", str(out / "expl" / "seed_0"),
                   "--methods", "joint,joint-cf,variational,gnnx-anomaly,baselines", "--top-n", "2", "--sample-n", "1",
                   "--history-batches", "4", "--full-scores", "--calibration", "--calibration-n", "4",
                   "--shard", "0/1"])
    files = list((out / "expl" / "seed_0").glob("explanations_4_6*.jsonl"))
    assert files
    recs = [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]
    assert recs and all(r["alert_class"] in ("prior", "long_memory", "window") for r in recs)
    for r in recs:
        for ex in r["explanations"]:
            for it in ex["top_items"]:
                assert 0 <= it["ev"] < data.num_events
                assert it["src"] == int(data.src[it["ev"]]) and it["dst"] == int(data.dst[it["ev"]])
    res = tmp_path / "results"
    (res / "expl").mkdir(parents=True)
    (out / "expl" / "seed_0").rename(res / "expl" / "seed_0")
    subprocess.run([sys.executable, "scripts/make_tables.py", "--results", str(res), "--out", str(tmp_path / "gen")],
                   check=True)
    assert (tmp_path / "gen" / "numbers.tex").exists() and (tmp_path / "gen" / "tab_faithfulness.tex").exists()