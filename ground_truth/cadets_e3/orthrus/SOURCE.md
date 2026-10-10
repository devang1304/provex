# CADETS E3 node-level ground truth (Orthrus / PIDSMaker)

The files in this directory are byte-exact copies from PIDSMaker.

| | |
|---|---|
| Repository | https://github.com/ubc-provenance/PIDSMaker |
| Commit | `438a05a7c55256f639e5b5945cf4828fe3ee078e` (`main` on 2026-10-09; committed 2026-10-02, "Merge pull request #61 from ubc-provenance/dev") |
| Path | `Ground_Truth/orthrus/E3-CADETS/` (CSVs) and `Ground_Truth/orthrus/readme.md` |
| License | Apache License 2.0, copied as `LICENSE` from the repository root |

## Files

Raw URLs: `https://raw.githubusercontent.com/ubc-provenance/PIDSMaker/438a05a7c55256f639e5b5945cf4828fe3ee078e/<path>`

| File | Upstream path | Rows | Bytes | git blob SHA-1 |
|---|---|---|---|---|
| `node_Nginx_Backdoor_06.csv` | `Ground_Truth/orthrus/E3-CADETS/node_Nginx_Backdoor_06.csv` | 8 | 688 | `cec40520e1c9e7b306841e60f15a56c2a5748554` |
| `node_Nginx_Backdoor_11.csv` | `Ground_Truth/orthrus/E3-CADETS/node_Nginx_Backdoor_11.csv` | 4 | 357 | `c000c90ca9584691e36047dd592c37c91aae3e15` |
| `node_Nginx_Backdoor_12.csv` | `Ground_Truth/orthrus/E3-CADETS/node_Nginx_Backdoor_12.csv` | 43 | 3225 | `55cf7adde693c4098045ffea97ca61adc656d1df` |
| `node_Nginx_Backdoor_13.csv` | `Ground_Truth/orthrus/E3-CADETS/node_Nginx_Backdoor_13.csv` | 24 | 1816 | `dcc5fc144824cf2a5f428891965376ed54f15eba` |
| `readme.md` | `Ground_Truth/orthrus/readme.md` | | 5343 | `e031ce90d8d45f9e9e058839c44d62ee42fa347b` |
| `LICENSE` | `LICENSE` | | 11357 | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |

## Format

One malicious node per line, no header:

```
UUID,{attr dict},index_id
674D8313-390A-11E8-BF66-D9AA8AFF4A69,{'subject': 'None nginx'},211776
```

* `UUID`: the CDM18 UUID of the subject / FileObject / NetFlowObject in the raw DARPA TC data.
* `{attr dict}`: a Python dict literal with one key, the node type, and its label:
  * `subject`: `'<path> <cmdLine>'`; in CADETS E3 the path is always `None`, so `'None nginx'` is the process `nginx`.
  * `file`: the file path, or `'None'` when PIDSMaker had no path.
  * `netflow`: `'<local ip>:<local port>-><remote ip>:<remote port>'`.
* `index_id`: **PIDSMaker's own database id, not a KAIROS `node2id.index_id`.** Do not use it to index KAIROS nodes. `kairos/cadets_e3/node_gt.py` maps the rows to KAIROS ids (by name by default, by UUID through the KAIROS database optionally).

KAIROS keys nodes by name (subject exec, file path, netflow `srcaddr,srcport,dstaddr,dstport`, and labels a netflow only with the remote `ip:port`). Several UUIDs therefore collapse onto one KAIROS node. For example, every `sshd` process is the single node `subject: sshd`.

## PIDSMaker settings for CADETS_E3

Source: `pidsmaker/config/config.py`, `DATASET_DEFAULT_CONFIG["CADETS_E3"]`, same commit.

```python
"timezone": "US/Eastern",
"start_date": "2018-04-02",
"end_date": "2018-04-14",
"train_dates": ["2018-04-02", "2018-04-03", "2018-04-04", "2018-04-05",
                "2018-04-07", "2018-04-08", "2018-04-09"],
"val_dates": ["2018-04-10"],
"test_dates": ["2018-04-06", "2018-04-11", "2018-04-12", "2018-04-13"],
"ground_truth_relative_path": [
    # "E3-CADETS/node_E_mail_Server.csv",
    "E3-CADETS/node_Nginx_Backdoor_06.csv",
    # "E3-CADETS/node_Nginx_Backdoor_11.csv",
    "E3-CADETS/node_Nginx_Backdoor_12.csv",
    "E3-CADETS/node_Nginx_Backdoor_13.csv",
],
"attack_to_time_window": [
    ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
    # ["E3-CADETS/node_Nginx_Backdoor_11.csv" , '2018-04-11 15:07:00', '2018-04-11 15:16:00'],
    ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
    ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
],
```

The times are US/Eastern wall-clock time, the convention KAIROS uses for its time-window file names (`ns_time_to_datetime_US`). The same four intervals, including 04-11, are in `kairos/cadets_e3/config.py` (`cadets_e3_attacks`).

## Notes

* PIDSMaker does **not** evaluate `Nginx_Backdoor_11`. Its entries are commented out above, and `readme.md` lists it as "partial succeed / removed": the attacker ran drakon in memory, but the download of the libdrakon implant and the sshd injection failed. We keep it as a fourth attack and report it separately. A pooled metric with and without 04-11 can be recomputed from the per-attack and per-day results in `detection_metrics.json`.
* The "Date" column of the Cadets E3 table in `readme.md` reads `0406` for every attack. That is an upstream typo; the attacks are on 04-06, 04-11, 04-12 and 04-13.
* The UUIDs come from PIDSMaker's own parse of the raw data. Some of them, such as files without a path, have no counterpart in the KAIROS database, because KAIROS only stores files with a `predicateObjectPath` in an event. `node_gt.py` reports these as unmapped.
