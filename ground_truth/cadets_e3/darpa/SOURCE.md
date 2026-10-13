# DARPA indicators for the CADETS E3 attacks

`iocs.json` lists, per attack, the IP addresses, files, processes and times from DARPA's official Engagement 3 ground-truth report (`TC_Ground_Truth_Report_E3_Update.pdf`, Kudu Dynamics / TA5.1). The report is linked from <https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md>.

It complements the node-level ORTHRUS labels in `../orthrus/`. Those labels omit some attack infrastructure that the report names, for example the 04-06 shellcode server 78.205.235.65, the libdrakon server and `/var/log/devc`. They also include processes that KAIROS merges by name with benign activity.

- All IP addresses in the report are fictional (README-E3).
- The report's address table for 04-06 writes `154.143.113.18`, while its event log and connection list write `154.145.113.18`. We keep the latter.