# Upstream KAIROS provenance

| | |
|---|---|
| Repository | https://github.com/ProvenanceAnalytics/kairos |
| Commit | `0e0b633beb46a1117c0a6d63be5d2481b59ac0dc` (`main` on 2026-10-09; committed 2023-08-23T23:50:12Z, "correct the database settings for theia e3") |
| Commit lookup | https://api.github.com/repos/ProvenanceAnalytics/kairos/commits/main |
| Paper | Cheng et al., "KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance", IEEE S&P 2024 |

## License

The repository has **no license file** at this commit. The repository root holds only `.gitignore`, `DARPA/`, `README.md`, `StreamSpot/` and `supplementary-material.pdf`. GitHub reports `license: null`, and `https://api.github.com/repos/ProvenanceAnalytics/kairos/license` returns 404.

Without a license, default copyright applies. The files are vendored unmodified, with attribution, for research reproducibility. Cite the paper (BibTeX in the upstream `README.md`), and ask the authors before redistributing beyond research use.

## Vendored files (`kairos/upstream/`, byte-exact)

Raw URL pattern: `https://raw.githubusercontent.com/ProvenanceAnalytics/kairos/0e0b633beb46a1117c0a6d63be5d2481b59ac0dc/<upstream path>`

Each file was checked against the upstream tree: `git hash-object <file>` equals the blob SHA-1 below.

| Local file | Upstream path | Bytes | git blob SHA-1 |
|---|---|---|---|
| `Makefile` | `DARPA/CADETS_E3/Makefile` | 520 | `ae219a0e4cc972c8e9ced7df551f597761bf52e3` |
| `anomalous_queue_construction.py` | `DARPA/CADETS_E3/anomalous_queue_construction.py` | 7366 | `7a8874ae6bdb0169ce79b0f62710f1e0487e14f0` |
| `attack_investigation.py` | `DARPA/CADETS_E3/attack_investigation.py` | 6253 | `2aa325e054a1dacf5b82759bd31f72bcda6eda23` |
| `config.py` | `DARPA/CADETS_E3/config.py` | 3075 | `7d53cf1061ebd3d78853b19e26add8906952ce35` |
| `create_database.py` | `DARPA/CADETS_E3/create_database.py` | 9688 | `08f4a75b7b6849e063a6e56fc9b6d1cb816a69d8` |
| `embedding.py` | `DARPA/CADETS_E3/embedding.py` | 4581 | `aaf920d834b2ea230c3db79ffe91de97c71ca725` |
| `evaluation.py` | `DARPA/CADETS_E3/evaluation.py` | 5671 | `86326613f01bb14adcc48857d07ba31367937851` |
| `kairos_utils.py` | `DARPA/CADETS_E3/kairos_utils.py` | 4375 | `b924fd35202cb6f86f959114acfd71b3f3e2c7e6` |
| `model.py` | `DARPA/CADETS_E3/model.py` | 2468 | `8bdfc3fcc5f60ea4b78b72f19caa35924ed90538` |
| `test.py` | `DARPA/CADETS_E3/test.py` | 7179 | `6298dff36fc959e2b4ab345d2211732bb2f0605e` |
| `train.py` | `DARPA/CADETS_E3/train.py` | 4569 | `0e172fa2ee1293ab2d531e5961a6ce48a6d71879` |
| `README_upstream.md` | `DARPA/README.md` ("Demo (DARPA CADETS E3)", the CADETS E3 setup/run section) | 2564 | `a41c329389c626b666c703872ea05dd6db8208dd` |
| `settings/database.md` | `DARPA/settings/database.md` (Postgres schema; see section "CADETS E3") | 13993 | `2ab2830bdd12bd7c7e5ab5d02c455848e51dbc83` |
| `settings/environment-settings.md` | `DARPA/settings/environment-settings.md` | 2245 | `66b60eebdbcbbb7f4ea5069183b9f4b82f520033` |
| `settings/requirements.txt` | `DARPA/settings/requirements.txt` | 1521 | `ae708549577d3cb085e0f8826917fbc53682783b` |

The relative links in `README_upstream.md` (`settings/...`) resolve inside `kairos/upstream/`. Its links to `CADETS_E3/...` point to the files next to it.

Not vendored, for reference only:

* `README.md` (repository root, blob `d28a39025f4b67c97ba731bb6f83417740d1da17`). It links the pretrained models on Google Drive, https://drive.google.com/drive/u/0/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C (`cadets3_models.pt`, Drive id `1x0ZSJnEn7kLfs3H-__LxtSE1fbd8vR0F`). It contains no raw-data link and defers to the paper and the supplementary material.
* `supplementary-material.pdf` (blob `a6744e97b16d134aa1219c4999757be1b600349b`).

## Derived code

* `kairos/patches/01..11-*.patch`: unified diffs, one per concern. Apply them in order to a copy of the files above.
* `kairos/cadets_e3/`: the result of applying all patches. It is the code to run.

See `kairos/README.md`.
