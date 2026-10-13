"""PROVEX v2: anomaly-faithful explanations for the KAIROS provenance IDS.

Package layout:
    provex.model      -- upstream-compatible KAIROS model and checkpoint loading
    provex.data       -- day graphs, edge-type labels, node names
    provex.replay     -- KAIROS-exact replay (pass 1 scores, pass 2 captures), alerts
    provex.explain    -- interventions, functional TGN replay, objectives, explainers
    provex.evaluate   -- anomaly-term metrics, ground truth, parity checks
"""

__version__ = "2.0.0.dev0"
