# Stored results

The existing `experiments/` and `optimization/` directories contain historical in-house GNN checkpoints, configurations, HPO records, and summaries. They are retained as archived outputs, not guaranteed to be regenerable with today's dependency versions.

`external_baselines/` contains compact metric outputs copied from the development workspace for the local descriptor, VAE-fingerprint, RGCN, and Atom-Attention-MPNN runs. They are single-run artifacts and do not replace the paper's three-seed summary values. Atom-Attention-MPNN test score CSVs are retained; bulky ensemble weights, TensorBoard event logs, caches, and intermediate predictions are excluded.

The manuscript tables in [`../paper/main.tex`](../paper/main.tex) remain the source for the reported paper comparison. See [`../docs/reproducibility.md`](../docs/reproducibility.md) for protocol and provenance differences.
