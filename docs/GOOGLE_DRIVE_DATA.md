# Cached Results and Large Data

Large raw datasets, OpenML caches, full-grid diagnostics, generated result
bundles, large figures, bootstrap outputs, DGP search logs, random-admissible
diagnostics, gated-random diagnostics, and acquisition allocation artifacts are
excluded from this lightweight GitHub release. They are hosted separately on
Google Drive:

[Google Drive data and cached results](https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing)

Recommended layout after download:

```text
data/raw/          # raw real-data benchmark files
data/cache/        # optional local dataset caches
outputs/results/   # cached experiment CSV/JSON outputs
outputs/figures/   # regenerated or cached figures
outputs/tables/    # regenerated or cached tables
```

The synthetic experiments do not require external data. Experiment 8 can use
standard numeric regression datasets if available locally, and otherwise falls
back to built-in or downloadable sources where supported by scikit-learn/OpenML
configuration.

The Drive archive covers released experiment outputs promised by the manuscript
and supplement, including DGP search logs, full-grid sensitivity results,
benchmark workflow summaries, random-admissible and gated-random diagnostics,
two-regime acquisition trajectories, allocation maps, bootstrap summaries, and
gamma-sensitivity endpoint metrics.
