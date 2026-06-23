# Cached Results and Large Data

Large raw datasets, OpenML caches, and generated result bundles are excluded
from this GitHub release. They will be hosted separately on Google Drive:

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
