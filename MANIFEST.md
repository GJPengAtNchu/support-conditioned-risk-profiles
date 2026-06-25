# Release Manifest

This manifest describes the lightweight GitHub release for:

**Support-Conditioned Risk Profiles for Model Evaluation under Heterogeneous Data Coverage**

Large artifacts are stored in the linked Google Drive folder:

[Google Drive data and cached results](https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing)

## Included In GitHub

- `main.pdf`: finalized main manuscript.
- `supplement.pdf`: finalized supplementary material.
- `README.md`: release overview and Google Drive pointer.
- `CITATION.cff`: citation metadata.
- `LICENSE`: software license.
- `environment.yml`, `requirements.txt`, `pyproject.toml`: lightweight environment metadata.
- `experiments/`: lightweight experiment scripts for synthetic, acquisition, and benchmark diagnostics.
- `scripts/`: lightweight reproduction, plotting, post-processing, and smoke-test utilities.
- `src/`: small shared support-profile utilities.
- `data/README.md`: instructions for placing raw/cached data after download.
- `results/README.md`: instructions for generated and cached result files.
- `figures/README.md`: instructions for generated and cached figures.
- `docs/GOOGLE_DRIVE_DATA.md`: Google Drive layout notes.
- `docs/REPRODUCIBILITY.md`: reproduction notes.

No LaTeX source files are included in `release/`.

## Stored In Google Drive

The following large or exhaustive artifacts are intentionally excluded from
GitHub and documented as Google Drive artifacts:

- DGP search logs:
  - Section 4.2 synthetic model-selection DGP search log.
  - KRR mechanism DGP search log.
- Model-selection outputs:
  - raw selected-rule metrics;
  - selected-frequency diagnostics;
  - tau sensitivity;
  - score-weight sensitivity;
  - full support-construction sensitivity grids.
- Benchmark workflow outputs:
  - dataset/model-family summaries;
  - split-level and candidate-level cached summaries;
  - random-admissible summaries;
  - gated-random summaries;
  - gate sensitivity;
  - full support-construction sensitivity grids.
- Acquisition outputs:
  - full two-regime trajectories;
  - full allocation maps;
  - bin-round heatmaps;
  - compact and full bootstrap summaries;
  - gamma-sensitivity endpoint metrics;
  - full controlled three-region conflict outputs.
- Raw caches and large generated artifacts:
  - raw datasets and OpenML/local caches;
  - large generated CSV/JSON bundles;
  - full-grid figures and large supplementary figures;
  - temporary full-run outputs and intermediate caches.

## Excluded From GitHub

- LaTeX source and build files (`*.tex`, `*.bib`, `*.aux`, `*.log`, and related build artifacts).
- Large raw datasets and raw caches.
- Model checkpoints or trained model dumps.
- Local machine paths, credentials, authentication material, and temporary files.
