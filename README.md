# Support-Conditioned Risk Profiles

Lightweight GitHub release for the paper:

**Support-Conditioned Risk Profiles for Model Evaluation under Heterogeneous Data Coverage**

This lightweight GitHub release contains the compiled manuscript PDFs, lightweight
code/configuration files, and compact result summaries. Large cached outputs, raw
experiment artifacts, full-grid sensitivity results, and large figures are stored
in the linked Google Drive folder.

**LaTeX source files are not included in this release folder.**

## Contents

```text
main.pdf              Finalized main manuscript.
supplement.pdf        Finalized supplementary material.
experiments/          Lightweight experiment scripts.
scripts/              Utility and post-processing scripts.
src/                  Small shared support-profile utilities.
configs/              Configuration files, if present.
data/README.md        Data and cache placement notes.
results/README.md     Notes for generated and cached result files.
figures/README.md     Notes for generated and cached figures.
docs/                 Reproducibility and Google Drive layout notes.
```

Large cached outputs and full-grid diagnostics are available here:

[Google Drive data and cached results](https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing)

## What Is Stored In Google Drive

The Google Drive archive is the intended location for large or exhaustive
artifacts that should not be committed to GitHub, including:

- DGP search logs for the synthetic model-selection and KRR mechanism studies;
- full-grid support-construction and binning sensitivity tables;
- raw selected-rule metrics and full candidate/cached experiment outputs;
- benchmark workflow dataset/model-family summaries and cached split-level files;
- random-admissible, gated-random, and gate-sensitivity diagnostics;
- full two-regime acquisition trajectories and allocation maps;
- bin-round heatmaps and large generated figures;
- compact and full bootstrap summaries;
- gamma-sensitivity endpoint metrics and full controlled three-region conflict outputs;
- raw caches and large intermediate CSV/JSON bundles.

## Inspecting Results

Use the compact files in this repository to inspect the release structure and run
lightweight checks. Use the Google Drive archive for full cached outputs and
large figures referenced by the manuscript and supplement.

Useful entry points:

```bash
python scripts/smoke_test.py
python scripts/download_data.py
python scripts/reproduce_main_figures.py --mode cached --outdir outputs
python scripts/reproduce_supplement_figures.py --mode cached --outdir outputs
```

If a full reproduction script is not obvious for a particular diagnostic, see the
configuration files, experiment scripts, and released cached outputs for
experiment details.

## Reproducibility Notes

The synthetic experiments generate their own data. Benchmark regression workflows
use external numeric regression datasets or cached benchmark files. Large cached
data and output artifacts are intentionally kept in Google Drive rather than
GitHub. Fixed seeds and cached outputs are used where appropriate to make the
paper tables and figures inspectable without rerunning every full-grid
experiment.

By default, scripts write generated files to `outputs/`, which is ignored by Git.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate scrp
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the associated paper and this software release.
See `CITATION.cff` for citation metadata.

## License

This code release is provided under the MIT License. See `LICENSE`.
