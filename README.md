# Support-Conditioned Risk Profiles

Code for reproducing the experiments in “Support-Conditioned Risk Profiles for
Learning under Nonuniform Design.”

This repository contains experiment scripts, plotting utilities, environment
metadata, and reproducibility notes. It intentionally does **not** include the
manuscript source, paper PDFs, LaTeX build files, large raw datasets, cached
results, or trained model dumps.

## Overview

Support-conditioned risk profiles summarize prediction error as a function of
empirical support. The experiments in this repository study profile estimation,
support-gap diagnostics, local-polynomial rate behavior, KRR bias--variance
reshaping, profile-aware model selection, acquisition, and real-data
applicability.

## Repository structure

```text
experiments/
  exp1_profile_estimability.py
  exp2_support_gap.py
  exp3_local_polynomial.py
  exp4_krr_decomposition.py
  exp5_support_only_insufficiency.py
  exp6_model_selection.py
  exp7_acquisition.py
  exp8_real_data.py
scripts/
  smoke_test.py
  reproduce_main_figures.py
  reproduce_supplement_figures.py
  make_tables.py
  download_data.py
docs/
  REPRODUCIBILITY.md
  GOOGLE_DRIVE_DATA.md
data/
  README.md
results/
  README.md
figures/
  README.md
```

By default, scripts write generated files to `outputs/`, which is ignored by
Git.

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

## Quick start

Run a lightweight syntax/import check:

```bash
python scripts/smoke_test.py
```

Run a fast synthetic experiment:

```bash
python experiments/exp6_model_selection.py --fast --outdir outputs
```

Create the output directory layout:

```bash
python scripts/make_tables.py
```

## Reproducing experiments

### Fast reproduction

Fast reproduction uses cached result files downloaded from Google Drive to
regenerate figures and tables.

```bash
python scripts/download_data.py
python scripts/reproduce_main_figures.py --mode cached --outdir outputs
```

Large cached result files will be hosted separately:

[Google Drive data and cached results](https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing)

### Full reproduction

Full reproduction reruns experiments from scripts and random seeds.

```bash
python experiments/exp1_profile_estimability.py --outdir outputs
python experiments/exp2_support_gap.py --fast --outdir outputs
python experiments/exp3_local_polynomial.py --fast --outdir outputs
python experiments/exp4_krr_decomposition.py --fast --outdir outputs
python experiments/exp5_support_only_insufficiency.py --fast --outdir outputs
python experiments/exp6_model_selection.py --fast --outdir outputs
python experiments/exp7_acquisition.py --fast --outdir outputs
python experiments/exp8_real_data.py --fast --outdir outputs
```

Remove `--fast` for full-scale runs where supported. Full runs can be
substantially slower than the commands above.

## Cached results and large data

This release excludes:

- raw benchmark datasets;
- OpenML and local dataset caches;
- large generated CSV/JSON result bundles;
- generated figures and tables;
- trained model dumps or checkpoints.

Use the Google Drive archive when available:

[Google Drive data and cached results](https://drive.google.com/drive/folders/1degj8NKU1FJTib9hocOrUMixivfM_5vt?usp=sharing)

See `docs/GOOGLE_DRIVE_DATA.md` for the expected local layout.

## Benchmark datasets

Synthetic experiments generate their own data. The real-data experiment
(`experiments/exp8_real_data.py`) uses standard numeric regression datasets
when available locally or through configured dataset loaders. Place external
datasets under `data/raw/` or pass:

```bash
python experiments/exp8_real_data.py --data-dir data/raw --outdir outputs
```

The manuscript results used cached benchmark files, which will be provided
through Google Drive rather than committed to GitHub.

## Expected runtime

Approximate runtimes depend strongly on hardware and whether cached results are
used.

- Smoke test: under 1 minute.
- Fast synthetic runs: minutes to tens of minutes.
- Full synthetic runs: tens of minutes to several hours.
- Real-data experiment: dataset-dependent; full repeated-split runs may take
  several hours.

## Citation

If you use this code, please cite:

```bibtex
@article{support_conditioned_risk_profiles_2026,
  title   = {Support-Conditioned Risk Profiles for Learning under Nonuniform Design},
  author  = {Authors},
  year    = {2026}
}
```

See `CITATION.cff` for software citation metadata.

## License

This code release is provided under the MIT License. See `LICENSE`.

## Contact

Please open a GitHub issue in the public repository once available:

`support-conditioned-risk-profiles`
