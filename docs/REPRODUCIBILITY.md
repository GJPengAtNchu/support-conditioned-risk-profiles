# Reproducibility Notes

The release supports two modes.

## Fast reproduction

Download cached result files from Google Drive and regenerate figures/tables:

```bash
python scripts/reproduce_main_figures.py --mode cached --outdir outputs
```

## Full reproduction

Rerun experiments from scripts and fixed seeds:

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

Full-scale runs used in the paper can take substantially longer than the
`--fast` commands. See the README for expected runtime ranges.
