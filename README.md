# LFQL Localized Factor ML

This repository contains a zero-history localized factor machine-learning framework for CSI1000 research. The framework uses 21 numeric factors and runtime industry one-hot columns for a 53-column LightGBM design matrix.

## Repository Contents

- `configs/`: feature set and experiment configuration.
- `data/`: lightweight tracked reference inputs and data metadata.
- `docs/`: protocol and reproducibility notes.
- `src/`: shared LFQL utilities and experiment implementation.
- `scripts/`: input verification, feature-panel build, and experiment runner.
- `tests/`: lightweight configuration and import tests.

Generated outputs, reports, model artifacts, prediction parquet files, and processed panels are local build products and are ignored by Git.

## Feature Contract

The numeric feature set contains 21 columns. Industry dummy columns are generated from `sw_l1_name` at runtime and are expected to contain 32 columns for the CSI1000 panel, producing 53 model input columns in total.

`bp` is defined as `1 / pb`; the framework does not create a separate `pb` model feature.

## Basic Checks

```bash
python scripts/verify_inputs.py
python -m unittest discover -s tests
```

To rebuild the processed panel when the full local raw data export is available:

```bash
python scripts/build_processed_from_raw.py
```

To run the experiment after the processed panel exists:

```bash
python scripts/run_localized_factor_ml.py
```
