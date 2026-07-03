# Reproducibility

Run the lightweight checks first:

```bash
python scripts/verify_inputs.py
python -m unittest discover -s tests
```

When the full local raw data export is available, rebuild the processed panel:

```bash
python scripts/build_processed_from_raw.py
```

Then run the experiment:

```bash
python scripts/run_localized_factor_ml.py
```

The experiment reads relative paths under `data/` and writes generated artifacts under `outputs/localized_factor_ml/`. Generated parquet files, models, predictions, reports, and logs are local build outputs.
