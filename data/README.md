# Data

This directory tracks lightweight reference inputs and metadata. Large raw data exports and processed panels are local inputs and are ignored by Git.

The processed feature panel is expected at `data/processed/feature_panel.parquet` when running the experiment. It can be rebuilt from the full local raw CSV export with `scripts/build_processed_from_raw.py`.
