# LFQL Protocol

This protocol defines an exploratory localized factor experiment using 21 numeric factors.

The local training window is a fixed half-width of +/-15 percentage points around the B-period threshold center. The primary core comparison uses a +/-3 percentage point interval. Raw interval boundaries are frozen from the 2021 empirical distribution.

The LightGBM design matrix contains 21 numeric features plus 32 industry dummy columns generated from `sw_l1_name` at runtime.
