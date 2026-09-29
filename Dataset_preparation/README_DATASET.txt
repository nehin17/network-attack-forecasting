FORECASTING DATASET PACKAGE
===========================

Version
-------
forecasting_v1

Task
----
Binary multi-horizon network attack forecasting.

Input
-----
15 consecutive 1-minute observations.

Each observation contains:
80 frozen numeric predictors.

Input tensor shape
------------------
(samples, 15, 80)

Targets
-------
Binary point forecasts at:

[1, 5, 10, 15, 30]

Meaning:

y[:, 0] = attack state exactly t + 1 minute
y[:, 1] = attack state exactly t + 5 minutes
y[:, 2] = attack state exactly t + 10 minutes
y[:, 3] = attack state exactly t + 15 minutes
y[:, 4] = attack state exactly t + 30 minutes

Final partitions
----------------
TRAIN      : (1220, 15, 80)
VALIDATION : (398, 15, 80)
TEST       : (396, 15, 80)

Important leakage rules
-----------------------
1. The scaler was fitted on TRAIN only.
2. VALIDATION and TEST were never used to fit preprocessing.
3. No resampling was performed.
4. TEST must not be used for:
   - architecture selection
   - hyperparameter selection
   - epoch selection
   - early stopping
   - threshold selection
   - checkpoint selection

5. Validation is used for model selection.
6. Thresholds are selected from validation only.
7. TEST is reserved for final evaluation.

Model-development notebooks
---------------------------
Ordinary model-development notebooks should load:

    X_train.npy
    y_train.npy
    X_validation.npy
    y_validation.npy

They should NOT load TEST during architecture development.

Final-evaluation notebook
-------------------------
The final evaluation notebook may load:

    X_test.npy
    y_test.npy

only after the architecture, checkpoint strategy, and
validation-derived thresholds have been frozen.

Rich metadata
-------------
Files under metadata/ are evaluation metadata.

They must NOT be silently inserted into the model predictor tensor.

They are preserved for later analyses including:

    - attack-type generalization
    - known vs unseen attack types
    - onset / continuation
    - source/destination analysis
    - graph analysis
    - episode-level warning analysis
    - lead-time evaluation

Integrity
---------
manifest/checksums.json contains SHA-256 checksums for the exported artifacts.

Do not manually edit files inside this frozen dataset package.

If the dataset pipeline changes, create a NEW dataset version
rather than silently changing forecasting_v1.