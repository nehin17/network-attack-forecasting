CICIDS binary forecaster pipeline integration
=============================================

Files in this package:
- model/src/finalize_cicids_binary_forecaster.py
  Re-runs the selected oversampling-vs-class-weighting comparison once and saves
  the validation-selected model together with its threshold and exact feature schema.
  This is required because the previous comparison scripts printed metrics but did
  not save the fitted comparison model.
- model/src/future_attack_predictor.py
  Replacement predictor loading the saved binary model bundle.
- model/src/network_prediction_pipeline.py
  Replacement pipeline retaining the existing current attack classifier and output engine.

Install/copy:
1. Copy these files into the corresponding paths in your local repository.
2. Run from the project root:
   .\.venv\Scripts\python.exe -u model\src\finalize_cicids_binary_forecaster.py
3. This writes:
   model/artifacts/cicids_binary_forecast/final_binary_forecaster.joblib
4. Only after that, run your pipeline from a caller that supplies a (10, N) matrix
   of consecutive minute-level CICIDS aggregate features in the exact saved feature
   order. The old (10, 70) multiclass CICIDS sequence is NOT compatible with the new
   binary forecaster's schema.

Important:
- The model is binary (ATTACK vs BENIGN), not future attack subtype classification.
- The current attack classifier is unchanged and still expects its existing UNSW input.
- This pipeline combines a CICIDS future forecast and UNSW current classification because
  that is how the existing project is structured; it is not yet a single-dataset live IDS.
- The selected threshold comes from validation only. The observed test result on the
  final threshold experiment was 100% recall and 10% precision for the oversampling model
  on 26 attack-positive test windows, so report this as a limited test-split result.
