import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "DATA",
    "processed",
    "UNSW",
    "UNSW_NB15_training-set.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "DATA",
    "processed",
    "UNSW",
    "UNSW_NB15_testing-set.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results",
    "unsw_xgboost"
)

os.makedirs(RESULT_DIR, exist_ok=True)


CATEGORICAL = [
    "proto",
    "service",
    "state",
]

DROP_COLUMNS = [
    "id",
    "label",
]

TARGET = "attack_cat"


def build_encoders(train_df):

    encoders = {}

    for column in CATEGORICAL:

        values = sorted(
            train_df[column]
            .astype(str)
            .unique()
            .tolist()
        )

        # 0 is reserved for UNKNOWN.
        mapping = {
            value: index + 1
            for index, value in enumerate(values)
        }

        encoders[column] = mapping

    return encoders


def encode_dataframe(df, encoders):

    df = df.copy()

    for column in CATEGORICAL:

        mapping = encoders[column]

        df[column] = (
            df[column]
            .astype(str)
            .map(mapping)
            .fillna(0)
            .astype(np.int32)
        )

    return df


def main():

    print("=" * 70)
    print("UNSW-NB15 — ATTACK TYPE — XGBOOST")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"\nTraining shape: {train_df.shape}")
    print(f"Testing shape : {test_df.shape}")

    # --------------------------------------------------
    # ENCODERS — FIT TRAIN ONLY
    # --------------------------------------------------

    encoders = build_encoders(train_df)

    # --------------------------------------------------
    # TARGET MAPPING — FIT TRAIN ONLY
    # --------------------------------------------------

    classes = sorted(
        train_df[TARGET]
        .astype(str)
        .unique()
        .tolist()
    )

    class_to_id = {
        name: index
        for index, name in enumerate(classes)
    }

    id_to_class = {
        index: name
        for name, index in class_to_id.items()
    }

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column not in DROP_COLUMNS + [TARGET]
    ]

    X_train_df = train_df[feature_columns].copy()
    X_test_df = test_df[feature_columns].copy()

    X_train_df = encode_dataframe(
        X_train_df,
        encoders
    )

    X_test_df = encode_dataframe(
        X_test_df,
        encoders
    )

    y_train = (
        train_df[TARGET]
        .astype(str)
        .map(class_to_id)
        .to_numpy()
    )

    y_test = (
        test_df[TARGET]
        .astype(str)
        .map(class_to_id)
        .to_numpy()
    )

    if np.isnan(y_test).any():
        raise ValueError(
            "Test set contains a target class "
            "not present in training."
        )

    X_train = X_train_df.to_numpy(
        dtype=np.float32
    )

    X_test = X_test_df.to_numpy(
        dtype=np.float32
    )

    print(f"\nInput features: {len(feature_columns)}")
    print(f"Classes: {len(classes)}")

    print("\nClasses:")
    for index, name in id_to_class.items():
        print(f"  {index}: {name}")

    print("\nEncoded shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # --------------------------------------------------
    # XGBOOST
    # --------------------------------------------------

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_test, y_test),
        ],
        verbose=False,
    )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {precision:.4f}")
    print(f"Macro Recall      : {recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")

    print("\nPer-class results:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0
        )
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    model_path = os.path.join(
        RESULT_DIR,
        "unsw_attack_type_xgboost.json"
    )

    model.save_model(model_path)

    # --------------------------------------------------
    # SAVE PREPROCESSING ARTIFACT
    # --------------------------------------------------

    preprocessing = {
        "feature_columns": feature_columns,
        "categorical_columns": CATEGORICAL,
        "encoders": encoders,
        "class_to_id": class_to_id,
        "id_to_class": {
            str(k): v
            for k, v in id_to_class.items()
        },
    }

    preprocessing_path = os.path.join(
        RESULT_DIR,
        "unsw_preprocessing.json"
    )

    with open(
        preprocessing_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            preprocessing,
            f,
            indent=2
        )

    # --------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------

    np.savez(
        os.path.join(
            RESULT_DIR,
            "test_predictions.npz"
        ),
        y_true=y_test,
        y_pred=predictions,
        probabilities=probabilities,
    )

    print("\nSaved:")
    print(model_path)
    print(preprocessing_path)
    print(
        os.path.join(
            RESULT_DIR,
            "test_predictions.npz"
        )
    )


if __name__ == "__main__":
    main()