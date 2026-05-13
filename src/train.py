"""
Train churn prediction models.
"""

import joblib
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import (
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)

from src.data_loader import load_or_generate

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix_chart,
    plot_precision_recall_curve,
    plot_roc_curve,
)

from src.feature_engineering import (
    add_engineered_features,
)

from src.pipeline import build_pipeline


def prepare_data():
    """
    Load dataset and prepare
    train/validation/test splits.
    """

    # Load churn dataset
    df = load_or_generate()

    # Split off test set first
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_SEED,
    )

    # Create validation split
    val_size = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df[TARGET],
        random_state=RANDOM_SEED,
    )

    # Compute plan medians
    # using training data only
    plan_medians = train_df.groupby("plan_type")["monthly_spend"].median().to_dict()

    # Add engineered features
    train_df = add_engineered_features(
        train_df,
        plan_medians,
    )

    val_df = add_engineered_features(
        val_df,
        plan_medians,
    )

    test_df = add_engineered_features(
        test_df,
        plan_medians,
    )

    # Remove target + ID columns
    drop_cols = [
        TARGET,
        "customer_id",
    ]

    X_train = train_df.drop(
        columns=drop_cols,
    )

    y_train = train_df[TARGET]

    X_val = val_df.drop(
        columns=drop_cols,
    )

    y_val = val_df[TARGET]

    X_test = test_df.drop(
        columns=drop_cols,
    )

    y_test = test_df[TARGET]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def train_models():
    """
    Train and compare models.
    """

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = prepare_data()

    print(
        f"\nSplit sizes -> "
        f"Train: {len(y_train)}, "
        f"Val: {len(y_val)}, "
        f"Test: {len(y_test)}"
    )

    # Models for comparison
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    }

    best_model = None
    best_pipeline = None
    best_auc = 0

    # Train and evaluate models
    for name, model in models.items():

        print(f"\nTraining {name}")
        print("-" * 40)

        # Build preprocessing + model pipeline
        pipeline = build_pipeline(model)

        # Train model
        pipeline.fit(
            X_train,
            y_train,
        )

        # Validation metrics
        val_results = evaluate_model(
            pipeline,
            X_val,
            y_val,
        )

        # Test metrics
        test_results = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        print(f"Validation ROC-AUC: " f"{val_results['roc_auc']:.4f}")

        print(f"Test ROC-AUC: " f"{test_results['roc_auc']:.4f}")

        print(test_results)

        # Track best model
        if test_results["roc_auc"] > best_auc:

            best_auc = test_results["roc_auc"]

            best_model = name

            best_pipeline = pipeline

    # Optional Neural Network

    try:

        from tensorflow import keras
        from sklearn.metrics import roc_auc_score

        print("\nTraining Neural Network")
        print("-" * 40)

        # Preprocess data for neural network
        preprocessor = build_pipeline(LogisticRegression()).named_steps["preprocessor"]

        X_train_nn = preprocessor.fit_transform(X_train)

        X_test_nn = preprocessor.transform(X_test)

        # Simple neural network
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(X_train_nn.shape[1],)),
                keras.layers.Dense(
                    64,
                    activation="relu",
                ),
                keras.layers.Dense(
                    32,
                    activation="relu",
                ),
                keras.layers.Dense(
                    1,
                    activation="sigmoid",
                ),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["AUC"],
        )

        model.fit(
            X_train_nn,
            y_train,
            epochs=10,
            batch_size=256,
            verbose=0,
        )

        # Predict probabilities
        nn_probs = model.predict(
            X_test_nn,
            verbose=0,
        ).flatten()

        nn_auc = roc_auc_score(
            y_test,
            nn_probs,
        )

        print(f"Test ROC-AUC: " f"{nn_auc:.4f}")

    except Exception as e:

        print(f"\nNeural network skipped: {e}")

    print("\nBest Model")
    print("-" * 40)

    print(best_model)

    print(f"ROC-AUC: {best_auc:.4f}")

    # Generate evaluation plots
    plot_roc_curve(
        best_pipeline,
        X_test,
        y_test,
        model_name=best_model,
    )

    plot_precision_recall_curve(
        best_pipeline,
        X_test,
        y_test,
    )

    plot_confusion_matrix_chart(
        best_pipeline,
        X_test,
        y_test,
    )

    # Save trained pipeline
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_pipeline,
        PIPELINE_PATH,
    )

    print("\nPipeline saved successfully.")


if __name__ == "__main__":

    train_models()
