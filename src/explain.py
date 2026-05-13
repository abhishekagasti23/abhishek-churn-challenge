"""
SHAP explainability functions.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import (
    PIPELINE_PATH,
    SHAP_FIGURES_DIR,
)


def load_pipeline():
    """
    Load saved pipeline from disk.
    """

    return joblib.load(PIPELINE_PATH)


def get_feature_names(pipeline):
    """
    Extract feature names after preprocessing.
    """

    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []

    for _, transformer, columns in preprocessor.transformers_:

        if hasattr(transformer, "named_steps"):

            last_step = list(transformer.named_steps.values())[-1]

            # Handle encoded categorical columns
            if hasattr(
                last_step,
                "get_feature_names_out",
            ):

                names = last_step.get_feature_names_out(columns)

                feature_names.extend(names.tolist())

            else:

                feature_names.extend(columns)

        else:

            feature_names.extend(columns)

    return feature_names


def get_explainer(
    model,
    X_transformed,
):
    """
    Select appropriate SHAP explainer
    based on model type.
    """

    model_name = model.__class__.__name__

    # Linear models
    if model_name == "LogisticRegression":

        return shap.LinearExplainer(
            model,
            X_transformed,
        )

    # Tree-based models
    elif model_name in [
        "RandomForestClassifier",
        "XGBClassifier",
        "LGBMClassifier",
    ]:

        return shap.TreeExplainer(model)

    # Generic fallback
    return shap.Explainer(
        model.predict,
        X_transformed,
    )


def create_shap_summary(
    pipeline,
    X,
):
    """
    Create SHAP summary plot
    showing overall feature importance.
    """

    # Use sample for faster SHAP computation
    X_sample = X.sample(
        min(300, len(X)),
        random_state=42,
    )

    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    # Transform features
    X_transformed = preprocessor.transform(X_sample)

    feature_names = get_feature_names(pipeline)

    # Select proper SHAP explainer
    explainer = get_explainer(
        model,
        X_transformed,
    )

    shap_values = explainer.shap_values(X_transformed)

    # Some explainers return list
    if isinstance(shap_values, list):

        shap_values = shap_values[1]

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
    )

    save_path = SHAP_FIGURES_DIR / "shap_summary.png"

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
    )

    plt.close()

    print(f"SHAP summary saved to: " f"{save_path}")


def explain_prediction(
    pipeline,
    X,
):
    """
    Return top feature
    driving prediction.
    """

    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    feature_names = get_feature_names(pipeline)

    # Select proper SHAP explainer
    explainer = get_explainer(
        model,
        X_transformed,
    )

    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):

        shap_values = shap_values[1]

    # Find strongest feature impact
    impact = np.abs(shap_values[0])

    top_idx = np.argmax(impact)

    top_feature = feature_names[top_idx]

    return top_feature


if __name__ == "__main__":

    print("Explainability module loaded.")
