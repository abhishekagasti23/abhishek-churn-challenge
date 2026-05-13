"""
Model evaluation functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import EDA_FIGURES_DIR


def evaluate_model(
    pipeline,
    X,
    y,
    threshold=0.5,
):
    """
    Calculate main classification metrics.
    """

    # Predict churn probabilities
    y_prob = pipeline.predict_proba(X)[:, 1]

    # Convert probabilities into binary predictions
    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # Main evaluation metrics
    results = {
        "roc_auc": round(
            roc_auc_score(y, y_prob),
            4,
        ),
        "precision": round(
            precision_score(y, y_pred),
            4,
        ),
        "recall": round(
            recall_score(y, y_pred),
            4,
        ),
        "f1_score": round(
            f1_score(y, y_pred),
            4,
        ),
    }

    print("\nClassification Report")
    print("-" * 40)

    print(
        classification_report(
            y,
            y_pred,
        )
    )

    return results


def plot_roc_curve(
    pipeline,
    X,
    y,
    model_name="Model",
):
    """
    Plot ROC curve.
    """

    y_prob = pipeline.predict_proba(X)[:, 1]

    # Calculate ROC values
    fpr, tpr, _ = roc_curve(
        y,
        y_prob,
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )

    # Random baseline
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    save_path = (
        EDA_FIGURES_DIR
        / "roc_curve.png"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(save_path)

    plt.close()


def plot_precision_recall_curve(
    pipeline,
    X,
    y,
):
    """
    Plot Precision-Recall curve.
    """

    y_prob = pipeline.predict_proba(X)[:, 1]

    precision, recall, _ = (
        precision_recall_curve(
            y,
            y_prob,
        )
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall,
        precision,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curve"
    )

    plt.tight_layout()

    save_path = (
        EDA_FIGURES_DIR
        / "precision_recall_curve.png"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(save_path)

    plt.close()


def plot_confusion_matrix_chart(
    pipeline,
    X,
    y,
    threshold=0.5,
):
    """
    Plot confusion matrix heatmap.
    """

    y_prob = pipeline.predict_proba(X)[:, 1]

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # Create confusion matrix
    cm = confusion_matrix(
        y,
        y_pred,
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    save_path = (
        EDA_FIGURES_DIR
        / "confusion_matrix.png"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(save_path)

    plt.close()