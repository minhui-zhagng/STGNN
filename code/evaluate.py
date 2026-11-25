"""Predict the original test graphs once and save six-class evaluation results."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from common import arguments, load_config
from model import StatusGNN_withEdgeAttr


# 1. Load the saved model
def load_model(training_folder, mode):
    # Only load trusted, locally generated PyTorch files.
    saved = torch.load(
        training_folder / "model.pt",
        map_location="cpu",
        weights_only=False,
    )

    if saved["config"]["edge_mode"] != mode:
        raise ValueError("Mode mismatch")

    model = StatusGNN_withEdgeAttr(
        hidden_channels=saved["config"]["hidden_dimension"]
    )
    model.load_state_dict(saved["model"])
    model.eval()
    return model


# 2. Predict without updating model parameters or replicating test samples
def predict(model, graphs):
    truth = []
    prediction = []

    with torch.no_grad():
        for graph in graphs:
            scores = model(graph)
            prediction.append(int(scores.argmax()))
            truth.append(int(graph.y.item()))

    return truth, prediction


# 3. Compute overall and class-wise metrics
def compute_metrics(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(range(6)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "classwise": classification_report(
            truth,
            prediction,
            labels=list(range(6)),
            output_dict=True,
            zero_division=0,
        ),
        "caveat": (
            "Historical input graph construction has known issues; see README. "
            "Not a verified independent reproduction."
        ),
    }


# 4. Save metrics, per-sample predictions, and confusion counts
def save_results(folder, truth, prediction, metrics):
    folder.mkdir(exist_ok=True)
    (folder / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "true_label", "predicted_label"])
        writer.writerows(
            (index + 1, label, predicted)
            for index, (label, predicted) in enumerate(zip(truth, prediction))
        )

    counts = confusion_matrix(truth, prediction, labels=list(range(6)))
    np.savetxt(folder / "confusion_counts.csv", counts, delimiter=",", fmt="%d")
    return counts


# 5. Draw a row-normalized percentage confusion matrix
def plot_confusion_matrix(counts, path):
    support = counts.sum(axis=1, keepdims=True)
    percent = np.divide(
        counts * 100.0,
        support,
        out=np.zeros((6, 6)),
        where=support > 0,
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(percent, cmap="Blues", vmin=0, vmax=100)

    for row in range(6):
        for column in range(6):
            axis.text(
                column,
                row,
                f"{percent[row, column]:.2f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if percent[row, column] > 50 else "black",
            )

    axis.set(
        xticks=range(6),
        yticks=range(6),
        xlabel="Predicted label",
        ylabel="True label",
    )
    figure.colorbar(image, ax=axis, label="Percentage (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main():
    args = arguments("Predict test CSVs without oversampling")
    config, output_path = load_config(args.config)
    training_folder = output_path / "training" / args.mode
    model = load_model(training_folder, args.mode)

    graphs = torch.load(
        output_path / "graphs" / args.mode / "test.pt",
        map_location="cpu",
        weights_only=False,
    )
    truth, prediction = predict(model, graphs)
    metrics = compute_metrics(truth, prediction)

    evaluation_folder = training_folder / "evaluation"
    counts = save_results(evaluation_folder, truth, prediction, metrics)
    plot_confusion_matrix(counts, evaluation_folder / "confusion_percent.png")

    overall_metrics = {
        key: value for key, value in metrics.items() if key != "classwise"
    }
    print(json.dumps(overall_metrics, indent=2))


if __name__ == "__main__":
    main()
