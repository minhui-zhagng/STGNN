"""Train STGNN for one edge-feature mode and save weights and configuration."""

import json
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from common import arguments, load_config
from model import StatusGNN_withEdgeAttr


# 1. Reproducibility and training data
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(config, output_path, mode):
    """Load training graphs only; preserve one graph per optimization step."""
    if mode not in config["edge_modes"]:
        raise ValueError("Unknown mode")

    graph_path = output_path / "graphs" / mode / "train.pt"
    # Only load trusted, locally generated PyTorch files.
    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)

    assert config["batch_size"] == 1, "Historical model requires one graph per batch"
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    labels = [int(graph.y.item()) for graph in graphs]
    counts = np.bincount(labels, minlength=6)
    if np.any(counts == 0):
        raise ValueError("A training class is absent")

    # Estimate weights from TRAIN labels only.
    weights = np.log1p(1.0 / counts)
    weights = weights / weights.sum() * 6
    return graphs, loader, counts, weights


# 2. Model, optimizer, and classification loss
def initialize_model(config, weights, device):
    model = StatusGNN_withEdgeAttr(
        hidden_channels=config["hidden_dimension"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    return model, optimizer, criterion


# 3. Training loop
def train_one_epoch(model, loader, optimizer, criterion, device, margin_coefficient):
    """Preserve the existing cross-entropy plus margin objective."""
    model.train()
    total_loss = 0.0

    for graph in loader:
        graph = graph.to(device)
        optimizer.zero_grad()

        scores = model(graph)
        target = graph.y.view(-1)

        margin = torch.clamp(scores.max() - scores[target[0]], min=0.0)
        classification_loss = criterion(scores.unsqueeze(0), target)
        loss = classification_loss + margin_coefficient * margin

        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

    return total_loss / len(loader.dataset)


def train_model(model, loader, optimizer, criterion, config, device):
    history = []

    for epoch in range(config["epochs"]):
        average_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            config["margin_coefficient"],
        )
        history.append(average_loss)
        print(epoch + 1, average_loss, flush=True)

    return history


# 4. Save model weights and the actual training configuration
def save_model(model, config, output_path, mode, counts, weights):
    folder = output_path / "training" / mode
    folder.mkdir(parents=True, exist_ok=True)

    effective_config = dict(
        config,
        edge_mode=mode,
        training_class_counts=counts.tolist(),
        class_weights=weights.tolist(),
    )

    checkpoint = {
        "model": model.state_dict(),
        "config": effective_config,
    }
    torch.save(checkpoint, folder / "model.pt")
    (folder / "config.json").write_text(
        json.dumps(effective_config, indent=2),
        encoding="utf-8",
    )


def main():
    args = arguments("Train one STGNN edge mode")
    config, output_path = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, loader, counts, weights = prepare_data(config, output_path, args.mode)
    model, optimizer, criterion = initialize_model(config, weights, device)
    train_model(model, loader, optimizer, criterion, config, device)
    save_model(model, config, output_path, args.mode, counts, weights)


if __name__ == "__main__":
    main()
