"""PyTorch training script for SageMaker Script Mode.

Reads hyperparameters from argparse (overridden per training job),
writes the trained model artifact and emits JSON metrics the pipeline
can gate on.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Classifier(nn.Module):
    def __init__(self, in_features: int = 20, hidden: int = 64, classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./out"))
    args = parser.parse_args()

    torch.manual_seed(42)
    # Synthetic dataset placeholder — swap in a real DataLoader
    X = torch.randn(2000, 20)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

    model = Classifier()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total, correct = 0, 0
    for epoch in range(args.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total += yb.numel()
            correct += (out.argmax(dim=1) == yb).sum().item()
        print(f"epoch={epoch + 1} loss={loss.item():.4f}")

    accuracy = correct / total
    print(f"final_accuracy={accuracy:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))

    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": {"value": accuracy}}, f)


if __name__ == "__main__":
    main()
