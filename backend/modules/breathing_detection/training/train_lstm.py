"""Train the lightweight breathing LSTM with validation-only model selection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.modules.breathing_detection.preprocessing.cache import (
    TemporalCache,
    build_temporal_cache,
    load_temporal_cache,
)
from backend.modules.breathing_detection.preprocessing.data import BreathingDataset
from backend.modules.breathing_detection.preprocessing.temporal import (
    TEMPORAL_FEATURE_NAMES,
    TOP_VARIANCE_FEATURES,
    INTERPOLATED_FEATURES,
    extract_interpolated_subcarriers,
    extract_temporal_statistics,
    extract_top_variance_subcarriers,
)
from backend.modules.breathing_detection.training.lstm import LightweightBreathingLSTM
from backend.modules.breathing_detection.training.periodicity_baseline import (
    binary_metrics,
    select_macro_f1_threshold,
)


class TemporalDataset(Dataset):
    def __init__(
        self,
        source: BreathingDataset,
        limit: int | None = None,
        representation: str = "top_variance",
    ) -> None:
        self.source = source
        self.length = min(len(source), limit) if limit is not None else len(source)
        self.representation = representation
        self.input_size = (
            INTERPOLATED_FEATURES
            if representation == "interpolated"
            else TOP_VARIANCE_FEATURES
            if representation == "top_variance"
            else len(TEMPORAL_FEATURE_NAMES)
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        sample = self.source[index]
        if sample.assessable:
            try:
                sequence = (
                    extract_interpolated_subcarriers(sample.x)
                    if self.representation == "interpolated"
                    else extract_top_variance_subcarriers(sample.x)
                    if self.representation == "top_variance"
                    else extract_temporal_statistics(sample.x)
                )
                weight = 1.0
            except ValueError:
                sequence = np.zeros((300, self.input_size), dtype=np.float32)
                weight = 0.0
        else:
            sequence = np.zeros((300, self.input_size), dtype=np.float32)
            weight = 0.0
        return (
            torch.from_numpy(sequence),
            torch.tensor(sample.label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


class CachedTemporalDataset(Dataset):
    def __init__(self, cache: TemporalCache) -> None:
        self.cache = cache

    def __len__(self) -> int:
        return len(self.cache.labels)

    def __getitem__(self, index: int):
        # Convert one float16 memory-mapped row to float32 for model input.
        sequence = np.asarray(self.cache.sequences[index], dtype=np.float32)
        return (
            torch.from_numpy(sequence),
            torch.tensor(float(self.cache.labels[index]), dtype=torch.float32),
            torch.tensor(float(self.cache.weights[index]), dtype=torch.float32),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/BreathingDetection"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/modules/breathing_detection/outputs/lightweight_lstm"),
    )
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--representation",
        choices=("interpolated", "top_variance", "statistics"),
        default="interpolated",
    )
    parser.add_argument("--projection-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Build/resume missing interpolated caches before training",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_balance(source: BreathingDataset, limit: int | None) -> tuple[int, int]:
    count = min(len(source), limit) if limit is not None else len(source)
    labels = [source._rows[index]["label"] for index in range(count)]
    return labels.count("empty"), labels.count("sleep")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_weight = 0.0
    for sequence, labels, weights in loader:
        sequence, labels, weights = (
            sequence.to(device),
            labels.to(device),
            weights.to(device),
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(sequence)
            losses = criterion(logits, labels) * weights
            denominator = weights.sum().clamp(min=1.0)
            loss = losses.sum() / denominator
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(losses.sum().detach().cpu())
        total_weight += float(weights.sum().detach().cpu())
    return total_loss / max(total_weight, 1.0)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities, labels, weights = [], [], []
    for sequence, batch_labels, batch_weights in loader:
        logits = model(sequence.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
        weights.append(batch_weights.numpy())
    probability = np.concatenate(probabilities)
    label = np.concatenate(labels).astype(np.int64)
    weight = np.concatenate(weights).astype(bool)
    return probability[weight], label[weight], int((~weight).sum())


def cached_or_live_dataset(
    source: BreathingDataset,
    limit: int | None,
    representation: str,
    cache_dir: Path | None,
    build_cache: bool,
) -> Dataset:
    if representation != "interpolated" or cache_dir is None:
        return TemporalDataset(source, limit, representation)
    count = min(len(source), limit) if limit is not None else len(source)
    cache = (
        build_temporal_cache(source, cache_dir, limit=limit)
        if build_cache
        else load_temporal_cache(cache_dir, source.split, count)
    )
    return CachedTemporalDataset(cache)


def loader_for_dataset(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    train_source = BreathingDataset(args.data_root, "train_id")
    val_source = BreathingDataset(args.data_root, "val_id")
    train_dataset = cached_or_live_dataset(
        train_source, args.max_train, args.representation, args.cache_dir, args.build_cache
    )
    val_dataset = cached_or_live_dataset(
        val_source, args.max_val, args.representation, args.cache_dir, args.build_cache
    )
    train_loader = loader_for_dataset(train_dataset, args.batch_size, True)
    val_loader = loader_for_dataset(val_dataset, args.batch_size, False)

    empty_count, sleep_count = class_balance(train_source, args.max_train)
    pos_weight = empty_count / max(sleep_count, 1)
    input_size = (
        INTERPOLATED_FEATURES
        if args.representation == "interpolated"
        else TOP_VARIANCE_FEATURES
        if args.representation == "top_variance"
        else len(TEMPORAL_FEATURE_NAMES)
    )
    model = LightweightBreathingLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        projection_size=(
            args.projection_size if args.representation == "interpolated" else None
        ),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device), reduction="none"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, device, criterion, None)
        val_probability, val_labels, val_skipped = predict(model, val_loader, device)
        threshold = select_macro_f1_threshold(val_labels, val_probability)
        val_metrics = binary_metrics(val_labels, val_probability, threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_f1": val_metrics["macro_f1"],
            "threshold": threshold,
        }
        history.append(row)
        print(json.dumps(row))
        if val_metrics["macro_f1"] > best_macro_f1 + 1e-6:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "threshold": threshold,
                    "epoch": epoch,
                    "representation": args.representation,
                    "input_size": input_size,
                    "projection_size": (
                        args.projection_size
                        if args.representation == "interpolated"
                        else None
                    ),
                    "hidden_size": args.hidden_size,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["threshold"])
    train_probability, train_labels, train_skipped = predict(model, train_loader, device)
    val_probability, val_labels, val_skipped = predict(model, val_loader, device)
    metrics = {
        "model": "lightweight_bidirectional_lstm",
        "representation": args.representation,
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "class_counts": {"empty": empty_count, "sleep": sleep_count},
        "train": binary_metrics(train_labels, train_probability, threshold),
        "validation": binary_metrics(val_labels, val_probability, threshold),
        "skipped": {"train": train_skipped, "validation": val_skipped},
        "history": history,
    }
    if args.evaluate_test:
        test_source = BreathingDataset(args.data_root, "test_id")
        test_dataset = cached_or_live_dataset(
            test_source,
            args.max_test,
            args.representation,
            args.cache_dir,
            args.build_cache,
        )
        test_loader = loader_for_dataset(test_dataset, args.batch_size, False)
        test_probability, test_labels, test_skipped = predict(model, test_loader, device)
        metrics["test"] = binary_metrics(test_labels, test_probability, threshold)
        metrics["skipped"]["test"] = test_skipped

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
