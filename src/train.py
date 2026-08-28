import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import get_model


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += predictions.eq(targets).sum().item()

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main() -> None:

    project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / "configs" / "training_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    config = load_config(config_path)

    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    patience = config["training"]["early_stopping_patience"]
    num_workers = config["training"]["num_workers"]

    data_dir = project_root / config["data"]["data_dir"]

    checkpoint_dir = (
        project_root / config["output"]["checkpoint_dir"]
    )

    model_name = config["output"]["model_name"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        json.dumps(
            {
                "event": "training_started",
                "architecture": architecture,
                "num_classes": num_classes,
                "device": str(device),
            }
        ),
        flush=True,
    )

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = checkpoint_dir / model_name

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
        }

        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "architecture": architecture,
                    "num_classes": num_classes,
                },
                save_path,
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "path": str(save_path),
                        "epoch": epoch,
                        "val_loss": round(val_loss, 4),
                    }
                ),
                flush=True,
            )

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "patience": patience,
                        }
                    ),
                    flush=True,
                )

                break

    print(
        json.dumps(
            {
                "event": "training_complete",
                "best_val_loss": round(best_val_loss, 4),
                "checkpoint": str(save_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
