import os
import torch
import torch.nn as nn
from tqdm import tqdm


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=running_loss / total, acc=100 * correct / total)

    return running_loss / len(loader), 100 * correct / total


# ── Evaluation loop ──────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        loop = tqdm(loader, desc="  Eval ", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=running_loss / total, acc=100 * correct / total)

    return running_loss / len(loader), 100 * correct / total


# ── Full training run ────────────────────────────────────────────────────────
def fit(model, train_loader, val_loader, optimizer, criterion,
        num_epochs, device, scheduler=None, output_dir: str = None):
    """
    Train for num_epochs, evaluating on val_loader each epoch.
    Returns history dict with train/val loss and accuracy lists.

    If output_dir is provided, saves a training_log.txt there.
    """
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    log_lines = []

    for epoch in range(1, num_epochs + 1):
        header = f"Epoch {epoch}/{num_epochs}"
        print(header)

        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer,
                                        criterion, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        line = (f"train_loss={t_loss:.4f}  train_acc={t_acc:.2f}%  "
                f"val_loss={v_loss:.4f}  val_acc={v_acc:.2f}%")
        print(line)
        log_lines.append(f"{header}\n{line}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fpath = os.path.join(output_dir, "training_log.txt")
        with open(fpath, "w") as f:
            f.write("\n".join(log_lines))
        print(f"Saved training log: {fpath}")

    return history