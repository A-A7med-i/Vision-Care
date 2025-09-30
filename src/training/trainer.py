import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.cuda import amp
import matplotlib.pyplot as plt
from src.config.constants import *
from sklearn.metrics import f1_score
from typing import Dict, Tuple, Optional, Any
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """
    Trainer class for multi-task training of DR and DME classification.

    This class handles training, evaluation, gradual unfreezing,
    learning rate scheduling, mixed precision, checkpointing,
    and history plotting.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        checkpoint_path: str,
        unfreeze: bool,
        unfreeze_step: Optional[int],
        device: Optional[str] = DEVICE,
        history_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.unfreeze = unfreeze
        self.optimizer = optimizer
        self.test_dataloader = test_dataloader
        self.unfreeze_step = unfreeze_step
        self.train_dataloader = train_dataloader

        self.device = device
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        self.scaler = amp.GradScaler(enabled=(self.device == "cuda"))

        self.history_path = history_path
        self.checkpoint_path = checkpoint_path
        self.best_avg_acc = float("-inf")

    @staticmethod
    def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute accuracy."""
        predictions = torch.argmax(y_pred, dim=1)
        correct = (predictions == y_true).sum().item()
        return correct / len(y_true)

    @staticmethod
    def f1_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute macro F1-score."""
        predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
        labels = y_true.cpu().numpy()
        return f1_score(labels, predictions, average="macro", zero_division=0)

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        training: bool = True,
    ) -> Tuple[float, float, float, float, float]:
        """
        Run one epoch (training or evaluation).

        Args:
            dataloader: train/test dataloader
            training: if True → training mode, else evaluation

        Returns:
            Tuple of (loss, acc_dr, acc_dme, f1_dr, f1_dme)
        """
        self.model.train() if training else self.model.eval()
        running_loss, acc_dr, acc_dme, f1_dr, f1_dme, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        context = torch.enable_grad() if training else torch.inference_mode()

        with context:
            for images, labels_dr, labels_dme in tqdm(
                dataloader, desc="Training" if training else "Testing", leave=False
            ):
                images, labels_dr, labels_dme = (
                    images.to(self.device),
                    labels_dr.to(self.device).long(),
                    labels_dme.to(self.device).long(),
                )

                with amp.autocast(self.device):
                    logits_dr, logits_dme = self.model(images)
                    total_loss = self.loss_fn(
                        logits_dr, labels_dr, logits_dme, labels_dme
                    )

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                running_loss += total_loss.item()
                acc_dr += self.accuracy_fn(labels_dr, logits_dr)
                acc_dme += self.accuracy_fn(labels_dme, logits_dme)
                f1_dr += self.f1_fn(labels_dr, logits_dr)
                f1_dme += self.f1_fn(labels_dme, logits_dme)
                count += 1

        return (
            running_loss / len(dataloader),
            acc_dr / count,
            acc_dme / count,
            f1_dr / count,
            f1_dme / count,
        )

    def gradual_unfreeze(self, epoch_idx: int) -> None:
        """
        Gradually unfreeze backbone blocks from the end.
        Args:
            epoch_idx (int): Current epoch index (0-based).
        """
        if self.unfreeze_step is None:
            return

        step = (epoch_idx + 1) // self.unfreeze_step

        try:
            blocks = list(self.model.backbone.features)
        except Exception:
            blocks = None

        if not blocks:
            return

        to_unfreeze = min(len(blocks), step)
        if to_unfreeze == 0:
            return

        for block in blocks[-to_unfreeze:]:
            for param in block.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
        print(f"Unfroze last {to_unfreeze} blocks at epoch {epoch_idx+1}")

    def plot_history(self, history: Dict[str, Any]) -> None:
        """Plot training and testing loss, accuracy, and F1 history."""
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(25, 5))

        # Loss Curve
        plt.subplot(1, 5, 1)
        plt.plot(
            epochs,
            history["train_loss"],
            color="#1f77b4",
            marker="o",
            linestyle="-",
            label="Train Loss",
        )
        plt.plot(
            epochs,
            history["test_loss"],
            color="#ff7f0e",
            marker="s",
            linestyle="--",
            label="Test Loss",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # DR Accuracy
        plt.subplot(1, 5, 2)
        plt.plot(
            epochs,
            history["train_acc_dr"],
            color="#1f77b4",
            marker="o",
            linestyle="-",
            label="Train DR",
        )
        plt.plot(
            epochs,
            history["test_acc_dr"],
            color="#ff7f0e",
            marker="o",
            linestyle="--",
            label="Test DR",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("DR Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # DME Accuracy
        plt.subplot(1, 5, 3)
        plt.plot(
            epochs,
            history["train_acc_dme"],
            color="#2ca02c",
            marker="s",
            linestyle="-",
            label="Train DME",
        )
        plt.plot(
            epochs,
            history["test_acc_dme"],
            color="#9467bd",
            marker="s",
            linestyle="--",
            label="Test DME",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("DME Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # DR F1
        plt.subplot(1, 5, 4)
        plt.plot(
            epochs,
            history["train_f1_dr"],
            color="#1f77b4",
            marker="o",
            linestyle="-",
            label="Train DR",
        )
        plt.plot(
            epochs,
            history["test_f1_dr"],
            color="#ff7f0e",
            marker="o",
            linestyle="--",
            label="Test DR",
        )
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.title("DR F1-Score")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # DME F1
        plt.subplot(1, 5, 5)
        plt.plot(
            epochs,
            history["train_f1_dme"],
            color="#2ca02c",
            marker="s",
            linestyle="-",
            label="Train DME",
        )
        plt.plot(
            epochs,
            history["test_f1_dme"],
            color="#9467bd",
            marker="s",
            linestyle="--",
            label="Test DME",
        )
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.title("DME F1-Score")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plt.tight_layout()
        if self.history_path:
            plt.savefig(self.history_path)
            print(f"Plots saved to {self.history_path}")
        plt.show()

    def save_checkpoint(self, epoch: int) -> None:
        """Save full training state."""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_avg_acc": self.best_avg_acc,
        }

        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

    def full_train(self) -> Dict[str, list]:
        """Run full training and evaluation loop across all epochs."""
        history = {
            "train_loss": [],
            "train_acc_dr": [],
            "train_acc_dme": [],
            "train_f1_dr": [],
            "train_f1_dme": [],
            "test_loss": [],
            "test_acc_dr": [],
            "test_acc_dme": [],
            "test_f1_dr": [],
            "test_f1_dme": [],
        }

        for epoch in range(self.epochs):
            if self.unfreeze:
                self.gradual_unfreeze(epoch)

            print(f"\nEpoch [{epoch+1}/{self.epochs}]")

            train_loss, train_acc_dr, train_acc_dme, train_f1_dr, train_f1_dme = (
                self.run_epoch(self.train_dataloader, training=True)
            )

            test_loss, test_acc_dr, test_acc_dme, test_f1_dr, test_f1_dme = (
                self.run_epoch(self.test_dataloader, training=False)
            )

            avg_test_acc = (test_acc_dr + test_acc_dme) / 2
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            if avg_test_acc > self.best_avg_acc:
                self.best_avg_acc = avg_test_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model with Avg Accuracy = {avg_test_acc:.4f}")

            # log history
            history["train_loss"].append(train_loss)
            history["train_acc_dr"].append(train_acc_dr)
            history["train_acc_dme"].append(train_acc_dme)
            history["train_f1_dr"].append(train_f1_dr)
            history["train_f1_dme"].append(train_f1_dme)
            history["test_loss"].append(test_loss)
            history["test_acc_dr"].append(test_acc_dr)
            history["test_acc_dme"].append(test_acc_dme)
            history["test_f1_dr"].append(test_f1_dr)
            history["test_f1_dme"].append(test_f1_dme)

            # logging
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"DR Acc: {train_acc_dr*100:.2f}% | DME Acc: {train_acc_dme*100:.2f}% | "
                f"DR F1: {train_f1_dr*100:.3f}% | DME F1: {train_f1_dme*100:.3f}%"
            )
            print(
                f"Test Loss: {test_loss:.4f} | "
                f"DR Acc: {test_acc_dr*100:.2f}% | DME Acc: {test_acc_dme*100:.2f}% | "
                f"DR F1: {test_f1_dr*100:.3f}% | DME F1: {test_f1_dme*100:.3f}%"
            )
            print(f"Learning Rate: {current_lr:.8f}")

        return history
