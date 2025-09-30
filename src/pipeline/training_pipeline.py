import torch
from typing import Any, Dict
from src.config.constants import *
from src.models.eyenet import EyeNet
from src.data.data_loader import Loader
from src.training.trainer import Trainer
from src.data.custom_data import EyeData
from src.loss.multitask_loss import MultiTaskLoss
from torch.utils.data import DataLoader, ConcatDataset
from src.preprocessing.fundus_preprocessing import Preprocessing


class Pipeline:
    """
    A training pipeline for Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) classification.

    Steps:
        1. Load and preprocess data
        2. Create training and testing dataloaders
        3. Build the EyeNet model with frozen backbone
        4. Set up optimizer and loss function
        5. Train and evaluate using the Trainer class
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the pipeline.

        Args:
            config (dict): Dictionary containing model, optimizer, and training parameters.
        """
        self.config = config
        self.device = DEVICE
        print(f"Using device: {self.device}")

    def load_data(self, image_path: str, label_path: str) -> Any:
        """
        Load dataset using a custom Loader.

        Args:
            image_path (str): Path to images.
            label_path (str): Path to labels.

        Returns:
            Any: Loaded dataset.
        """
        loader = Loader(image_path, label_path)
        return loader.load_data()

    def preprocess_data(self, data: Any) -> Any:
        """
        Preprocess the data (normalization, resizing, augmentations, etc.).

        Args:
            data (Any): Raw dataset.

        Returns:
            Any: Preprocessed dataset.
        """
        return Preprocessing.process_dataset(data=data)

    def create_dataloader(self, data: Any, is_train: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader with optional augmentation and combination.

        Args:
            data (Any): Dataset to wrap with DataLoader.
            is_train (bool): Whether it's for training (shuffle + augment).

        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        batch_size = self.config["training_params"]["batch_size"]

        if is_train and self.config["data_params"]["augment_and_combine"]:
            original_dataset = EyeData(data, is_train=False)
            augmented_dataset = EyeData(data, is_train=True)
            combined_dataset = ConcatDataset([original_dataset, augmented_dataset])
            return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        dataset = EyeData(data, is_train=is_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    def build_model(self) -> torch.nn.Module:
        """
        Build EyeNet model and freeze its backbone initially.

        Returns:
            nn.Module: EyeNet model with frozen backbone.
        """
        model_params = self.config["model_params"]
        model = EyeNet(
            hidden_units=model_params["hidden_units"],
            dropout_p=model_params["dropout_p"],
        ).to(self.device)

        for param in model.backbone.parameters():
            param.requires_grad = False

        return model

    def setup_training_components(
        self, model: torch.nn.Module
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        """
        Set up loss function and optimizer with layer-wise learning rates.

        Args:
            model (nn.Module): The EyeNet model.

        Returns:
            tuple: (loss function, optimizer)
        """
        loss_fn = MultiTaskLoss().to(self.device)

        optim_params = self.config["optimizer_params"]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.backbone.parameters(),
                    "lr": optim_params["lr_backbone"],
                    "weight_decay": optim_params["wd_backbone"],
                },
                {
                    "params": model.shared_fc.parameters(),
                    "lr": optim_params["lr_shared"],
                    "weight_decay": optim_params["wd_shared"],
                },
                {
                    "params": model.dr_classifier_head.parameters(),
                    "lr": optim_params["lr_head_dr"],
                    "weight_decay": optim_params["wd_head_dr"],
                },
                {
                    "params": model.dme_classifier_head.parameters(),
                    "lr": optim_params["lr_head_dme"],
                    "weight_decay": optim_params["wd_head_dme"],
                },
            ]
        )

        return loss_fn, optimizer

    def run(self) -> Dict[str, list]:
        """
        Execute the full training pipeline.

        Returns:
            dict: Training history including loss, accuracy, and F1 scores.
        """
        print("Step 1: Loading data...")
        train_data = self.load_data(TRAIN_IMAGES, LABEL_TRAIN)
        test_data = self.load_data(TEST_IMAGES, LABEL_TEST)
        print("✅ Done.\n")

        print("Step 2: Preprocessing data...")
        preprocessed_train_data = self.preprocess_data(data=train_data)
        preprocessed_test_data = self.preprocess_data(data=test_data)
        print("✅ Done.\n")

        print("Step 3: Creating data loaders...")
        train_dataloader = self.create_dataloader(
            data=preprocessed_train_data, is_train=True
        )
        test_dataloader = self.create_dataloader(
            data=preprocessed_test_data, is_train=False
        )
        print("✅ Done.\n")

        print("Step 4: Building model...")
        model = self.build_model()
        print("✅ Done.\n")

        print("Step 5: Setting up loss function and optimizer...")
        loss_fn, optimizer = self.setup_training_components(model)
        print("✅ Done.\n")

        print("Step 6: Starting training...")
        training_params = self.config["training_params"]
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=training_params["epochs"],
            device=self.device,
            checkpoint_path=CHECKPOINT_PATH,
            history_path=HISTORY_PLOT_PATH,
            unfreeze=training_params["unfreeze_backbone"],
            unfreeze_step=training_params["unfreeze_step"],
        )

        history = trainer.full_train()
        trainer.plot_history(history)
        print("✅ Training complete!")

        return history
