import dataclasses
from loguru import logger
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch_geometric.loader import DataLoader
import mlflow

from project.config import MODEL_SIGNATURE, TrainingConfig
from project.dataset import MoleculeDataset
from project.model import GNN

# Specify tracking server
mlflow.set_tracking_uri("http://localhost:5000")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def count_parameters(model: GNN):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_metrics(y_pred: np.ndarray[int], y_true: np.ndarray[int], epoch: int, mode: str) -> None:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    mlflow.log_metric(key=f"Precision-{mode}", value=float(precision), step=epoch)
    mlflow.log_metric(key=f"Recall-{mode}", value=float(recall), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        mlflow.log_metric(key=f"ROC-AUC-{mode}", value=float(roc), step=epoch)
    except:
        logger.debug(f"ROC AUC: not defined")
        mlflow.log_metric(key=f"ROC-AUC-{mode}", value=float(0.0), step=epoch)


def train_epoch(
    model: GNN,
    epoch: int,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: BCEWithLogitsLoss,
) -> float:
    # Enumerate over the data
    all_preds = []
    all_labels = []

    step = 0
    running_loss = 0.0
    for _, batch in enumerate(tqdm(data_loader)):
        # Use GPU
        batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass on the model
        pred = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)

        # Calculate the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Update tracking
        step += 1
        running_loss += loss.item()

        label_raw = torch.sigmoid(pred).cpu().detach().numpy()
        all_preds.append(np.rint(label_raw))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    log_metrics(all_preds, all_labels, epoch, "Train")

    return running_loss / step


def test_epoch(
    model: GNN, epoch: int, data_loader: DataLoader, loss_fn: BCEWithLogitsLoss
) -> float:
    # Enumerate over the data
    all_preds = []
    all_labels = []

    step = 0
    running_loss = 0.0
    for _, batch in enumerate(tqdm(data_loader)):
        # Use GPU
        batch.to(device)

        # Forward pass on the model
        pred = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)

        # Calculate the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()

        # Update tracking
        step += 1
        running_loss += loss.item()

        label_raw = torch.sigmoid(pred).cpu().detach().numpy()
        all_preds.append(np.rint(label_raw))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    log_metrics(all_preds, all_labels, epoch, "Test")

    return running_loss / step


def train_model(config: TrainingConfig, n_epochs: int = 300):
    with mlflow.start_run():
        for field in dataclasses.fields(config):
            mlflow.log_param(field.name, getattr(config, field.name))

        # Loading the dataset
        logger.info("Loading the dataset...")
        train_dataset = MoleculeDataset(root="data", filename="HIV_train_oversampled.csv")
        test_dataset = MoleculeDataset(root="data", filename="HIV_test.csv", test=True)

        sample_data = train_dataset[0]
        (feature_dim, edge_dim) = (sample_data.x.shape[1], sample_data.edge_attr.shape[1])

        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        # Instantiate the model
        model = GNN(feature_size=feature_dim, edge_size=edge_dim, config=config)
        model = model.to(device)

        n_params = count_parameters(model)
        mlflow.log_param("n_params", n_params)
        logger.info(f"Constructed model with {n_params} parameters")

        # Loss, optimizer, and scheduler
        pos_weight = torch.tensor([config.pos_weight], dtype=torch.float32).to(device)
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = SGD(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
        )
        scheduler = ExponentialLR(optimizer=optimizer, gamma=config.scheduler_gamma)

        # Start training
        best_test_loss = np.inf
        early_stopping_counter = 0
        for epoch in range(n_epochs):
            if early_stopping_counter <= 10:
                # Training
                model.train()
                train_loss = train_epoch(model, epoch, train_loader, optimizer, scheduler, loss_fn)

                logger.info(f"Epoch {epoch} | Train Loss {train_loss}")
                mlflow.log_metric(key="Train Loss", value=float(train_loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:
                    test_loss = test_epoch(model, epoch, test_loader, loss_fn)

                    logger.info(f"Epoch {epoch} | Test Loss {test_loss}")
                    mlflow.log_metric(key="Test Loss", value=float(test_loss), step=epoch)

                    if float(test_loss) < best_test_loss:
                        best_test_loss = test_loss
                        # Save the currently best model
                        mlflow.pytorch.log_model(model, "model", signature=MODEL_SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
            else:
                logger.info("Early stopping due to no improvement")
                break

        logger.info(f"Finishing training with best test loss: {best_test_loss}")
        return [best_test_loss]
