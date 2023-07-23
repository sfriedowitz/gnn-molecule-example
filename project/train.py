import numpy as np
import torch
from tqdm import tqdm

from project.model import GNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model: GNN):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model: GNN, epoch, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Forward pass on the model
        pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        # Calculate the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        # Update tracking
        running_loss += loss.item()
        step += 1

        label_raw = torch.sigmoid(pred).cpu().detach().numpy()
        all_preds.append(np.rint(label_raw))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    return running_loss / step
