#!/usr/bin/env python3

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from .lossfuns import SimCLRLoss  

def train_simclr(model, dataloader, optimizer, device, epochs=100):
    """
    TQDM-based loop for self-supervised pre-training using SimCLR.
    
    Args:
        model (nn.Module): The encoder (or full SimCLR model) to be pre-trained.
        dataloader (DataLoader): DataLoader providing batches of augmented image pairs (x_i, x_j).
        optimizer (Optimizer): Optimizer to update the model parameters.
        device (torch.device): Device on which to run training (e.g. CPU or GPU).
        epochs (int): Number of training epochs.
    """
    model.to(device)
    model.train()

    # Instantiate the SimCLR loss function (NT-Xent Loss) with a temperature parameter tau.
    criterion = SimCLRLoss(tau=0.07)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_samples = 0

        # Create a tqdm progress bar for the current epoch.
        pbar = tqdm(dataloader, desc=f"[Pretrain] Epoch {epoch}/{epochs}")
        for x_i, x_j in pbar:
            # Move both augmented views to the device.
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()

            # Compute representations for both augmented views.
            z_i = model(x_i)
            z_j = model(x_j)

            # Compute the NT-Xent loss using the instantiated criterion.
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            bs = x_i.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

            # Update the progress bar with the current batch loss.
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / total_samples
        print(f"  -> Average Loss: {avg_loss:.4f}")