#!/usr/bin/env python

"""
linear.py

This script demonstrates linear evaluation for self-supervised pre-training.
It defines a simple linear classifier that is fine-tuned on top of a frozen encoder
from a pre-trained SSL model (e.g., SimCLR). The code also includes an example of how to use the fine-tuning routine.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  

# =============================================================================
# Linear Classifier Definition
# =============================================================================
class LinearClassifier(nn.Module):
    def __init__(self, in_feats, num_classes=3):
        """
        Initialize the Linear Classifier.

        Args:
            in_feats (int): Number of input features (output size from encoder).
            num_classes (int): Number of target classes. Default is 3.
        """
        super().__init__()
        # Define a fully connected layer to map encoder features to class logits.
        self.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        """
        Forward pass for the linear classifier.
        
        Args:
            x (Tensor): Input feature tensor.
        Returns:
            Tensor: Output logits corresponding to each class.
        """
        return self.fc(x)

# =============================================================================
# Fine-tuning Routine for Linear Evaluation
# =============================================================================
def fine_tune(simclr_model, classifier, dataloader, optimizer, criterion, device, epochs=5):
    """
    Fine-tune a linear classifier on top of a frozen encoder from a pre-trained SSL model.
    The simclr_model's encoder is used to extract features and the classifier is trained on labeled data.
    Uses tqdm for a progress bar.

    Args:
        simclr_model (nn.Module): Pre-trained SSL model containing an encoder.
        classifier (nn.Module): Linear classifier to be trained.
        dataloader (DataLoader): DataLoader for labeled data.
        optimizer (Optimizer): Optimizer for training the classifier.
        criterion (Loss): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): Device for computation (CPU or GPU).
        epochs (int): Number of epochs to fine-tune. Default is 5.
    """
    # Freeze the SSL model by setting it to evaluation mode.
    simclr_model.eval()
    classifier.to(device)
    classifier.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_samples = 0
        correct = 0

        # Create a progress bar for the current epoch.
        pbar = tqdm(dataloader, desc=f"[FineTune] Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            # Move images and labels to the specified device.
            imgs, labels = imgs.to(device), labels.to(device)

            # Zero out gradients for the classifier.
            optimizer.zero_grad()

            # Extract features from the frozen encoder.
            with torch.no_grad():
                feats = simclr_model.encoder(imgs)
            
            # Forward pass through the linear classifier.
            logits = classifier(feats)
            
            # Compute the classification loss.
            loss = criterion(logits, labels)
            loss.backward()  # Backpropagate the loss.
            optimizer.step()  # Update classifier parameters.

            # Update running statistics for loss and accuracy.
            bs = labels.size(0)
            running_loss += loss.item() * bs
            total_samples += bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # Update progress bar with current loss.
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute average loss and accuracy for the epoch.
        avg_loss = running_loss / total_samples
        acc = 100.0 * correct / total_samples
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")