#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset import *
from src.model import *
from src.train import train_simclr
from linear import  LinearClassifier, fine_tune

##############################################################################
# Main
##############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Unlabeled dataset for self-supervised pretraining
    # -------------------------------------------------------------------------
    # dataset_root: Directory containing subfolders: healthy, unhealthy, rubbish, bothcells
    # test_dir: Directory containing test images (e.g., .png files)
    # -------------------------------------------------------------------------
    dataset_root = "data"  
    test_dir = "data"             

    # Create the Unlabeled Dataset for self-supervised pretraining
    unlabeled_dataset = UnlabeledPapSmearDataset(root_dir=dataset_root, transform=simclr_transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Create the Labeled Dataset for fine-tuning (classification)
    labeled_dataset = LabeledPapSmearDataset(root_dir=dataset_root, transform=classification_transform)

    # Compute sample weights for class balancing
    sample_weights = compute_class_weights(labeled_dataset)

    # Create sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Use sampler in DataLoader
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, sampler=sampler, num_workers=4)

    print(f"Labeled dataset size: {len(labeled_dataset)}")

    # Create the Test Dataset for inference
    test_dataset = PapSmearTestDataset(root_dir=test_dir, transform=classification_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Print out dataset sizes to verify
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    print(f"Labeled dataset size: {len(labeled_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # -------------------------------------------------------------------------
    # Example Iteration through each dataloader
    # -------------------------------------------------------------------------

    # Iterate through one batch of the unlabeled dataset
    print("\nIterating through a batch of Unlabeled Dataset (for SimCLR pretraining):")
    for batch in unlabeled_loader:
        x_i, x_j = batch
        print(f"Unlabeled batch shapes: x_i: {x_i.shape}, x_j: {x_j.shape}")
        break

    # Iterate through one batch of the labeled dataset
    print("\nIterating through a batch of Labeled Dataset (for classification):")
    for batch in labeled_loader:
        images, labels = batch
        print(f"Labeled batch shapes: images: {images.shape}, labels: {labels.shape}")
        break

    # Iterate through one batch of the test dataset
    print("\nIterating through a batch of Test Dataset (for inference):")
    for batch in test_loader:
        images, filenames = batch
        print(f"Test batch shapes: images: {images.shape}")
        print(f"Filenames: {filenames}")
        break

    # 2) Build SimCLR model
    base_resnet = models.resnet50(weights=None)
    in_feats = base_resnet.fc.in_features  # Store in_features of FC layer

    # Create SimCLR model
    simclr_model = SimCLR(base_model=base_resnet, out_dim=128).to(device)

    optimizer_pretrain = optim.Adam(simclr_model.parameters(), lr=1e-3)

    # 3) Self-Supervised Pretraining
    print("=== SimCLR Pretraining ===")
    train_simclr(simclr_model, unlabeled_loader, optimizer_pretrain, device, epochs=1)

    # 4) Fine-tune on labeled data
    for param in simclr_model.parameters():
        param.requires_grad = False  # Freeze SimCLR model

    labeled_ds = LabeledPapSmearDataset(dataset_root, transform=classification_transform)
    labeled_loader = DataLoader(labeled_ds, batch_size=64, shuffle=True, num_workers=7)

    # Build linear classifier
    classifier = LinearClassifier(in_feats, num_classes=3).to(device)
    optimizer_ft = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n=== Fine-Tuning on Labeled Data ===")
    fine_tune(simclr_model, classifier, labeled_loader, optimizer_ft, criterion, device, epochs=5)

    # 5) Inference on test images
    test_ds = PapSmearTestDataset(dataset_root, transform=classification_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    simclr_model.eval()
    classifier.eval()

    print("\n=== Inference + Plotting ===")
    predictions = []
    images_for_plot = []
    filenames = []

    with torch.no_grad():
        for img_tensor, fname in test_loader:
            img_tensor = img_tensor.to(device)
            feats = simclr_model.encoder(img_tensor)
            logits = classifier(feats)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = labeled_ds.class_names[pred_idx]

            predictions.append(pred_class)
            filenames.append(fname)

            images_for_plot.append(img_tensor[0].cpu())  # Move image back to CPU for plotting

    # 6) Plot some results
    num_to_plot = min(len(images_for_plot), 8)
    plt.figure(figsize=(12, 6))
    for i in range(num_to_plot):
        plt.subplot(2, 4, i+1)
        inv = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_inv = inv(images_for_plot[i])
        img_np = img_inv.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title(f"{filenames[i][0]} => {predictions[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("Predictions on test images:")
    for f, p in zip(filenames, predictions):
        print(f"{f[0]} => {p}")


if __name__ == "__main__":
    main()
