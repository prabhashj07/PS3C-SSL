#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
import argparse

from src.dataset import *
from src.model import *
from src.train import train_simclr
from linear import LinearClassifier, fine_tune

##############################################################################
# Main
##############################################################################
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train SimCLR and classify Pap smear images.')
    parser.add_argument('--dataset_root', type=str, default='data',
                        help='Root directory containing training datasets')
    parser.add_argument('--test_dir', type=str, default='data',
                        help='Directory containing test images')
    parser.add_argument('--pretrain_batch_size', type=int, default=32,
                        help='Batch size for self-supervised pretraining')
    parser.add_argument('--ft_batch_size', type=int, default=64,
                        help='Batch size for fine-tuning')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='Number of pretraining epochs')
    parser.add_argument('--ft_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4,
                        help='Learning rate for pretraining')
    parser.add_argument('--ft_lr', type=float, default=1e-3,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--out_dim', type=int, default=128,
                        help='Output dimension for SimCLR projection head')
    args = parser.parse_args()

    # Print arguments
    print("\n=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=====================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Create datasets and dataloaders
    unlabeled_dataset = UnlabeledPapSmearDataset(
        root_dir=args.dataset_root, 
        transform=simclr_transform
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        batch_size=args.pretrain_batch_size,
        shuffle=True, 
        num_workers=4
    )

    labeled_dataset = LabeledPapSmearDataset(
        root_dir=args.dataset_root, 
        transform=classification_transform
    )
    sample_weights = compute_class_weights(labeled_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.ft_batch_size,
        sampler=sampler,
        num_workers=4
    )

    test_dataset = PapSmearTestDataset(
        root_dir=args.test_dir, 
        transform=classification_transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size,
        shuffle=False, 
        num_workers=4
    )

    # 2) Initialize models
    base_resnet = models.resnet50(weights=None)
    simclr_model = SimCLR(
        base_model=base_resnet, 
        out_dim=args.out_dim
    ).to(device)
    
    # 3) Pretraining
    optimizer_pretrain = optim.Adam(
        simclr_model.parameters(), 
        lr=args.pretrain_lr
    )
    print("=== SimCLR Pretraining ===")
    train_simclr(
        simclr_model, 
        unlabeled_loader, 
        optimizer_pretrain, 
        device, 
        epochs=args.pretrain_epochs
    )

    # 4) Fine-tuning
    classifier = LinearClassifier(
        in_feats=base_resnet.fc.in_features,
        num_classes=3
    ).to(device)
    optimizer_ft = optim.Adam(
        classifier.parameters(), 
        lr=args.ft_lr
    )
    print("\n=== Fine-Tuning ===")
    fine_tune(
        simclr_model, 
        classifier, 
        labeled_loader, 
        optimizer_ft, 
        nn.CrossEntropyLoss(), 
        device, 
        epochs=args.ft_epochs
    )

    # 5) Inference and visualization
    simclr_model.eval()
    classifier.eval()
    
    predictions = []
    images_for_plot = []
    filenames = []
    
    with torch.no_grad():
        for img_tensor, fname in test_loader:
            feats = simclr_model.encoder(img_tensor.to(device))
            pred_class = labeled_dataset.class_names[classifier(feats).argmax().item()]
            
            predictions.append(pred_class)
            filenames.extend(fname)
            images_for_plot.append(img_tensor[0].cpu())

    # Visualization
    num_to_plot = min(8, len(images_for_plot))
    plt.figure(figsize=(12, 6))
    for i in range(num_to_plot):
        plt.subplot(2, 4, i+1)
        img = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )(images_for_plot[i]).permute(1, 2, 0).clamp(0, 1).numpy()
        
        plt.imshow(img)
        plt.title(f"{filenames[i]}\nPred: {predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()