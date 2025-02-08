#!/usr/bin/env python3

import os
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler


##############################################################################
# White-Padding Utility
##############################################################################
def white_pad(pil_img, desired_size=224):
    """
    Pads a PIL image to (desired_size x desired_size) with a white background.
    For minimal GPU resources, you could keep images smaller (e.g. 128 x 128).
    """
    old_size = pil_img.size  # (width, height)
    ratio = float(desired_size) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]

    pil_resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (desired_size, desired_size), (255, 255, 255))
    paste_pos = (
        (desired_size - new_size[0]) // 2,
        (desired_size - new_size[1]) // 2
    )
    new_img.paste(pil_resized, paste_pos)
    return new_img


##############################################################################
# Unlabeled Dataset for SimCLR
##############################################################################
class UnlabeledPapSmearDataset(Dataset):
    """
    Collects all images from subfolders (healthy, unhealthy, rubbish, bothcells)
    ignoring labels. For self-supervised pretraining.
    """
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        for subdir in os.listdir(root_dir):
            subpath = os.path.join(root_dir, subdir)
            if os.path.isdir(subpath):
                for f in glob(os.path.join(subpath, "*")):
                    if os.path.isfile(f):
                        self.image_paths.append(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = white_pad(pil_img, 224)

        # Two augmented views per sample (SimCLR)
        x_i = self.transform(pil_img)
        x_j = self.transform(pil_img)
        return x_i, x_j


##############################################################################
# Labeled Dataset (bothcells -> unhealthy)
##############################################################################
class LabeledPapSmearDataset(Dataset):
    """
    Merges 'bothcells' -> 'unhealthy', resulting in 3 classes: healthy, rubbish,
    unhealthy.
    """
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.folder_to_class = {
            'healthy': 'healthy',
            'rubbish': 'rubbish',
            'unhealthy': 'unhealthy',
            'bothcells': 'unhealthy'
        }

        self.samples = []
        for subdir in os.listdir(root_dir):
            subpath = os.path.join(root_dir, subdir)
            if os.path.isdir(subpath) and (subdir in self.folder_to_class):
                mapped_class = self.folder_to_class[subdir]
                for f in glob(os.path.join(subpath, "*")):
                    if os.path.isfile(f):
                        self.samples.append((f, mapped_class))
        
        self.class_names = sorted(list(set(self.folder_to_class.values())))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_str = self.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = white_pad(pil_img, 224)

        if self.transform:
            pil_img = self.transform(pil_img)

        label = self.class_to_idx[cls_str]
        return pil_img, label


##############################################################################
# Test Dataset
##############################################################################
class PapSmearTestDataset(Dataset):
    """
    For unlabeled .png in root_dir. We do inference, then display predicted labels.
    """
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        for fname in os.listdir(root_dir):
            full_path = os.path.join(root_dir, fname)
            if os.path.isfile(full_path) and fname.lower().endswith('.png'):
                self.image_paths.append(full_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_img = Image.open(path).convert("RGB")
        pil_img = white_pad(pil_img, 224)

        if self.transform:
            pil_img = self.transform(pil_img)
        
        return pil_img, os.path.basename(path)


##############################################################################
# Transforms
##############################################################################
# For self-supervised pretraining (SimCLR's data augment)
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# For classification + test inference
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


##############################################################################
# Class-Weighted Sampler
##############################################################################
def compute_class_weights(dataset):
    """
    Computes class weights for imbalanced datasets.
    Returns sample weights for use in WeightedRandomSampler.
    """
    class_counts = {cls: 0 for cls in dataset.class_to_idx.keys()}
    
    # Count occurrences of each class
    for _, label in dataset.samples:
        class_counts[label] += 1

    # Compute weights (inverse of frequency)
    total_samples = len(dataset.samples)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign weights to each sample
    sample_weights = [class_weights[label] for _, label in dataset.samples]

    return sample_weights


##############################################################################
# Dataloader Setup 
##############################################################################
def get_dataloaders(dataset_root, test_dir, batch_size=32, num_workers=4):
    """
    Returns DataLoaders for self-supervised (unlabeled), classification (labeled),
    and test datasets.
    """
    dataset_root = "../data/"  
    test_dir = "../data/"

    # Create datasets
    unlabeled_dataset = UnlabeledPapSmearDataset(root_dir=dataset_root, transform=simclr_transform)
    labeled_dataset = LabeledPapSmearDataset(root_dir=dataset_root, transform=classification_transform)
    test_dataset = PapSmearTestDataset(root_dir=test_dir, transform=classification_transform)

    # Compute sample weights for balanced sampling
    sample_weights = compute_class_weights(labeled_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create dataloaders
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return unlabeled_loader, labeled_loader, test_loader