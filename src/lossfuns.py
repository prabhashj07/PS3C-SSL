#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################
# SimCLR Loss 
##############################################################################
"""
SimCLR Loss is used in the SimCLR framework for contrastive learning.
It brings representations of two augmented views of the same image closer
while pushing apart representations of different images.
"""
class SimCLRLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(SimCLRLoss, self).__init__()
        self.tau = tau  # Temperature parameter for contrastive loss
    
    def forward(self, z1, z2):
        # z1 and z2 are the feature representations of two augmented views of the same image
        B = z1.shape[0]  # Batch size
        out = torch.cat([z1, z2], dim=0)  # Concatenate both views into a single tensor [2*B, D]
        
        # Compute similarity matrix using dot product and temperature scaling
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.tau)  # [2*B, 2*B]
        
        # Create a mask to exclude self-similarity (diagonal elements)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * B, device=sim_matrix.device)).bool()  # [2*B, 2*B]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1)  # Remove diagonal and reshape [2*B, 2*B-1]
        
        # Compute positive similarity (between corresponding pairs of z1 and z2)
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.tau)  # [B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # Duplicate for both views [2*B]
        
        # Compute the contrastive loss as the negative log-likelihood
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


##############################################################################
# DCLW Loss 
##############################################################################
"""DCLWLoss (Dynamic Contrastive Learning Weighting Loss) applies a weighting
function to the positive pairs. It adjusts the contribution of each positive pair
based on the similarity scores, while also computing a contrastive loss for negatives."""
class DCLWLoss(nn.Module):
    def __init__(self, tau=0.1, sigma=0.5):
        super(DCLWLoss, self).__init__()
        self.tau = tau  # Temperature parameter
        self.sigma = sigma  # Scaling parameter for weighting
        self.SMALL_NUM = np.log(1e-45)  # Small constant for numerical stability
        # Weighting function for positive pairs:
        # It adjusts the weight based on the similarity between each pair.
        self.weight_fn = lambda z1, z2: 2 - z1.size(0) * F.softmax((z1 * z2).sum(dim=1) / self.sigma, dim=0).squeeze()
        
    def forward(self, z1, z2):
        # Compute cross-view similarity matrix (comparing each element in z1 with each in z2)
        cross_view_distance = torch.mm(z1, z2.t())  # [B, B]
        
        # Positive loss: similarity between corresponding pairs (diagonal elements)
        positive_loss = -torch.diag(cross_view_distance) / self.tau  # [B]
        positive_loss = positive_loss * self.weight_fn(z1, z2)  # Apply weighting
        
        # Negative loss: combines similarities from both within-view and cross-view comparisons
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.tau  # [B, 2*B]
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)  # Mask to exclude self-similarity
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * self.SMALL_NUM, dim=1, keepdim=False)  # [B]
        
        # Total loss: average of the weighted positive and negative losses
        return (positive_loss + negative_loss).mean()


##############################################################################
# VICReg Loss 
##############################################################################
"""VICReg (Variance-Invariance-Covariance Regularization) Loss simultaneously
enforces invariance between views (via MSE loss), maintains a certain level of variance,
and minimizes redundancy between different dimensions (covariance loss)."""
class VICRegLoss(nn.Module):
    def __init__(self, l=25, mu=25, nu=1):
        super(VICRegLoss, self).__init__()
        self.l = l  # Weight for invariance (similarity) loss
        self.mu = mu  # Weight for variance loss
        self.nu = nu  # Weight for covariance loss
        self.sim_loss = nn.MSELoss()  # Mean squared error for invariance loss

    def off_diagonal(self, x):
        # Return off-diagonal elements of a square matrix (used for covariance loss)
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def std_loss(self, z_a, z_b):
        # Variance loss: encourages standard deviation of embeddings to be close to 1
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)  # Standard deviation of z_a
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)  # Standard deviation of z_b
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))  # Hinge loss to push std above 1
        return std_loss
    
    def cov_loss(self, z_a, z_b):
        # Covariance loss: encourages off-diagonal elements of the covariance matrix to be close to zero
        N = z_a.shape[0]  # Batch size
        D = z_a.shape[1]  # Feature dimension
        z_a = z_a - z_a.mean(dim=0)  # Center z_a
        z_b = z_b - z_b.mean(dim=0)  # Center z_b
        cov_z_a = (z_a.T @ z_a) / (N - 1)  # Covariance matrix of z_a
        cov_z_b = (z_b.T @ z_b) / (N - 1)  # Covariance matrix of z_b
        cov_loss = self.off_diagonal(cov_z_a).pow_(2).sum() / D + self.off_diagonal(cov_z_b).pow_(2).sum() / D
        return cov_loss

    def forward(self, z1, z2):
        # Compute the three components of the VICReg loss
        _sim_loss = self.sim_loss(z1, z2)  # Invariance loss: MSE between the two views
        _std_loss = self.std_loss(z1, z2)   # Variance loss: keeps feature variance above a threshold
        _cov_loss = self.cov_loss(z1, z2)   # Covariance loss: reduces redundancy in feature dimensions
        # Combine the losses using the specified weights
        loss = self.l * _sim_loss + self.mu * _std_loss + self.nu * _cov_loss
        return loss


##############################################################################
# Barlow Twins Loss 
##############################################################################
"""Barlow Twins Loss is designed to make the cross-correlation matrix between
embeddings from two views as close as possible to the identity matrix.
The diagonal is enforced to be 1 (on-diagonal loss) and the off-diagonals are minimized."""
class BarlowLoss(nn.Module):
    def __init__(self, lambd=0.0051):
        super(BarlowLoss, self).__init__()
        self.lambd = lambd  # Weight for off-diagonal terms

    def off_diagonal(self, x):
        # Return off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # Compute the cross-correlation matrix between the two views
        B = z1.shape[0]  # Batch size
        c = z1.T @ z2  # Cross-correlation matrix [D, D]
        c.div_(B)  # Normalize by batch size
        
        # On-diagonal loss: force the diagonal elements to be close to 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # Off-diagonal loss: force the off-diagonal elements to be close to 0
        off_diag = self.off_diagonal(c).pow_(2).sum()
        # Total loss is a weighted sum of the two components
        loss = on_diag + self.lambd * off_diag
        return loss


##############################################################################
# SimSiam Loss 
##############################################################################
"""SimSiam Loss is used in the SimSiam framework which does not require negative pairs.
It uses a stop-gradient mechanism to avoid collapse by computing the similarity
between one view and a detached version of the other."""
class SimSiamLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SimSiamLoss, self).__init__()
        self.alpha = alpha  # Weight for the first term (z1 predicting z2)
        self.beta = beta    # Weight for the second term (z2 predicting z1)

    def forward(self, z1, z2):
        # Compute similarity between z1 and z2 (with z2 detached to stop gradients)
        sim_1 = -(F.normalize(z1, dim=-1) * F.normalize(z2.detach(), dim=-1)).sum(dim=-1).mean()
        # Compute similarity between z2 and z1 (with z1 detached to stop gradients)
        sim_2 = -(F.normalize(z2, dim=-1) * F.normalize(z1.detach(), dim=-1)).sum(dim=-1).mean()
        # Combine both terms with the respective weights
        loss = self.alpha * sim_1 + self.beta * sim_2
        return loss


##############################################################################
# TiCo Loss 
##############################################################################
"""TiCo Loss introduces a covariance regularization term that is updated using
an exponential moving average. It is designed to balance the similarity between
two views and the alignment with a running estimate of the covariance matrix."""
class TiCoLoss(nn.Module):
    def __init__(self, beta=0.9, rho=8):
        super(TiCoLoss, self).__init__()
        self.beta = beta  # Exponential moving average factor for covariance matrix
        self.rho = rho    # Weight for the covariance regularization term

    def forward(self, C, z1, z2):
        # Normalize embeddings
        z_1 = F.normalize(z1, dim=-1)
        z_2 = F.normalize(z2, dim=-1)
        
        # Update covariance matrix using exponential moving average
        B = torch.mm(z_1.T, z_1) / z_1.shape[0]  # Batch covariance
        C = self.beta * C + (1 - self.beta) * B  # Update C
        
        # Compute TiCo loss:
        # The first term encourages alignment between z1 and z2.
        # The second term regularizes the embeddings with the updated covariance matrix.
        loss = - (z_1 * z_2).sum(dim=1).mean() + self.rho * (torch.mm(z_1, C) * z_1).sum(dim=1).mean()
        return loss, C
