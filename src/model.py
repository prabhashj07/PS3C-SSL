import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import copy

# Reference: https://github.com/soumitri2001/SmallDataSSL/blob/main/model.py

##############################################################################
# SimCLR Model
##############################################################################
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, 'fc'):
            in_feats = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("base_model doesn't have .fc attribute.")

        self.projector = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            nn.BatchNorm1d(in_feats),
            nn.ReLU(),
            nn.Linear(in_feats, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z

##############################################################################
# SimSiam Model
##############################################################################
class SimSiam(nn.Module):
    def __init__(self, base_model, out_dim=512):
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, 'fc'):
            in_feats = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("base_model doesn't have .fc attribute.")

        self.projector = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            nn.BatchNorm1d(in_feats),
            nn.ReLU(),
            nn.Linear(in_feats, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z


##############################################################################
# DCLW Model
##############################################################################
class DCLW(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, 'fc'):
            in_feats = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("base_model doesn't have .fc attribute.")

        self.projector = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            nn.BatchNorm1d(in_feats),
            nn.ReLU(),
            nn.Linear(in_feats, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z


##############################################################################
# VICReg Model
##############################################################################
class VICReg(nn.Module):
    def __init__(self, base_model, out_dim=4096):
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, 'fc'):
            in_feats = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("base_model doesn't have .fc attribute.")

        self.projector = nn.Sequential(
            nn.Linear(in_feats, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z



##############################################################################
# Barlow Model
##############################################################################
class Barlow(nn.Module):
    def __init__(self, base_model, out_dim=4096):
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, 'fc'):
            in_feats = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("base_model doesn't have .fc attribute.")

        self.projector = nn.Sequential(
            nn.Linear(in_feats, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = self.bn(z)
        return z



##############################################################################
# TiCo Model
##############################################################################
class TiCo(nn.Module):
    def __init__(self, base_model, out_dim=4096):
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, 'fc'):
            in_feats = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("base_model doesn't have .fc attribute.")

        self.projection_head = nn.Sequential(
            nn.Linear(in_feats, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.deactivate_requires_grad(self.encoder_momentum)
        self.deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.encoder(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.encoder_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def deactivate_requires_grad(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def update_momentum(self, model, model_ema, m):
        for model_ema, model in zip(model_ema.parameters(), model.parameters()):
            model_ema.data = model_ema.data * m + model.data * (1.0 - m)

    def schedule_momentum(self, iter, max_iter, m=0.99):
        return m + (1 - m)*np.sin((np.pi/2)*iter/(max_iter-1))