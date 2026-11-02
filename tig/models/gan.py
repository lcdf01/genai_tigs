"""Conditional DCGAN generators (from experiments/*_gan/conditional_gan/cdcgan/)."""

import torch
import torch.nn as nn


class MnistCDCGANGenerator(nn.Module):
    def __init__(self, nz=100, n_classes=10):
        super().__init__()
        self.nz = nz
        self.label_emb = nn.Embedding(n_classes, 10)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + 10, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z, labels):
        x = torch.cat([z.view(z.size(0), -1), self.label_emb(labels)], 1)
        return self.main(x.view(-1, self.nz + 10, 1, 1))
