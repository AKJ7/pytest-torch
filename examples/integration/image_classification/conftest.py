"""Shared fixtures: a trained model, the deployment pipeline and a validation set."""
import pytest
import torch
from torch.nn import functional as F

from .model import MEAN, STD, InferencePipeline, ResNet, make_images


@pytest.fixture(scope="session")
def trained_resnet():
    torch.manual_seed(0)
    model = ResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    images, labels = make_images(256, seed=0)
    x = (images.permute(0, 3, 1, 2).float() / 255 - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    for _ in range(4):
        for i in range(0, 256, 32):
            loss = F.cross_entropy(model(x[i : i + 32]), labels[i : i + 32])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


@pytest.fixture
def pipeline(trained_resnet, torch_device):
    return InferencePipeline(trained_resnet).to(torch_device).eval()


@pytest.fixture(scope="session")
def val_loader():
    images, labels = make_images(160, seed=1)
    return [(images[i : i + 40], labels[i : i + 40]) for i in range(0, 160, 40)]
