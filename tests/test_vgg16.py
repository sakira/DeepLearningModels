import pytest

import torch

from vgg16.vgg16 import VGG16

device = 'cpu' if torch.cuda.is_available() else 'cpu'

# @pytest
def test_model():
    vgg16_model = VGG16().to(device)
    print(vgg16_model)
    torch.manual_seed(0)
    X = torch.rand(1, 3, 224, 224).to(device)
    pred = vgg16_model(X)
    print(pred)

    assert True