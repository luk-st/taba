import torch.nn as nn
from torch import optim
from torchvision import models


def get_noise_model(device="cuda", lr=0.01, out_features=2):
    modified_EfficientNetV2L = models.efficientnet_v2_s()
    modified_EfficientNetV2L.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=128, out_features=out_features),
    )
    model_to_train = modified_EfficientNetV2L.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_to_train.parameters(), lr=lr, weight_decay=0.01)
    return model_to_train, criterion, optimizer
