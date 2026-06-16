import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes=9):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
