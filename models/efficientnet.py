import torchvision.models as models
import torch.nn as nn

def get_efficientnet(num_classes=9):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
