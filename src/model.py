from torch import nn
from torchvision.models import resnet18

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
