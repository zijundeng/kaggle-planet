from torch import nn
from torchvision import models


class Planet(nn.Module):
    def __init__(self, num_classes=17, base_net=models.resnet152()):
        super(Planet, self).__init__()
        self.base_net = base_net
        self.base_net.fc = nn.Linear(self.base_net.fc.in_features, num_classes)
        self.main_clf = nn.Sequential(nn.ReLU(), nn.Dropout(), nn.Linear(num_classes, num_classes), nn.Sigmoid())

    def forward(self, x):
        x1 = self.base_net(x)
        x2 = self.main_clf(x1)
        return x1, x2
