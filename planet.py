import torch
from torch import nn
from torchvision import models


class Planet(nn.Module):
    def __init__(self, num_classes=17, base_net_pretrained=False, base_net_pretrained_path=None, use_sigmoid=False):
        super(Planet, self).__init__()
        self.base_net = models.resnet152()
        if base_net_pretrained:
            if base_net_pretrained_path is None:
                raise RuntimeError('path of pretrained base net should be specified')
            self.base_net.load_state_dict(torch.load(base_net_pretrained_path))
        self.base_net.fc = nn.Linear(self.base_net.fc.in_features, num_classes)
        self.fc2 = nn.Linear(num_classes, num_classes)
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.base_net(x)
        x2 = self.fc2(self.dropout(self.relu(x1)))
        if self.use_sigmoid:
            return self.sigmoid(x1), self.sigmoid(x2)
        else:
            return x1, x2

    def open_sigmoid(self):
        self.use_sigmoid = True
