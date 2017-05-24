import torch
from torch import nn
from torchvision import models
from constant import *
from torchvision.models.inception import InceptionAux


def get_res152(pretrained=False, snapshot_path=None):
    net = models.resnet152()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_res152_path))
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if snapshot_path:
        net.load_state_dict(snapshot_path)
    return net


def get_inception_v3(pretrained=False, snapshot_path=None):
    net = models.inception_v3()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_inception_v3_path))
    net.AuxLogits = InceptionAux(768, num_classes)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if snapshot_path:
        net.load_state_dict(snapshot_path)
    return net


def get_vgg19(pretrained=False, snapshot_path=None):
    net = models.vgg19()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_vgg19_path))
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    if snapshot_path:
        net.load_state_dict(snapshot_path)
    return net


def get_dense201(pretrained=False, snapshot_path=None):
    net = models.densenet201()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_dense201_path))
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    if snapshot_path:
        net.load_state_dict(snapshot_path)
    return net
