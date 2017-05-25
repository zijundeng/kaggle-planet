import torch
from torch import nn
from torchvision import models
from constant import *
from torchvision.models.inception import InceptionAux
import math


def get_res152(pretrained=False, snapshot_path=None):
    net = models.resnet152()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_res152_path))
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if snapshot_path:
        net.load_state_dict(torch.load(snapshot_path))
    return net


def get_inception_v3(pretrained=False, snapshot_path=None):
    net = models.inception_v3()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_inception_v3_path))
    net.AuxLogits = InceptionAux(768, num_classes)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if snapshot_path:
        net.load_state_dict(torch.load(snapshot_path))
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
    for m in net.classifier.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    if snapshot_path:
        net.load_state_dict(torch.load(snapshot_path))
    return net


def get_dense201(pretrained=False, snapshot_path=None):
    net = models.densenet201()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_dense201_path))
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    if snapshot_path:
        net.load_state_dict(torch.load(snapshot_path))
    return net


def get_squeeze(pretrained=False, snapshot_path=None):
    net = models.squeezenet1_1()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_dense201_path))

    final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        final_conv,
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13)
    )
    for m in net.classifier.modules():
        if isinstance(m, nn.Conv2d):
            gain = 2.0
            if m is final_conv:
                m.weight.data.normal_(0, 0.01)
            else:
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                u = math.sqrt(3.0 * gain / fan_in)
                m.weight.data.uniform_(-u, u)
            if m.bias is not None:
                m.bias.data.zero_()

    net.num_classes = num_classes

    if snapshot_path:
        net.load_state_dict(torch.load(snapshot_path))
    return net

