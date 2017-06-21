import functools
import math

import torch
from torch import nn
from torchvision import models

from configuration import *


def _weights_init(model, pretrained):
    for m in model.modules():
        if pretrained:
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        else:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


def _make_model(get_model):
    @functools.wraps(get_model)
    def wrapper_get_model(num_classes, pretrained=False, snapshot_path=None):
        net = get_model(num_classes, pretrained)
        net = _MultiLabelNet(net, num_classes)
        if snapshot_path:
            net.load_state_dict(torch.load(snapshot_path))
            print 'load snapshot %s' % snapshot_path
        else:
            _weights_init(net, pretrained)
        return net

    return wrapper_get_model


@_make_model
def get_res152(num_classes, pretrained):
    net = models.resnet152()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_res152_path))
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


@_make_model
def get_inception_v3(num_classes, pretrained):
    net = models.inception_v3()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_inception_v3_path))
    net.AuxLogits.fc = nn.Linear(net.AuxLogits.fc.in_features, num_classes)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


@_make_model
def get_vgg19(num_classes, pretrained):
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
    return net


@_make_model
def get_dense201(num_classes, pretrained):
    net = models.densenet201()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_dense201_path))
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net


class _MultiLabelNet(nn.Module):
    def __init__(self, base_net, num_classes):
        super(_MultiLabelNet, self).__init__()
        self.base_net = base_net
        self.cross_label = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        aux = self.base_net(x)
        if len(aux) == 2:
            aux1, aux2 = aux
            x = self.cross_label(aux1)
            return aux1, aux2, x
        x = self.cross_label(aux)
        if self.training:
            return aux, x
        else:
            return x
