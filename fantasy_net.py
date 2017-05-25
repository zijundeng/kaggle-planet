from torchvision import models
from torch import nn
from utils import *


class FantasyNet(nn.Module):
    def __init__(self, res152=get_res152(), dense201=get_dense201(), inception_v3=get_inception_v3(),
                 vgg19=get_vgg19()):
        super(FantasyNet, self).__init__()


    def forward(self, x):
        pass