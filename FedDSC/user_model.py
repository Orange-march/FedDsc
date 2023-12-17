import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as models
import my_resnet

class LW_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(LW_ResNet, self).__init__()
        self.in_planes = 8
        self.rp = 0

        #-----------[7, 7]-------------#
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        #---------[3, 3] * 2-----------#
        # self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1)
        # self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.cl = nn.Conv2d(8, 8, kernel_size=1, bias=False)       #compatibility layer
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # self.semi_f = out
        # out = self.cl(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.rp = out
        out = self.linear(out)
        return out



def ResNet10():
    return LW_ResNet(my_resnet.BasicBlock, [1,1,1,1])

def ResNet33():
    return LW_ResNet(my_resnet.BasicBlock, [2,4,6,3])
