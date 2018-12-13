import numpy as np
import torch
import torch.cuda
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


def initilize_modules(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            # m.weight.data.uniform_(-math.sqrt(3. / n), math.sqrt(3. / n))
            # m.bias.data.zero_()
            torch.nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()
            torch.nn.init.constant(m.weight, 1)
            torch.nn.init.constant(m.bias, 0)


def fliplr(tensor):
    inv_idx = Variable(torch.arange(tensor.size(3)-1, -1, -1).long()).cuda()
    # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
    inv_tensor = tensor.index_select(3, inv_idx)
    return inv_tensor.contiguous()


def upsample_nn_nearest(x):
    return F.upsample(x, scale_factor=2, mode='nearest')


def generate_pyramid(image):
    # TODO resize area
    pyramid = [image]
    for i in range(3):
        pyramid.append(F.avg_pool2d(pyramid[i], 2, 2))
    return pyramid


def generate_max_pyramid(image):
    # TODO resize area
    pyramid = [image]
    for i in range(3):
        pyramid.append(F.max_pool2d(pyramid[i], 2, 2))
    return pyramid
