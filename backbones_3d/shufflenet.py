'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x



class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.groups = groups
        mid_planes = out_planes//4
        if self.stride == 2:
            out_planes = out_planes - in_planes
        g = 1 if in_planes==24 else groups
        self.conv1    = nn.Conv3d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1      = nn.BatchNorm3d(mid_planes)
        self.conv2    = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2      = nn.BatchNorm3d(mid_planes)
        self.conv3    = nn.Conv3d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3      = nn.BatchNorm3d(out_planes)
        self.relu     = nn.ReLU(inplace=True)

        if stride == 2:
            self.shortcut = nn.AvgPool3d(kernel_size=(2,3,3), stride=2, padding=(0,1,1))


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        if self.stride == 2:
            a = self.shortcut(x)
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)

        return out


class ShuffleNet(nn.Module):
    def __init__(self,
                 groups,
                 width_mult=1,
                 num_classes=400):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.groups = groups
        num_blocks = [4,8,4]

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(num_groups))
        out_planes = [int(i * width_mult) for i in out_planes]
        self.in_planes = out_planes[0]
        self.conv1   = conv_bn(3, self.in_planes, stride=(1,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(out_planes[1], num_blocks[0], self.groups)
        self.layer2  = self._make_layer(out_planes[2], num_blocks[1], self.groups)
        self.layer3  = self._make_layer(out_planes[3], num_blocks[2], self.groups)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) # Return the features before pooling

        return out

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = ShuffleNet(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model(groups=3, num_classes=600, width_mult=1)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)


