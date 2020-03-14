import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return mish(input)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_mish=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.mish = Mish()
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.mish)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.mish)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.mish)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_mish:
            rep = rep[1:]
        else:
            rep[0] = Mish()

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class HourglassNet(nn.Module):
    def __init__(self, depth, channel):
        super(HourglassNet, self).__init__()
        self.depth = depth
        self.hg = []
        for _ in range(self.depth):
            self.hg.append(nn.ModuleList([
                Block(channel,channel,3,1,start_with_mish=True,grow_first=True),
                Block(channel,channel,3,1,start_with_mish=True,grow_first=True),
                Block(channel,channel,3,1,start_with_mish=True,grow_first=True),
                Block(channel,channel,3,1,start_with_mish=True,grow_first=True)
            ]))
        self.hg = nn.ModuleList(self.hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)

        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class XceptionHourglass(nn.Module):
    def __init__(self, output_channel=1):
        super(XceptionHourglass, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.mish = Mish()

        self.conv2 = nn.Conv2d(32, 96, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(96)

        self.block1 = HourglassNet(4, 96)
        self.bn3 = nn.BatchNorm2d(96)
        self.block2 = HourglassNet(4, 96)

        self.sigmoid = nn.Sigmoid()

        self.conv3 = nn.Conv2d(96, output_channel, 5, 1, 2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mish(x)

        out1 = self.block1(x)
        x = self.bn3(out1)
        x = self.mish(x)
        out2 = self.block2(x)

        r = self.sigmoid(out1 + out2)
        r = F.interpolate(r, scale_factor=2)
        r = self.conv3(r)
        r = self.sigmoid(r)
        return r
