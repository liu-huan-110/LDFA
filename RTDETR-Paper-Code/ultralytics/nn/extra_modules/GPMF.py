import torch
import torch.nn as nn
from ..modules.conv import Conv, ConvTranspose

__all__ = ['PMC', 'CPMC', 'CGPU', 'CGPD']


class PMC(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)

    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))


class CPMC(nn.Module):
    """CSP Bottleneck with Parallel Multi-scale Convolution."""
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = PMC(c_)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class CGPU(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(inc, inc, 1),
            nn.Hardsigmoid()
        )
        self.conv = Conv(inc, inc, k=1)
        self.up_branch1 = ConvTranspose(inc, inc // 2, 2, 2)
        self.up_branch2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc, inc // 2, k=1)
        )

    def forward(self, x):
        channel_gate = self.gate(self.globalpool(x))
        x_up = torch.cat([self.up_branch1(x), self.up_branch2(x)], dim=1) * channel_gate
        return self.conv(x_up)


class CGPD(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(inc, inc, 1),
            nn.Hardsigmoid()
        )
        self.conv = Conv(inc, inc, k=1)
        self.down_branch1 = Conv(inc, inc // 2, 3, 2)
        self.down_branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(inc, inc // 2, k=1)
        )

    def forward(self, x):
        channel_gate = self.gate(self.globalpool(x))
        x_down = torch.cat([self.down_branch1(x), self.down_branch2(x)], dim=1) * channel_gate
        return self.conv(x_down)
