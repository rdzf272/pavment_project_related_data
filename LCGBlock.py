import torch
import torch.nn as nn

class LCGBlock(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = LightConv(c1, c_, 1, act=False)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1) 
        self.m = nn.Sequential(*(LGhostBottleneck(c_, c_) for _ in range(n)))
   def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class LGhostBottleneck(nn.Module):


    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        # print('-----------------stride:',s)
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()
        self.shortcut2=nn.Sequential(LightConv(c1, c2, 1, act=False))
        self.Pointwise_Convolution1 = nn.Conv2d(in_channels=c1,
                                    out_channels=c2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.Pointwise_Convolution2 = nn.Conv2d(in_channels=c1,
                                    out_channels=c2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        # return self.conv(x) + self.Pointwise_Convolution1(self.shortcut(x))+self.Pointwise_Convolution2(self.shortcut2(x))
        # return self.Pointwise_Convolution1(self.conv(x)) + self.Pointwise_Convolution2(self.shortcut(x))
        return self.conv(x)+self.shortcut(x)
