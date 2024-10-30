import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, size = None):
        super().__init__()
        self.bilinear = bilinear

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(size = size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            if in_channels == 256:
                self.up = nn.Upsample(size = size, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels // 2 , out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            if in_channels == 256:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels // 2 , out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2,kernel_size=1,padding=0,bias=False)
        if in_channels == 256:
            self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0, bias=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv1(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.concat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.layer1 = (Down(64, 256))
        self.layer2 = (Down(256, 512))
        self.layer3 = (Down(512, 1024))
        self.layer4 = (Down(1024, 2048))
        self.up0 = (Up(2048, 1024, bilinear, size = (15,15)))
        self.up1 = (Up(1024, 512, bilinear, size = (31,31)))
        self.up2 = (Up(512, 256, bilinear, size = (63,63)))
        self.up3 = (Up(256, 64, bilinear, size = (127,127)))

    def forward(self, x):
        x1 = self.inc(x)#127*64
        x2 = self.layer1(x1)#63*256
        x3 = self.layer2(x2)#31*512
        x4 = self.layer3(x3)#15*1024
        x5 = self.layer4(x4)#7*2048
        y4 = self.up0(x5, x4)#15*1024
        y3 = self.up1(y4, x3)#31*512
        y2 = self.up2(y3, x2)#63*256
        y1 = self.up3(y2, x1)#127*64
        return y1,y2,y3,y4,x4

    def forward_refine(self, x):
        x1 = self.inc(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        y4 = self.layer3(x3)
        y3 = self.up1(y4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up3(y2, x1)
        return y1,y2,y3,y4



    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    net = UNet()
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()


    a = net(var)
    print('*************')
    #var = torch.FloatTensor(1, 3, 255, 255).cuda()

    b = net.forward_refine(var)
    print(b)