import torch.nn as nn

ndf=64  # discriminator의 feature map 크기

class Discriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()

        # input은 이미지(rgb=3) 크기 만큼이다. DCGAN에서는 Discirimator에 LeakyReLU를 사용한다
        self.conv1=nn.Conv2d(in_channels=3, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1=nn.LeakyReLU(0.2, inplace=True)

        #
        self.conv2=nn.Conv2d(ndf, ndf*2, 4,2,1, bias=False)
        self.norm2=nn.BatchNorm2d(ndf*2)
        self.act2=nn.LeakyReLU(0.2, inplace=True)

        #
        self.conv3=nn.Conv2d(ndf*2, ndf*4, 4,2,1, bias=False)
        self.norm3=nn.BatchNorm2d(ndf*4)
        self.act3=nn.LeakyReLU(0.2, inplace=True)

        #
        self.conv4=nn.Conv2d(ndf*4, ndf*8, 4,2,1, bias=False)
        self.norm4=nn.BatchNorm2d(ndf*8)
        self.act4=nn.LeakyReLU(0.2, inplace=True)

        #
        self.conv5=nn.Conv2d(ndf*8, 1, 4,1,0, bias=False) # output은 1
        self.act5=nn.Sigmoid()

    def forward(self, x):

        x=self.conv1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.act2(x)
        x=self.conv3(x)
        x=self.norm3(x)
        x=self.act3(x)
        x=self.conv4(x)
        x=self.norm4(x)
        x=self.act4(x)
        x=self.conv5(x)
        x=self.act5(x)
        return x.view(-1,1)





