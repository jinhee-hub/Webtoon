import torch.nn as nn

numz=100  # 입력 noise의 크기
ngf=64  # Gnerator에서의 feature map 크기
ndf=64  # discriminator의 feature map 크기


class Generator(nn.Module):
    def __init__(self, numz, ngf):
        super().__init__()

        # input은 latent noise z 100x1
        # deconvolution을 위해서 사용하는 ConvTranspose2d
        self.dconv1=nn.ConvTranspose2d(in_channels=numz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=False  )
        self.norm1=nn.BatchNorm2d(ngf*8)   # batch normalization
        self.act1=nn.ReLU()    # ReLU activation funtion

        #
        self.dconv2=nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False)
        self.norm2=nn.BatchNorm2d(ngf*4)
        self.act2=nn.ReLU()

        #
        self.dconv3=nn.ConvTranspose2d(ngf*4, ngf*2, 4,2,1, bias=False)
        self.norm3=nn.BatchNorm2d(ngf*2)
        self.act3=nn.ReLU()

        #
        self.dconv4=nn.ConvTranspose2d(ngf*2, ngf, 4,2,1, bias=False)
        self.norm4=nn.BatchNorm2d(ngf)
        self.act4=nn.ReLU()

        # output은 이미지 채널(rgb=3) 수이다.
        self.dconv5=nn.ConvTranspose2d(ngf, 3, 4,2,1, bias=False)
        self.act5=nn.Tanh() # 마지막에는 hyperbolic tangent를 사용한다.


    def forward(self, z):  # input z
        z = self.dconv1(z)
        z = self.norm1(z)
        z = self.act1(z)

        z = self.dconv2(z)
        z = self.norm2(z)
        z = self.act2(z)

        z = self.dconv3(z)
        z = self.norm3(z)
        z = self.act3(z)

        z = self.dconv4(z)
        z = self.norm4(z)
        z = self.act4(z)

        z = self.dconv5(z)
        z = self.act5(z)

        return z





