import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image

import generator   # Generator 코드
import discriminator  # discriminator 코드

seed=20
random.seed(seed)
torch.manual_seed(seed)

# GPU 환경 확인
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

# GPU 세팅
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# 웹툰 이미지 데이터가 있는 경로 및 이미지 확인하기
path='.../webtoondata/'
imagepath=path+'nano/0.png'
img=cv2.imread(imagepath)
print(img.shape)
#plt.imshow(img)
#plt.show()

# 사전 parameter 세팅
batch_size=32  # 배치 크기
image_size=64 # 3x64x64의 이미지
lr = 0.00002 # learning rate

# 이미지를 원하는 형태(크기, 텐서, 정규화 등등)로 변환 3x64x64
tf=transforms.Compose([transforms.Resize(64),
                       transforms.RandomCrop(64),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                       ])

# 경로에 있는 웹툰 이미지 데이터로 train dataset 만들기
train_data=ImageFolder(root=path, transform=tf)
train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)


# 데이터 시각화하기
dataiter=iter(train_loader)
images,labels=dataiter.next()

def imshow(image):
    image=image/2+0.5
    np_img=image.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
#imshow(torchvision.utils.make_grid(images, nrow=8))
#print(images.shape)
#plt.show()


numz=5  # Generator의 입력 크기(noise)
ngf=64  # Gnerator에서의 feature map 크기
ndf=64  # discriminator의 feature map 크기


# weight 초기화
def weights_init(model):
    classname=model.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)  # 평균 0, 분산 0.02
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# Generator
G=generator.Generator(numz, ngf).to(DEVICE)
# Generator 내의 모든 weight들의 평균을 0, 분산을 0.02로 초기화해줌
G.apply(weights_init)
#print(G)

# Discriminator
D=discriminator.Discriminator(ndf).to(DEVICE)
# Discriminator의 모든 weight들의 평균을 0, 분산을 0.02로 초기화
D.apply(weights_init)
#print(D)

# Loss funtion으로 Binary Cross Entropy Loss를 사용함
criterion=nn.BCELoss()

# Generator와 Discriminator에 대한 각각의 optimizer 설정
optimizerG=optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD=optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))


### train 하기
fixed_noise=torch.randn(64, numz, 1,1, device=DEVICE)
epochs=101
imgs=[]
G.train()
D.train()
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        realX=batch_x.to(DEVICE)  # 이미지 데이터

        y_real=torch.ones(batch_size, 1).to(DEVICE)  # label을 1로 채운 텐서
        y_fake=torch.zeros(batch_size, 1).to(DEVICE)  # label을 0으로 채운 텐서

        # Generator
        G.zero_grad()

        noise = torch.randn(batch_size, numz, 1, 1, device=DEVICE)  # Genertor에서 활용할 latent space vector생성
        fakeX=G(noise).detach()  # Generator로 fake 이미지 생성
        D_fake_pred = D(fakeX)  # Discriminator로 fake 이미지를 판별함
        G_loss = criterion(D_fake_pred, y_real)
        G_loss.backward()  # Generator의 back propagation
        optimizerG.step()  # Generator의 parameter 업데이트

        # Discriminator
        D.zero_grad()
        D_real_pred=D(realX)  # real data를 D가 판별함
        D_real_loss=criterion(D_real_pred, y_real)  # real data를 진짜라고 잘 판별하는지 loss를 구함

        D_fake_pred = D(fakeX)  # Generator가 생성해낸 가짜 data를 Discriminator가 판별
        D_fake_loss = criterion(D_fake_pred, y_fake)  # 가짜 data의 loss
        #  G(z) = fakeX -> D(fakeX) fake input에 D
        # D(X)  real input에 D

        D_loss = (D_real_loss + D_fake_loss)/2  # Discriminator의 전체 loss
        D_loss.backward()
        optimizerD.step()  # Discriminator의 parameter 업데이트

    if epoch % 10 == 0:
        print('epoch: %.0f,  G_loss: %.6f,  D_loss: %.6f' % (epoch, G_loss.item(), D_loss.item()))


G.eval()

with torch.no_grad():
    fakeimg=G(fixed_noise).detach().cpu()
print(fakeimg.shape)

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4, i+1)
    plt.imshow(to_pil_image(0.5*fakeimg[i]+0.5), cmap='gray')
plt.show()

