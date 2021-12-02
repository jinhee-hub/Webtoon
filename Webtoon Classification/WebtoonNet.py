import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np


# 필요한 parameter 세팅하기

learning_rate=0.001
batch_size=32
epochs=1
image_size=224
n_classes=6

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

print(DEVICE)

# local 경로에 있는 내가 수집한 이미지 data를 이용해서 dataset 만들기

# 단일 이미지 데이터 확인
path='.../nano'
imagepath=path+'/'+'0.png'
img=cv2.imread(imagepath)
print(img.shape)
#plt.imshow(img)
#plt.show()

# 원하는 형태로 이미지를 변환하기 위한 작업
tf=transforms.Compose([transforms.Resize((256,256)),
                          transforms.RandomCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
                          ])

# path에 있는 data를 train_data로 만들기
# ImageFolder는 경로 내의 폴더이름별로 이미지를 분류해줌. ex) A 폴더의 이미지들은 label이 A임. B 폴더는 B label
train_data=ImageFolder(root='.../train', transform=tf)
classes=train_data.classes   # 폴더 개수만큼 class가 생김
print(classes)

# test data
test_data=ImageFolder(root='.../test', transform=tf)
testclasses=test_data.classes   # 폴더 개수만큼 class가 생김
print(testclasses)

# train data를 batch_size 만큼으로 쪼개고, 무작위로 섞을 수 있다.
train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_data, batch_size=50, shuffle=True)


# 이미지와 label이 맞나 확인해 볼 수 있음.
dataiter=iter(train_loader)
images,labels=dataiter.next()

# 이미지 data를 batch_size 만큼 모아서 보기 위한 작업
def imshow(image):
    image=image/2+0.5
    np_img=image.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
imshow(torchvision.utils.make_grid(images, nrow=8))
print(images.shape)
#plt.show()


# 논문 리뷰를 위해 구현해놓은 DenseNet을 그대로 사용하였다.
# 자세한 내용은 내 github 주소:https://github.com/jinny6876/Projects의 AI paper implemnetation의 DenseNet을 참고하시면 됩니다.

# DenseNet 구현에 필요한 BottleNet구조
class BottleNeck(nn.Module):
    def __init__(self, n_in, growth_rate):  # growth rate만큼 node 수를 확보한다.
        super().__init__()

        # Bottleneck은 Batch normalization, ReLU, 1x1 conv인 layer와  BN, ReLU, 3x3 conv인 Layer 두 부분으로 되어 있다.
        # 1x1에 의해 4*growth_rate 만큼의 output이 나오고, 이것이 3x3의 input으로 들어가 growth_rate 만큼의 output이 나온다.
        self.norm1 = nn.BatchNorm2d(n_in)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(n_in, 4 * growth_rate, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential()  # 이를 들어온 현재 Layer에 들어온 input을 concatenate하기 위해 사용한다.

    def forward(self, x):
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = torch.cat((x, out), 1)  # input인 x와 현재 layer에서 계산된 out을 하나의 Tensor로 만들어줌.

        return out

# Dense Block 이후에 1x1 conv와 pooling를 해주는 Layer
class Transition_Layer(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()

    self.norm1=nn.BatchNorm2d(n_in)
    self.act1=nn.ReLU()
    self.conv1=nn.Conv2d(n_in, n_out, kernel_size=1)  # 1x1 conv
    self.pool1=nn.AvgPool2d(2, stride=2)  # 2x2 average pooling, stride=2

  def forward(self,x):
    x=self.norm1(x)
    x=self.act1(x)
    x=self.conv1(x)
    x=self.pool1(x)

    return x


# DenseNet 정의
class DenseNet(nn.Module):
  def __init__(self, bottleNeck, growth_rate, num_blocks, num_classes):
    super().__init__()

    n_in=2*growth_rate  # 논문에서 growth rate의 2배로 정함

    self.conv1=nn.Conv2d(3, n_in, kernel_size=7, stride=2, padding=3)  # rgb 3 (3x224x224)이므로 input=3임
    self.pool1=nn.MaxPool2d(3, stride=2, padding=1)

    # DenseBlock1 & Transition Layer1
    self.block1=self.denseBlock(bottleNeck, n_in, growth_rate, num_blocks[0])
    n_in+=num_blocks[0]*growth_rate  # denseblock 내에서 growth rate만큼 channel수가 증가함
    n_out=int(n_in*0.5)        # Transition Layer에서 Average pooling으로 절반으로 감소함
    self.trans1=Transition_Layer(n_in, n_out)

    # DenseBlock2 & Transition Layer2
    n_in=n_out
    self.block2=self.denseBlock(bottleNeck, n_in, growth_rate, num_blocks[1])
    n_in+=num_blocks[1]*growth_rate
    n_out=int(n_in*0.5)
    self.trans2=Transition_Layer(n_in, n_out)

    # DenseBlock3 & Transition Layer3
    n_in=n_out
    self.block3=self.denseBlock(bottleNeck, n_in, growth_rate, num_blocks[2])
    n_in+=num_blocks[2]*growth_rate
    n_out=int(n_in*0.5)
    self.trans3=Transition_Layer(n_in, n_out)

    # DenseBlock4
    n_in=n_out
    self.block4=self.denseBlock(bottleNeck, n_in, growth_rate, num_blocks[3])
    n_in+=num_blocks[3]*growth_rate
    self.norm1=nn.BatchNorm2d(n_in)   # 마지막 block에서는 transition Layer 대신에 BN, ReLU, 7x7 Global Average Pooling 을 해줘야함
    self.act1=nn.ReLU()
    self.pool2=nn.AvgPool2d(7, stride=1) # 7x7 average pooling

    # Fully connected Layer
    self.fc=nn.Linear(n_in, num_classes)

  # DenseBlock 만들기
  def denseBlock(self, bottleNeck, n_in, growth_rate, num_denseblock):  # Resnet의 residual Layer와 비슷함
    layers=[]
    for i in range(num_denseblock):                 # dense block 내의 bottleNeck 수. 논문에서는 DensNet121은 각 block에 따라 6,12,24,16 개임
      layers.append(bottleNeck(n_in, growth_rate))  # bottleNeck을 이용해서 Dense Block 생성함
      n_in+=growth_rate
    return nn.Sequential(*layers)

  def forward(self, x):
    x=self.conv1(x)
    x=self.pool1(x)
    x=self.block1(x)
    x=self.trans1(x)
    x=self.block2(x)
    x=self.trans2(x)
    x=self.block3(x)
    x=self.trans3(x)
    x=self.block4(x)
    x=self.norm1(x)
    x=self.act1(x)
    x=self.pool2(x)

    x=x.view(x.size(0), -1)
    x=self.fc(x)

    return x

num_classes=len(classes)  # Dataset의 class 수

model=DenseNet(bottleNeck=BottleNeck, growth_rate=12, num_blocks=[6,12,24,16], num_classes=num_classes).to(DEVICE)
#print(model)
#summary(model, input_size=(3,224,224))


# Optimizer로 Adam을 사용하였고, Cross Entropy loss를 loss function으로 사용하였다.
optimizer=optim.Adam(model.parameters(), lr=learning_rate)
criterion=nn.CrossEntropyLoss()


# train하기
def train(train_loader, model, criterion, optimizer, device):
    running_loss = 0
    count = 0
    correct = 0

    model.train()
    for batch in train_loader:
        X = batch[0]
        y_true = batch[1]

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward
        y_pred = model(X)
        loss = criterion(y_pred, y_true)
        running_loss += loss.item()
        batch_correct = torch.argmax(y_pred, dim=1).eq(y_true).sum().item()
        batch_count = len(X)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += batch_count
        correct += batch_correct

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / count
    return model, epoch_loss, epoch_accuracy

# test set으로 evaluation해보기
def evaluation(test_loader, model, criterion, device):
    running_loss = 0
    count = 0
    correct = 0

    model.eval()
    for batch in test_loader:
        X = batch[0]
        y_true = batch[1]

        X = X.to(device)
        y_true = y_true.to(device)

        y_pred = model(X)
        loss = criterion(y_pred, y_true)
        batch_correct = torch.argmax(y_pred, dim=1).eq(y_true).sum().item()
        batch_count = len(X)
        running_loss += loss.item() * X.size(0)

        count += batch_count
        correct += batch_correct

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_accuracy = correct / count

    return model, epoch_loss, epoch_accuracy

# train
train_losses=[]
train_acc=[]
for epoch in range(epochs+1):
  model, train_loss, train_accuracy=train(train_loader, model, criterion, optimizer, DEVICE)
  train_acc.append(train_accuracy)
  train_losses.append(train_loss)

  if epoch%3==0:
      print('epoch: ', epoch, 'train_loss: ', train_loss)
      print('          train accuracy: ', train_accuracy)

# test
with torch.no_grad():
 model, test_loss, test_accuracy=evaluation(test_loader, model, criterion, DEVICE)
print('test loss: ', test_loss, '    test accuracy: ', test_accuracy)

ep=list(range(epochs+1))
plt.figure()
plt.title('Train loss')
plt.plot(ep, train_losses, 'bo-')
plt.savefig('train_loss.png')
plt.show(block=True)


# Thumbnail 이미지로 Test
thum_test_data = ImageFolder(root='.../thum_data', transform=tf)
thum_test_loader=DataLoader(thum_test_data, batch_size=num_classes, shuffle=False)

target=[]
predict=[]
for thums in thum_test_loader:
    thum_data=thums[0].to(DEVICE)
    thum_y_true=thums[1].to(DEVICE)

    thum_y_pred = model(thum_data)
    predictLabel=thum_y_pred.argmax(dim=1)

    for i in range(num_classes):
        target.append(classes[thum_y_true[i]])
        predict.append(classes[predictLabel[i]])

print("True Labels:      ", target)
print("Predicted Labels: ", predict)

thum_dataiter=iter(thum_test_loader)
thum_images,labels=thum_dataiter.next()
imshow(torchvision.utils.make_grid(thum_images, nrow=num_classes))
plt.savefig('thumbnails.png')
plt.show()


