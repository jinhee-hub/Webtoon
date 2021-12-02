DCGAN을 이용해서 웹툰을 따라서 만들어보는 프로젝트입니다. 

웹툰 데이터는 네이버 웹툰에서 크롤링을 통해 수집하였습니다.(이 프로젝트를 위해서만 사용하였습니다.)

수집한 웹툰데이터에서 웹툰 그림의 얼굴 이미지만을 수집하기 위해서 pre-trained 모델인 YOLOv5를 사용하였습니다.
YOLOv5를 이용하여 human face로 캐릭터 얼굴을 detection하는 경우, confidence 0.4 이하이거나 detection이 이루어지지 않는 경우가 많았습니다.
그래서 labelimg라는 레이블링 tool을 사용하여 toonface로 labeling하여 새로 학습할 수 있도록 하였습니다.

총 600장의 웹툰 이미지(6 종류의 웹툰)를 학습시켜서 YOLOv5가 웹툰 캐릭터 얼굴을 detection하도록하였습니다. pre-trained된 YOLOv5에서 custom dataset을 학습시키는 코드는 다른 웹 사이트의 코드를 참조하여 진행하였고, 20 epochs동안 train하였고, confidence 0.7이상에 대해서만 detection하도록 하였습니다.
전체적으로 캐릭터 얼굴 이미지를 잘 detection하였지만, 웹툰 자체에서 얼굴을 눈코입 없이 간단하게만 표현하는 경우나 얼굴과 비슷하게 생긴 물체를 detection하는 등 문제점이 일부 있어서 직접 수작업으로 안 좋은 데이터를 제거하였습니다.

웹툰 하루만네가되고싶어의 썸네일 이미지를 가지고 학습된 YOLOv5로 캐릭터 얼굴을 detection한 결과

![tobu detect](https://user-images.githubusercontent.com/54815470/142968225-678ae409-6387-4f04-9c37-53b775671890.jpg)

이렇게 수집한 웹툰 캐릭터 얼굴 데이터를 6 종류의 웹툰 별로 분류하여 저장하였고, 각 웹툰에 맞게 이미지를 생성시킬 수 있도록 DCGAN을 적용하였습니다.

1.DCGAN으로 웹툰 생성하기

결과부터 적으면.... 웹툰 이미지를 생성하는데 실패하였습니다. 
각 웹툰별로 약 900장의 웹툰 캐릭터 데이터를 확보하였기 때문에 확보한 웹툰 데이터의 숫자가 적어서 Discriminator에서 Overfitting이 발생한 것으로 생각합니다.
100 epochs에 대해서 학습을 진행하였고, Generator에서는 loss가 증가하고, Discriminator에서는 loss가 감소하였습니다.
Discriminator가 input으로 들어온 이미지가 진짜인지 가짜인지 혼동해야하는데 너무 확실하게 판별을 해버렸습니다.

DCGAN을 이용한 학습...

epoch: 0,  G_loss: 5.837123,  D_loss: 0.004208

epoch: 10,  G_loss: 9.988934,  D_loss: 0.000146

epoch: 20,  G_loss: 10.968609,  D_loss: 0.000043

epoch: 30,  G_loss: 11.627301,  D_loss: 0.000014

epoch: 40,  G_loss: 13.657906,  D_loss: 0.000003

epoch: 50,  G_loss: 13.952927,  D_loss: 0.000001

epoch: 60,  G_loss: 14.999187,  D_loss: 0.000001

epoch: 70,  G_loss: 14.471410,  D_loss: 0.000000

epoch: 80,  G_loss: 17.116364,  D_loss: 0.000000

epoch: 90,  G_loss: 16.569757,  D_loss: 0.000000

epoch: 100,  G_loss: 16.798510,  D_loss: 0.000000


데이터를 몇 만장 이상은 확보해야 overfitting이 발생하지 않을것으로 생각되는데, 웹툰이 아직 많은 회차가 나온것이 아니라서 현실적으로 획득하기 어려울거이라고 예상합니다.
그래서 augmentation이나 다른 overfitting을 줄일 수 있는 방법을 찾고, 더 최적화가 잘 된 GAN 모델에 대한 자료조사를 더 할 예정입니다.

+앞으로 Data 확보 및 StyleGAN2 모델 사용 그리고 ada(Adaptive Discriminator Augmentation)를 사용하면 비교적 적은 데이터에서 웹툰이미지를 생성할 수 있을 것으로 생각됩니다.
관련 논문과 내용을 공부한 후 적용해볼 예정입니다.

2.StyleGAN2 + Ada 로 웹툰 얼굴 생성하기

Nvidia에서 구현한 StyleGAN2-ada-pytorch 모델을 바탕으로 다시 웹툰 생성을 시도하였습니다.
ffhq dataset으로 pretrained된 모델에 1000장보다 적은 양의 웹툰 data를 사용하여 학습시켰고, 학습은 GPU RTX2060 Super를 사용하여 약 4시간 반 정도 진행하였습니다. RTX2060 Super 8GB는 NvLabs에서 만든 StyleGAN2-ada 모델의 requirement를 만족시키지 못하는 장비입니다. 이 장비로 학습시키려고 했더니 Cuda memory allocation issue가 발생하였습니다. 이를 해결하기 위해 1024x1024의 이미지를 256x256이미지로 변환하였고, batch_size도 작게 수정하여 학습을 진행하였습니다.

먼저 웹툰 이미지 data를 NvLabs에서 제공하는 StyleGAN2_ada 모델에 사용할 수 있도록 dataset_tool.py를 이용하여 TFRecords로 변환시켰습니다.

python dataset_tool.py --source=~/wtdata --dest=~/tfRecords/wtdata

그 후, 학습을 진행시켰습니다. setting은 부족한 장비에서도 학습이 진행되도록 batch_size=8로 낮추고, pretrained 모델을 사용하기 위해 ffhq의 .pkl을 사용하였습니다. 또한, x-flip, ada를 통해 부족한 data 커버하고 augmentation을 진행하였습니다.

python train.py --outdir=~/out --snap=1 --batch=8 --aug=ada --data=~/tfRecords/wtdata --augpipe=bgcfnc --mirror=True --metrics=None --resume=ffhq256

개인 pc에서 약 4시간 반 정도의 학습을 진행하였습니다.

학습한 결과를 바탕으로 생성시킨 가짜 웹툰 이미지입니다. 랜덤채팅의 그녀! 웹툰 데이터만을 학습한 결과입니다. 

![gen1](https://user-images.githubusercontent.com/54815470/143019655-2ca897ea-6413-424b-833c-93b96ec2aaa7.png)
![gen2](https://user-images.githubusercontent.com/54815470/143019665-2e941067-352e-4a2a-b301-cf1320750d6f.png)
![gen3](https://user-images.githubusercontent.com/54815470/143019630-5b2c19ba-2f61-4f34-85ad-a5bd738dca11.png)
![gen4](https://user-images.githubusercontent.com/54815470/143019678-c269394f-15ba-4f6d-8013-25b5d7952d01.png)
![gen5](https://user-images.githubusercontent.com/54815470/143019690-1f21fcb7-71fa-44b9-9254-3194543f3bdc.png)

data 수 부족과 함께 충분한 시간동안 학습되지 않아 완벽한 결과를 내지는 못했습니다. 눈이 짝짝이거나, 머리카락이 남자인지 여자인지 애매한 경우가 가장 많았습니다. 특히, 심각했던 부분은 생성한 이미지의 형태가 실제 웹툰에 등장하는 인물과 매우 흡사하다는 점입니다. 랜덤채팅의 그녀!에 등장하는 인물들이 한정되다보니, 학습에 사용한 인물들의 디자인에 다양성이 부족하여 발생한 것으로 생각됩니다.  









