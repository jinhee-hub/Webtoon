# **웹툰 프로젝트**
 
### Crawling.py

인터넷에서 웹툰 데이터를 크롤링해오는 코드입니다. 
크롤링한 이미지는 웹툰 한컷씩 크롤링되지 않고, 여러 컷이 분리되어 합쳐져 하나의 이미지로 만들어져 있습니다.
이 웹툰 프로젝트는 웹툰의 한컷 한컷의 만화 그림 이미지를 사용해야하기 때문에 바로 사용할 수가 없었고, 이를 해결하기 위해서 전처리 과정을 통해 한컷 한컷을 분리해내는 과정을 진행하였습니다.

크롤링 예시

![crawlingImage](https://user-images.githubusercontent.com/54815470/133966483-6217447f-3f18-42e7-9270-dea5ac71a7ef.png)

웹툰을 크롤링하면 여러 컷이 동시에 한 이미지로 묶어서 나옵니다.


### PreProcess.py

크롤링된 웹툰 데이터를 전처리하여 학습에 사용할 수 있게 한컷한컷 분리하는 코드입니다.
Canny edge detection과 Morphological transformation, HoughLinesP를 사용하여 웹툰 이미지에서 하나의 컷을 분리해내었습니다.

Preprocessing 예시

1. edge detection과 Morphological Transformation으로 하나의 컷이라고 인식되는 영역을 분리합니다.

![MorphTFImage](https://user-images.githubusercontent.com/54815470/133968632-1e70963a-b713-487d-bd00-bbd237c53efc.png)


2. 한 컷으로 선택된 영역을 분리해내고, 말풍선과 같이 그림을 그리는 사각형 박스를 넘어간 부분을 잘라냅니다.

![CropImage](https://user-images.githubusercontent.com/54815470/133968647-8a6ede78-470b-46e8-8d27-12f17bd71eee.png)


최종적으로 사각형 박스에 잡힌 웹툰 이미지를 Data로 사용합니다.

이렇게 크롤링한 데이터를 전처리하는 과정을 거쳤지만 여러가지 문제점이 남아있었습니다.

1. 크롤링과정에서 잘려나간 이미지를 자동으로 합체해주지는 못했습니다.
2. 웹툰 내에 있는 말풍선을 분리해낸 완전한 이미지를 만들 수는 없었고, 이로 인해 이미지 내에 텍스트와 말풍선이 그대로 남아 있습니다.
3. 웹툰 컷을 분리해내는 과정에서 깔끔하게 하나의 컷으로 분리하지 못하고, 말풍선이 튀어나온 배경까지 하나의 컷으로 인식하는 문제점이 일부 있었습니다.
4. 말풍선에 의해 웹툰의 컷이 이어지거나 깔끔하게 컷 분리가 이루어지지 않는 경우, 여러 개의 컷을 하나의 컷으로 인식하였습니다. 이 경우, 데이터로 사용할 수 없어서 버려졌습니다.
5. 여백이 남아있는 컷이 일부 있었습니다. 여백이 남아있는 컷이 대부분은 아니지만, 무시하기에는 많았기 때문에, 이를 그대로 사용하였습니다. 다음에는 더 정교한 전처리과정을 통해 발전시킬 예정입니다. 


### RandExp.py

전처리 과정을 거친 이미지 중에서 1000개의 데이터만 따로 빼내어 인공지능 학습을 위한 하나의 dataset을 만들었습니다.
전처리 과정을 거친 후에도 남아있는 불완전한 이미지(말풍선만 있다거나, 아무의미 없는 데이터)는 수작업으로 제거하였습니다.
보통 데이터 1000개당 10~20개의 불필요한 이미지가 있었고, 많으면 50개 정도의 불필요한 이미지가 있었습니다.
이러한 데이터는 수작업으로 다른 괜찮은 데이터로 채워넣어서 1000개의 dataset을 완성시켰습니다.
동일한 방법으로 250개의 웹툰 이미지를 Test dataset으로 만들었습니다.

## 뉴럴네트워크 모델로 그림체 분류하기

### WebtoonNet.py

DenseNet을 이용해서 앞의 과정을 통해 획득한 웹툰 data를 분류하는 작업을 하였습니다. 웹툰 이미지가 어떤 그림체의 웹툰인지 분류하였습니다.
DenseNet은 제가 Github에 논문을 리뷰한 AI paper implementation의 DenseNet에서 구현한 모델을 사용하였습니다.

30 epochs동안 학습을 진행하였고, train_dataset에 대해서 96.9%의 높은 정확도를 보였습니다.

epoch:  0 train_loss:  0.025087374344468116
          train accuracy:  0.6981666666666667
          
epoch:  3 train_loss:  0.011884582887093226
          train accuracy:  0.8611666666666666
          
epoch:  6 train_loss:  0.008652871427436669
          train accuracy:  0.9011666666666667
          
epoch:  9 train_loss:  0.006698773395890991
          train accuracy:  0.9263333333333333
          
epoch:  12 train_loss:  0.005389556522170703
          train accuracy:  0.9365
          
epoch:  15 train_loss:  0.004634129839638869
          train accuracy:  0.944
          
epoch:  18 train_loss:  0.004011173550738022
          train accuracy:  0.9565
          
epoch:  21 train_loss:  0.0035094624849346776
          train accuracy:  0.9615
          
epoch:  24 train_loss:  0.0036466527516798427
          train accuracy:  0.9595
          
epoch:  27 train_loss:  0.003175077874911949
          train accuracy:  0.9651666666666666
          
epoch:  30 train_loss:  0.0027046352627221495
          train accuracy:  0.9688333333333333
          

epoch에 대해서 train_loss를 그래프로 나타내었습니다. 더 많이 학습할수록 loss가 감소하였습니다.

![train_loss](https://user-images.githubusercontent.com/54815470/134124116-672f3518-a052-4d77-a253-84b253c20403.png)

250개의 웹툰 data로 이루어진 testset에 대해서는 95.9%의 정확도를 나타내었습니다.

test loss:  0.14933344076077143     test accuracy:  0.9593333333333334

마지막으로 웹툰의 Thumbnail 이미지로 테스트한 결과, 완벽하지는 않지만 모델이 어느 정도 그림체를 잘 분류하였습니다.

![thums](https://user-images.githubusercontent.com/54815470/134125762-f7ffc7c7-775e-4ecf-aa77-9e34d8f07fad.png)

True Labels:       ['chim', 'dreamcorp', 'heromaker', 'nano', 'randchat', 'tobu']

Predicted Labels:  ['chim', 'dreamcorp', 'nano', 'nano', 'randchat', 'tobu']

썸네일로 테스트한 결과 히어로메이커를 나노마신으로 분류한 것 이외에는 그림체를 잘 분류하였습니다. 

%학습에 사용된 웹툰은 제가 즐겨보거나 봤던 네이버 웹툰의 6개의 웹툰 data를 사용하였습니다.

% chim은 "이말년시리즈", dreamcorp는 "꿈의 기업, heromaker는 "히어로메이커", nano는 "나노마신", randchat은 "랜덤채팅의 그녀!", tobu는 "하루만 네가 되고 싶어" 입니다.  
  
%크롤링한 데이터는 웹툰 개인 프로젝트를 위해서만 사용하였습니다. 
