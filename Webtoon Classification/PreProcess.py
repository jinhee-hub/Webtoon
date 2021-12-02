import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

# 웹툰 이미지에서 빈공간 제외하고 그림만 detection하는 코드

path=".../webtoon_crawling_folder"

os.chdir(path)
images=os.listdir(path)

cnt=1
for i in range(len(images)):
    image=cv2.imread(images[i])
    img=image.copy()   # crop하기 위해 미리 복사본을 만들어 놓음
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환

    # edge detection
    canny=cv2.Canny(gray, 2, 1)  # 엣지 검출 과정에서 배경이 아닌 모든 그림을 검출하기 위해서 최대한 민감하게 적용함.

    # morphological transformation
        # 삐져나온 부분이 있더라도 하나의 컷에 담겼다면, 최대한 하나의 사각형으로 묶어준다.
        # 이유는, 웹툰 한 컷에 대해서 내가 원하는 현재 컷 전체 이미지 이외에 불필요한 부분컷들이 생길 수 있기 때문이다.
        # 컷 사이의 공백이 작은 경우를 대비해서, col방향은 적당히 넓게, row방향은 최대한 짧게 잡았다.
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))  # 웹툰에 따라서 수치를 변경해줘야함.
    # 히어로 메이커 (2,1) 하루만 네가 되고 싶어
    # kernel으로 최대한 이미지 범위를 뽑으면, closing으로 dilation과 erosion을 적용시켜 하나의 컷을 분명하게 만든다.
    closing=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    #plt.imshow(closing)
    #plt.show()

    # 외곽선을 찾음.
    # closing으로 배경이 아닌 하나의 컷으로 인식된 웹툰 이미지의 경계를 찾음. 이미지 전체를 자르는 것이기 때문에 External을 사용함.
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # c는 크롤링된 웹툰 데이터에서 하나의 분리된 컷으로 인식되는 이미지 영역


    for c in contours:
        x,y,w,h=cv2.boundingRect(c) # contour한 영역을 사각형의 좌표값을 잡아줌.

        # 잘못 검출된 이미지 제거
        # 앞에서 완전히 노이즈를 제거하지 못해서, 전체 컷이 아닌 한 컷의 부분컷이 잡히는 것을 없애줌
        # 기준은 이미지 크기: 크롤링된 이미지에서 30%도 되지 않는 컷은 과감히 버림
        if h < (image.shape[0])*0.3 or w<(image.shape[1])*0.3:
            continue
        crop_img=img[y:y+h, x:x+w]
        #plt.imshow(crop_img)
        #plt.show()

        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        edge = cv2.Canny(crop_gray, 5000, 1500, apertureSize=5, L2gradient=True )
        lines=cv2.HoughLinesP(edge, 1, np.pi/180, 300, minLineLength=200, maxLineGap=500)

        # 현재 찾은 line에서 최소 x,y, 최대 x,y 를 찾아서 웹튼 이미지 좌표를 찾음
        minx=0
        maxx=crop_img.shape[1]
        miny=0
        maxy=crop_img.shape[0]
        xs=[]
        ys=[]

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line[0]
                xs.append(x1)
                xs.append(x2)
                ys.append(y1)
                ys.append(y2)

            #plt.imshow(crop_img)
            #plt.show()


            if len(xs)>2:
                minx=min(xs)
                maxx=max(xs)
            else:
                if len(xs)==2:
                    if max(xs)<maxx//2:
                        minx=min(xs)
                    else:
                        maxx=max(xs)

            if len(ys)>2:
                miny=min(ys)
                maxy=max(ys)
            else:
                if len(ys) == 2:
                    if max(ys) < maxy // 2:
                        miny = min(ys)
                    else:
                        maxy = max(ys)
            #print(minx, miny, maxx, maxy)

            if maxy-miny < maxy*0.3 or maxx-minx < maxx*0.3:  # 가끔 x나 y값이 매우 낮은 점이 잡혀서 의미없는 data 생기는 것 방지
                continue
            else:
                new_img = crop_img[miny:maxy, minx:maxx]

                #plt.imshow(new_img)
                #plt.show()
                cv2.imwrite(".../%d.png" % cnt, new_img)
                cnt += 1