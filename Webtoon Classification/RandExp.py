import os
import random
import shutil

# 학습을 위해 전처리된 이미지 중 1000개의 이미지만 뽑아서 정리함
# 경로 내에 1부터 숫자로 이름지어진 전처리된 이미지 리스트
path='.../Preprocessed_webtoon'
img_list=os.listdir(path)
#print(len(img_list))

# 이미지 개수 내에서 중복되지 않게 무작위로 1 ~ 이미지 개수 내의 숫자 뽑기
sellist=[]
ran_num=random.randint(1, len(img_list))

# 무작위 숫자 번호로 제목을 갖는 전처리된 이미지에서 중복되지 않게 1000개를 뽑음
for i in range(1000):
    while ran_num in sellist:
        ran_num = random.randint(10, len(img_list))
    sellist.append(ran_num)

sellist.sort()
#print(sellist)
#print(len(sellist))

# 1000개의 이미지를 뽑아서 새로운 폴더에 저장해줌

wn=''
def read_allfile(path, sellist):
    output=sellist
    file_list=[]

    for i in output:
        if os.path.isdir(path+"/"+str(i)+'.png'):
            file_list.extend(read_allfile(path+"/"+str(i)+'.png'))
        elif os.path.isfile(path+"/"+str(i)+'.png'):
            file_list.append(path+"/"+str(i)+'.png')

    return file_list

def copyfile(file_list, new_path):
    i=0
    for src_path in file_list:
        file=src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path+"/"+str(i)+".png")
        i+=1

src_path=".../Preprocessed_webtoon/%s"%wn
new_path=".../data/webtoonName"

file_list=read_allfile(src_path, sellist)
copyfile(file_list, new_path)