from bs4 import BeautifulSoup
import requests, re, os
import os
import shutil
import urllib.request
from urllib.request import urlretrieve


headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 safari/537.36'}

url='https://comic.naver.com/webtoon/weekday.nhn'
response=requests.get(url, headers=headers)  # web에서 네이버 웹툰 메인 페이지(요일별 웹툰나오는 페이지)의 소스 접근

#print(response.text)

soup=BeautifulSoup(response.content, 'html.parser')  # BeutifulSoup을 이용해서 response를 우리가 이해할 수 있게 해줌
#print(soup)

class_title_a_list=soup.select('a.title')
#print(class_title_a_list)

webtoonLink={}
webtoonName=[]

for a in class_title_a_list:
    a_href=a.get('href')   # 각 웹툰의 요일별 title_id를 출력할 수 있음.
    #print(a_href)
    aa_href=a_href[:-12]
    a_text=a.get_text()    # 각 웹툰의 제목
    #print(a_text)
    a_link=f'https://comic.naver.com/{aa_href}' # 웹툰 제목의 링크
    #print(a_link)
    webtoonName.append(a_text)
    webtoonLink[a_text]=a_link

name=webtoonName[0]
url1=webtoonLink[name]


page_num = 50
wn=''

for i in range(1,page_num+1):

    # Webtoon Url : ex) 바른연애길잡이
    url = "https://comic.naver.com/webtoon/list?titleId=687915&weekday=mon&page={0}".format(i)

    # 크롤링 우회
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}
    html = requests.get(url, headers=headers)
    result = BeautifulSoup(html.content, "html.parser")

    webtoonName = result.find("span", {"class", "wrt_nm"}).parent.get_text().strip().split('\n')

    cwd = os.getcwd()
    files = os.listdir(cwd)
    print(cwd)

    if os.path.isdir(os.path.join(cwd, webtoonName[0])) == False:
        os.mkdir(webtoonName[0])

    print(webtoonName[0] + "page {0} folder created successfully!".format(i))
    os.chdir(os.path.join(cwd, webtoonName[0]))

    title = result.findAll("td", {"class", "title"})
    wn=webtoonName[0]
    for t in title:

        # 웹툰 디렉토리 안에 회차별로 디렉토리 만들기
        if os.path.isdir(os.path.join(cwd, webtoonName[0], (t.text).strip())):
            break
        os.mkdir((t.text).strip())

        # 회차별 디렉토리로 이동
        os.chdir(os.getcwd() + "//" + (t.text).strip())
        print(os.getcwd())
        # 각 회차별 url
        url = "https://comic.naver.com" + t.a['href']
        # 헤더 우회해서 링크 가져오기
        html2 = requests.get(url, headers=headers)
        result2 = BeautifulSoup(html2.content, "html.parser")

        # webtoon image 찾기
        webtoonImg = result2.find("div", {"class", "wt_viewer"}).findAll("img")
        num = 1  # image_name

        for i in webtoonImg:
            saveName = os.getcwd() + "//" + str(num) + ".png"
            with open(saveName, "wb") as file:
                src = requests.get(i['src'], headers=headers)
                file.write(src.content)  #
            num += 1

        os.chdir("..")

        # 한 회차 이미지 저장 완료!
        print((t.text).strip() + "   saved completely!")

    os.chdir("..")

''' 다운 받은 파일 바탕화면으로 이동시킴
wn=''
def read_allfile(path):
    output=os.listdir(path)
    file_list=[]
    print(output)

    for i in output:
        if os.path.isdir(path+"/"+i):
            file_list.extend(read_allfile(path+"/"+i))
        elif os.path.isfile(path+"/"+i):
            file_list.append(path+"/"+i)

    return file_list

def copyfile(file_list, new_path):
    i=0
    for src_path in file_list:
        file=src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path+"/"+str(i)+".png")
        i+=1

src_path=".../%s"%wn
new_path="..."

file_list=read_allfile(src_path)
copyfile(file_list, new_path)
'''
