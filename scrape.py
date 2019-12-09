from __future__ import print_function
from bs4 import BeautifulSoup
import requests
from six.moves.urllib import parse
import csv

titles=["phap-luat", "the-gioi","xuat-ban", "kinh-doanh", "cong-nghe", "the-thao", "giai-tri"]
start="https://news.zing.vn/"
_start="https://news.zing.vn"
links=[]

for title in titles:
    for page in range(21, 40):
        url=start+title+"/trang"+str(page)+".html"
        r=requests.get(url)
        soup=BeautifulSoup(r.text,"lxml")

        aSelection=soup.select('div.responsive article.article-item p.article-thumbnail a')

        for a in aSelection:
            links.append(_start+a.attrs['href'])

contents=[]
for link in links:
    rr=requests.get(link)
    soup=BeautifulSoup(rr.text,"lxml")

    pSelection=soup.select('div.the-article-body p')
    if(len(pSelection) != 0):
        for p in pSelection:
            if (p.text != '') and ('margin' not in p.text) and ('\t' not in p.text) and ('\n' not in p.text) and (len(p.text) >= 100) and ("- " not in p.text):
                contents.append(p.text)

    

with open('content.csv','w') as file:
    for line in contents:
        file.write(line)
        file.write('\n')
file.close()