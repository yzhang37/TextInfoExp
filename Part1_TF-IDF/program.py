# encoding: utf-8

import sys;
import string;
import jieba;
import re;
import os;
import numpy;
import os.path;
from bs4 import BeautifulSoup
import bs4.element

reload(sys);
sys.setdefaultencoding('utf-8')

basepath = os.getcwd().replace('\\', '/');
finpath = basepath + '/data/computer'
foutpath = basepath + '/output'


class FileInfo(object):
    def __init__(self):
        self.name = ''
        self.subname = ''
        self.frequence = {}
        self.wordcount = 0
        self.recommended = []


def GetFreq(f_data):
    data = FileInfo()
    cutf = jieba.cut(f_data)
    out_data = string.join(cutf)
    
    # fout = open(foutpath + '/' + '1.txt', 'w')
    # fout.write(out_data)
    
    wordlist = out_data.split()
    
    total = {}
    for word in wordlist:
        if total.has_key(word):
            total[word] += 1
        else:
            total[word] = 1;
        data.wordcount += 1
    
    data.frequence = total
    return data
    # for (key, value) in total.items():
    #    fout.write(str(key) + ' ' + str(value) + '\n')


def fetchdatafromxml(filename):
    res = ""
    soup = BeautifulSoup(open(filename), u'html.parser')

    p = soup.find("p", class_=u'abstracts')
    if p is None or p.string is None:
        return ""
    else:
        res = res + string.strip(soup.title.string) + '\n'
        for item in p.contents:
            if type(item) == bs4.element.NavigableString:
                res = res + string.strip(item.string)
    t = re.findall(u'[A-Za-z\u4e00-\u9fff]+', res)
    res = string.join(t)
    return res

def fetchkeywordfromxml(filename):
    kw_list = []
    soup = BeautifulSoup(open(filename), u'html.parser')
    lists = soup.find_all("dt")
    for list in lists:
        if type(list) == bs4.element.Tag:
            if list.string != None:
                keyword = ""
                keyword = list.string.strip().replace(u' ', u'')
                if keyword.find(u'关键词') == 0:
                    nextitem = list
                    while nextitem.name != u'dd':
                        nextitem = nextitem.next_sibling
                    for a in nextitem.children:
                        keyword = a.string
                        if keyword != None and keyword.strip().replace(u' ', u'') != u'':
                            keyword = keyword.strip().replace(u' ', u'')
                            if kw_list.count(keyword) == 0:
                                kw_list.append(keyword)
    return kw_list

if __name__ == '__main__':
    filelist = os.listdir(finpath)
    filedata = []
    filcount = 0
    for subpath in filelist:
        path = os.path.join(finpath, subpath)
        if os.path.isfile(path):
            s = fetchdatafromxml(path)
            if s != "":
                filedata.append(GetFreq(s))
                filedata[filcount].name = path
                filedata[filcount].subname = subpath
                filedata[filcount].recommended = fetchkeywordfromxml(path)
                filcount += 1
    
    for data in filedata:
        tf_idf = []
        for (k, v) in data.frequence.items():
            tf = float(v) / float(data.wordcount)
            idf_J = 0
            for fil in filedata:
                if fil.frequence.has_key(k):
                    idf_J += 1
                    
            idf = numpy.log10(float(filcount) / float(1 + idf_J))
            tf_idf.append((k, tf*idf))
    
        tf_idf.sort(key=lambda w: w[1], reverse = 1)
        fout = open(os.path.join(foutpath, data.subname + '.txt'), 'w')
        
        fout.write(u'TF-IDF 计算得到关键词:\n')
        for i in range(0, min(len(tf_idf) - 1, 6)):
            fout.write(str(tf_idf[i][0]) + ' ')  
        fout.write('\n')
        '''
        if you want to output all the data
        reckoned by TF-IDF, remove this blk cmt.
        
        fout.write('\n\n')
        for (key, value) in tf_idf:
            fout.write(str(key) + ' ' + str(value) + '\n')
        '''
        
        '''
        Here we want to extract the original
        recommended tag for the file.
        '''
        fout.write('原文作者推荐关键词:\n')
        for keyword in data.recommended:
            fout.write(str(keyword) + ' ')
        fout.write('\n')
        fout.close()
