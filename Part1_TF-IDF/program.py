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
    res = string.join(re.findall(u'[\u4e00-\u9fff]+', res))
    return res


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
        
        for i in range(0, min(len(tf_idf), 4) - 1):
            fout.write(str(tf_idf[i][0]) + ' ')  
        fout.write('\n\n')
        for (key, value) in tf_idf:
            fout.write(str(key) + ' ' + str(value) + '\n')
            
        fout.close()
