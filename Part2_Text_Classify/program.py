# encoding: utf-8

import sys
import os
import jieba
import re
import math
import string
import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

reload(sys);
sys.setdefaultencoding('utf-8')

basepath = os.getcwdu()
trainpath = os.path.join(basepath, u'data_train')
testpath = os.path.join(basepath, u'data_test')
validpath = os.path.join(basepath, u'data_valid')

svm_cache_path = os.path.join(basepath, u'svm_cache')
nor_cache_path = os.path.join(basepath, u'nor_cache')
jieba_cache_path = os.path.join(basepath, u'word_cache')
tfidf_cache_path = os.path.join(basepath, u'tfidf_cache')

TF_IDF_MAX = 4
# Here I Want To Use
# 

def DetectPath(fullPathName):
    if not os.path.exists(fullPathName):
            os.makedirs(fullPathName)

class FileInfo(object):
    def __init__(self):
        self.class_ = '' # like, Art, Computer, Science, etc.
        self.filename = ''
        self.subfilename = ''
        self.wordcount = 0
        self.frequence = {}
        self.tag = []

class IndexInfo(object):
    def __init__(self):
        self.y = ''
        # predict class, and yet converted into 
        # integer value
        self.tag = []

def LoadFile(AbsFilePath):
    fin = open(AbsFilePath, 'r')
    fdata = fin.read()
    
    #convert data from utf-8 to ucs
    data_ucs = fdata.decode('utf-8')
    #remove all punctuation
    data_ucs = ''.join(re.findall(u'[a-zA-z\u4e00-\u9fff]+', \
                                  data_ucs))
    fin.close()
    #return data in ucs
    return data_ucs

def WordSplit(textdata):
    result = {}
    words = jieba.cut(textdata)
    count = 0
    for word in words:
        count += 1
        if result.has_key(word):
            result[word] += 1
        else:
            result[word] = 1;
    return (result, count)

def LoadTrainData(trainfilesPath):
    result = []
    if not os.path.exists(trainfilesPath):
        raise ValueError
    # fetch all the child folder
    for subfolder in os.listdir(trainfilesPath):
        subfolderfullpath = os.path.join(trainfilesPath, \
                                         subfolder)
        if os.path.isdir(subfolderfullpath):
            # set current classname
            clsName = subfolder
            for docName in os.listdir(subfolderfullpath):
                docfullpath = os.path.join(subfolderfullpath, \
                                           docName)
                if os.path.isfile(docfullpath):
                    finf = FileInfo()
                    # conf: classname, filename
                    #  and relative name
                    finf.class_ = clsName
                    finf.filename = docfullpath
                    finf.subfilename = docName
                    # call loadfile
                    fdata = LoadFile(docfullpath)
                    # call jieba
                    (finf.frequence, finf.wordcount) = WordSplit(fdata)
                    # write wordcount
                    # write frequency
                    result.append(finf)
    return result

def TF_IDF(traindata):
    filcount = len(traindata)
    for fdata in traindata:
        tf_idf = []
        for (k, v) in fdata.frequence.items():
            tf = float(v) / float(fdata.wordcount)
            idf_J = 0
            for fil in traindata:
                if fil.frequence.has_key(k):
                    idf_J += 1
                    
            idf = math.log10(float(filcount) / float(1 + idf_J))
            tf_idf.append((k, tf*idf))
    
        tf_idf.sort(key=lambda w: w[1], reverse = 1)
        
        for i in range(0, min(len(tf_idf), TF_IDF_MAX)):
            fdata.tag.append(tf_idf[i][0])

#fetch cache data files
def FetchCatchFiles(clsDict, clsData):
    trainFile = os.listdir(jieba_cache_path)
    for fil in trainFile:
        fullpath = os.path.join(jieba_cache_path, \
                                fil)
        if os.path.isfile(fullpath):
            # create a new classDict 
            if not clsDict.has_key(fil):
                clsDict[fil] = len(clsDict)
                # use Dict, instead of List
                clsData.append({})
            idx = clsDict[fil]
            fincache = open(fullpath, 'r')
            fdata = fincache.read()
            fdata_ucs = fdata.decode('utf-8')
            '''
            at first, I use a list to save all
            the TF-IDF hi-freq word
            
            But now I want to use dict, because 
            of its high search speed.                
            '''
            word_i = 0
            for tword in fdata_ucs.split():
                clsData[idx][tword] = word_i 
                word_i += 1

            fincache.close()
    
    index_list = []
    tfidf_files = os.listdir(tfidf_cache_path)
    for cls in tfidf_files:
        fullpath = os.path.join(tfidf_cache_path, \
                                cls)
        if os.path.isdir(fullpath):
            idx = clsDict[cls]
            cls_fils = os.listdir(fullpath)
            
            for cls_fil in cls_fils:
                filpath = os.path.join(fullpath, cls_fil)
                if os.path.isfile(filpath):
                    index_fil = IndexInfo()
                    index_fil.y = idx + 1
                    fin = open(filpath, 'r')
                    fdata = fin.read()
                    for strnum in fdata.split():
                        index_fil.tag.append(string.atoi(strnum))
                    
                    index_list.append(index_fil)
    return index_list
    
def CalcCatchFiles(clsDict, clsData):
    # if there is no cache tf-idf files
    traindata = LoadTrainData(trainpath)
    # get total file number
    TF_IDF(traindata)

    # for each file
    # in those traindata files
    clsDictMax = {}
    for fdata in traindata:
        if not clsDict.has_key(fdata.class_):
            clsDict[fdata.class_] = len(clsDict)
            clsData.append({})
        idx = clsDict[fdata.class_]
        
        # detect whether a word
        # exists in the dictionary
        
        # word_index
        word_i = 0
        if clsDictMax.has_key(idx):
            word_i = clsDictMax[idx]
        for word in fdata.tag:
            if not clsData[idx].has_key(word):
                clsData[idx][word] = word_i
                word_i += 1
        clsDictMax[idx] = word_i
        
    for (classname, idx) in clsDict.items():
        fcache = open(os.path.join(jieba_cache_path, \
                                   classname), 'w')
        listout = clsData[idx].items();
        # sort via index
        listout.sort(cmp = lambda a, b:a[1]-b[1])
        for (w, i) in listout:
            fcache.write(w + '\n')
        fcache.close()
        
    # since we now have calc all the
    # traindata, we made a tf-idf cache
    # for all the original train data    
    
    index_list = []
    
    for t_fil in traindata:
        index_fil = IndexInfo()
        idx = clsDict[t_fil.class_]
        index_fil.y = idx
        # make sure the output directory exists
        DetectPath(os.path.join(tfidf_cache_path, \
                                t_fil.class_))
        fcache = open(os.path.join(tfidf_cache_path, \
                                   t_fil.class_ + '/' + \
                                   t_fil.subfilename), 'w')
        for t in t_fil.tag:
            tag_id = clsData[idx][t]
            index_fil.tag.append(tag_id + 1)
            fcache.write(str(tag_id) + '\n')
        fcache.close()
        index_list.append(index_fil)
    return index_list

def MakeX(train_data, wordcountsum, wordcnt):
    filecnt = len(train_data)
    res = []
    for fil in train_data:
        line = []
        for t in fil.tag:
            line.append(t)
        res.append(line)
    return res;

def MakeY(train_data):
    filecnt = len(train_data)
    res = np.zeros([filecnt, 1])
    for i in range(0, filecnt):
        res[i] = train_data[i].y
    return res;

if __name__ == '__main__':
    print('Choose one vs all algorithm (0) or SVM algorithm (1)')
    calcMode = 0 #input('Number 0 ~ 1?\n')
    if calcMode == 0: # One Vs All
        print 'One Vs All Selected'
        DetectPath(nor_cache_path)
    else:
        print 'SVM Selected'
        DetectPath(svm_cache_path)
    print '================================'
    # make the cache folder to save temp data
    DetectPath(jieba_cache_path)
    DetectPath(tfidf_cache_path)
    
    clsDict = {}
    clsData = []
    train_data = []
    trainFile = os.listdir(jieba_cache_path)
    if len(trainFile) == 0:
        train_data = CalcCatchFiles(clsDict, clsData)
    else:
        # detect cache file, using cached TF-IDF file:
        train_data = FetchCatchFiles(clsDict, clsData)
    
    wordcountsum = []
    sum_=0
    for i in range(0, len(clsData)):
        wordcountsum.append(sum_)
        sum_+=len(clsData[i])

    X = MakeX(train_data, clsData, sum_)
    y = MakeY(train_data)
    
    
