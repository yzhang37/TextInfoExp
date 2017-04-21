# encoding: utf-8

import sys
import os
import jieba
import re
import math
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

reload(sys);
sys.setdefaultencoding('utf-8')

basepath = os.getcwdu()
trainpath = os.path.join(basepath, u'data_train')
testpath = os.path.join(basepath, u'data_test')
validpath = os.path.join(basepath, u'data_valid')

train_cachepath = os.path.join(basepath, u'cached_train')
test_cachepath = os.path.join(basepath, u'cached_test')
valid_cachepath = os.path.join(basepath, u'cached_valid')

other_tag = u'_Others_'

# TF_IDF_MAX = 4

def DetectPath(fullPathName):
    if not os.path.exists(fullPathName):
            os.makedirs(fullPathName)

class TrainDataType(object):
    def __init__(self):
        # the 
        self.y = []
        self.data = []
        self.X_train_counts = None
        self.target_name = []
        self._target_name_solve = {}
        self.file_name = []
    
    def ExistTarget(self, name):
        return self._target_name_solve.has_key(name)
    
    def AddTarget(self, name):
        if not self._target_name_solve.has_key(name):
            i = len(self._target_name_solve)
            self._target_name_solve[name] = i
            self.target_name.append(name)
            return i
        else:
            return self._target_name_solve[name]
    
    def GetTargetId(self, name):
        if self._target_name_solve.has_key(name):
            return self._target_name_solve[name]

def LoadFileSp(AbsFilePath):
    fin = open(AbsFilePath, 'r')
    fdata = fin.read()
    fin.close()
    #convert data from utf-8 to ucs
    data_ucs = fdata.decode('utf-8')
    return data_ucs

def LoadFile(AbsFilePath):
    data_ucs = LoadFileSp(AbsFilePath)
    #remove all punctuation
    data_ucs_chs = ''.join(re.findall(u'[\u4e00-\u9fff]+', \
                                  data_ucs))
    list_eng =  re.findall(u'[a-zA-z\s]+', data_ucs)
    data_ucs_eng = ""
    for word in list_eng:
        word = word.strip()
        if len(word) > 0:
            data_ucs_eng += ' ' + word.strip()
    return data_ucs_chs + " " + data_ucs_eng.strip()

# here I want to fetch the 
# test data file correct answers
def LoadSrcFile():
    SrcTestDict = {}
    srcfilepath = os.path.join(basepath, 'src/id2class.txt')
    with open(srcfilepath, 'r') as fin:
        filedata = fin.readlines()
        
        for line in filedata:
            linedata = line.split()
            SrcTestDict[linedata[0]] = linedata[1]
    return SrcTestDict

# function definition of BuildCache
# obj: train_data object containing text dataq
# data_path
# 
def BuildCache(data_obj, data_path, cached_path):
    DetectPath(cached_path)
    contents = os.listdir(cached_path)
    if len(contents) == 0:
        print(u'Creating tokenized data...\n')
        DetectPath(data_path)
        contents = os.listdir(data_path)
        for classname in contents:
            contpath = os.path.join(data_path, \
                                    classname)
            if os.path.isdir(contpath):
                savecontpath = os.path.join(cached_path, \
                                            classname)
                DetectPath(savecontpath)
                files = os.listdir(contpath)
                for filename in files:
                    filepath = os.path.join(contpath, \
                                            filename)
                    if os.path.isfile(filepath):
                        fildata = LoadFile(filepath)
                        res = jieba.cut(fildata)
                        fildata = " ".join(res)
                                                
                        i = data_obj.AddTarget(classname)
                        data_obj.y.append(i)
                        data_obj.file_name.append(filename)
                        data_obj.data.append(fildata)
                        fout = open(os.path.join(savecontpath, \
                                                 filename), 'w')
                        fout.write(fildata)
                        fout.close()
        fout = open(os.path.join(cached_path, u'.class'), 'w')
        for word in data_obj.target_name:
            fout.write(word + '\n')
        fout.close()
        data_obj.AddTarget(other_tag)
    else:
        print(u'Loading cached data...\n')
        classfilpath = os.path.join(cached_path, u'.class')
        if os.path.exists(classfilpath):
            dat = LoadFileSp(classfilpath)
            for cls in dat.split():
                data_obj.AddTarget(cls)
        
        for classname in contents:
            contpath = os.path.join(cached_path, \
                                    classname)
            if os.path.isdir(contpath):
                files = os.listdir(contpath)
                for filename in files:
                    filepath = os.path.join(contpath, \
                                            filename)
                    if os.path.isfile(filepath):
                        fildata = LoadFileSp(filepath)
                        i = data_obj.AddTarget(classname)
                        data_obj.y.append(i)
                        data_obj.file_name.append(filename)
                        data_obj.data.append(fildata)
        data_obj.AddTarget(other_tag)
        
# this function should just work like 
# BuildCache, but it doesn't provide
# support for classification in advance
def BuildTestCache(data_obj, data_path, cached_path, predictDict, trainCls, includeOther):
    DetectPath(cached_path)
    contents = os.listdir(cached_path)
    if len(contents) == 0:
        print(u'Creating tokenized data for test data...\n')
        DetectPath(data_path)
        files = os.listdir(data_path)
        # here we detect test files in the folder
        for filename in files:
            filepath = os.path.join(data_path, \
                                    filename)
            if os.path.isfile(filepath):
                savecontpath = os.path.join(cached_path, \
                                filename)
                fildata = LoadFile(filepath)
                res = jieba.cut(fildata)
                fildata = " ".join(res)
            
                fout = open(savecontpath, 'w')
                fout.write(fildata)
                fout.close()
                
                
                purefilename = os.path.splitext(filename)
                if predictDict.has_key(purefilename[0]):
                    s = predictDict[purefilename[0]]
                    if trainCls.GetTargetId(s) is not None:
                        data_obj.y.append(trainCls.GetTargetId(s))
                    else:
                        if includeOther == False:
                            continue
                        # means other class
                        data_obj.y.append(trainCls.GetTargetId(other_tag))
                    # there is no need to add
                    # preset class id here
                    data_obj.file_name.append(filename)
                    # the data after being processed.
                    data_obj.data.append(fildata)

                    
    else:
        print(u'Loading cached test data...\n')
        for filename in contents:
            filepath = os.path.join(cached_path, filename)
            if os.path.isfile(filepath):
                fildata = LoadFileSp(filepath)
                
                purefilename = os.path.splitext(filename)
                if predictDict.has_key(purefilename[0]):
                    s = predictDict[purefilename[0]]
                    if trainCls.GetTargetId(s) is not None:
                        data_obj.y.append(trainCls.GetTargetId(s))
                    else:
                        if includeOther == False:
                            continue
                        # means other class
                        data_obj.y.append(trainCls.GetTargetId(other_tag))
                data_obj.file_name.append(filename)
                data_obj.data.append(fildata)

train_data = TrainDataType()
valid_data = TrainDataType()
test_data = TrainDataType()
BuildCache(train_data, trainpath, train_cachepath)
BuildCache(valid_data, validpath, valid_cachepath)

print('''Choose a classifer:
0. Naive Bayes
1. SGD (Stochastic gradient descent)
''')
chose = 1
text_clf = None
if chose == 0:
    text_clf = Pipeline([('vect', CountVectorizer()), \
                     ('tfidf', TfidfTransformer()), \
                     ('NBclf', MultinomialNB())])
else:
# the following use SVC model
    text_clf = Pipeline([('vect', CountVectorizer()), \
                         ('tfidf', TfidfTransformer()), \
                         ('NBclf', SGDClassifier(loss='hinge', \
                                   penalty='l2', alpha=1e-3, \
                                   n_iter=5, random_state=42)), \
                         ])

_clf = text_clf.fit(train_data.data, train_data.y)
predicted = text_clf.predict(valid_data.data)

print(np.mean(predicted == valid_data.y))
from sklearn import metrics
print(metrics.classification_report(valid_data.y, \
                              predicted, \
                              None, \
                              valid_data.target_name))

# in the following directions, we want to predict
# the class of test text files.
print('Now deal with test files\n')


test_preDict = LoadSrcFile()
BuildTestCache(test_data, testpath, test_cachepath, \
               test_preDict, train_data, True)
tested = text_clf.predict(test_data.data)
print('Output the test report:\n')
print(metrics.classification_report(test_data.y, \
                              tested, \
                              None, \
                              train_data.target_name))
fres = open(os.path.join(basepath, 'test_data.txt'), 'w')
for index in range(0, len(test_data.y)):
    purfile = os.path.splitext(test_data.file_name[index])
    line = purfile[0] + '\t' + \
                test_preDict[purfile[0]] + '\t' + \
                train_data.target_name[tested[index]] + '\t'
    if tested[index] != test_data.y[index]:
        line += '!'
        if test_data.y[index] == train_data.GetTargetId(other_tag):
            line += '**'
    line += '\n';
    fres.write(line)
fres.close()
# word vectorizer processor
# print('Creating Text Vector for Training data...')
# count_vct = CountVectorizer()
# X_train_counts = count_vct.fit_transform(train_data.data)

# define a sklearn Tfidf transformer
# tfidf_transformer = TfidfTransformer()
# tfidf fit and transform data
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tfidf.shape
# clf = MultinomialNB().fit(X_train_tfidf, train_data.y)


