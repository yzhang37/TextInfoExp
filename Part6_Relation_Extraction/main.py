#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 03:14:10 2017

@author: X230大青椒
"""

# -*- coding: utf-8 -*-

import sys
import os
import langconv
import traceback
import cPickle
from gensim.models import Word2Vec
import jieba
reload(sys)
sys.setdefaultencoding('utf-8')

basepath = os.getcwdu()
data_path = os.path.join(basepath, 'data')
cached_path = os.path.join(basepath, 'cached')
# 全局声明数据文件和缓存文件
people_filename = os.path.join(data_path, 'people.txt')
people_cached = os.path.join(cached_path, 'people.txt')

# jieba处理程序
jieba.load_userdict(os.path.join(data_path, 'people.txt'))

# 繁体转简体程序
class ChineseConv(object):
    def __init__(self):
        self._simpconv = langconv.Converter('zh-hans')
        self._tradconv = langconv.Converter('zh-hant')
        
    def s2t(self, sentence):
        return self._tradconv.convert(sentence)
    
    def t2s(self, sentence):
        return self._simpconv.convert(sentence)

cc = ChineseConv()

def simp_chinese(path):
    file_idx = 0
    for base, dirs, files in os.walk(path):
        for file in files:
            if file != u'.DS_Store' and file != u'Desktop.ini':
                try:
                    fin = open(os.path.join(base, file), 'r')
                    msg = fin.read().decode('utf-8', 'ignore')
                    fin.close()
                    
                    msg = cc.t2s(msg)
                    
                    fout = open(os.path.join(base, file), 'w')
                    fout.write(msg)
                except:
                    traceback.print_exc()
                finally:
                    fout.close()
                    file_idx += 1
                    if file_idx % 100 == 0:
                        print(u'处理%d个文件' % file_idx)
    print(u'处理%d个文件' % file_idx)

def get_people():
    try:
        fp_in = open(people_filename, 'r')
        dat = {}
        rev_dat = {}
        r_data = [dat, rev_dat]
        for line in fp_in:
            dat.setdefault(line.strip(), len(dat))
        fp_in.close()
        for k, v in dat.items():
            rev_dat[v] = k
        fout = open(people_cached, 'wb')
        cPickle.dump(fout, r_data)
        fout.close()
        return r_data
    except:
        traceback.print_exc()

def get_relation(filename):
    try:
        fin = open(filename, 'r')
        for line in fin:
            n1, n2, rel = line.split('\t', 2)
            
        fin.close()
    except:
        traceback.print_exc()

if __name__ == '__main__':

        
    