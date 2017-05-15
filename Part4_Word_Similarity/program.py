#!/usr/bin/env python2
# encoding: utf-8
"""
Created on Mon May  8 10:34:48 2017

@author: X230_NZ666
@e-mail: Unicorn_X230@outlook.com
"""

import sys
import logging
import os
import codecs
import re
import string
import numpy as np

reload(sys);
sys.setdefaultencoding('utf-8')

pwd = os.getcwdu()
dataset = os.path.join(pwd, u'data/dataset.txt')

class Node(object):
    def __init__(self):
        self.parent = None
        self.childs = []
        self.tag = ''
        
class Aggregation(Node):
    def __init__(self):
        self.words = {}        

def SetParent(aggc, WordDict):
    length = len(aggc.tag)
    if length == 0:
        return
    else:
        if length >= 6:
            s = aggc.tag[0:5]
        elif length == 5:
            s = aggc.tag[0:4]
        elif length >= 3:
            s = aggc.tag[0:2]
        elif length == 2:
            s = aggc.tag[0:1]
        else:
            return
            
        agg = None
        if WordDict.has_key(s):
            agg = WordDict[s]
        elif len(s) > 0:
            agg = Node()
            agg.tag = s
            WordDict[s] = agg
            SetParent(agg, WordDict)
            
        if agg != None:
            agg.childs.append(aggc)
            aggc.parent = agg

def LoadCilin(path):
    AggList = []
    WordDict = {}
    try:
        if isinstance(path, unicode) or isinstance(path, str):
            fin = codecs.open(path, mode='r',encoding='utf-8')
            for line in fin:
                tag, attr = line.split(u' ', 1)
                agg = Aggregation()
                agg.tag = tag
                
                for word in attr.split(u' '):
                    word = word.strip()
                    agg.words[word] = 1
                    
                    # 检测是否有重复的单词映射。如果有，转换为list存储
                    if word == u'人民':
                        pass
                    if WordDict.has_key(word):
                        if isinstance(WordDict[word], list):
                            WordDict[word].append(agg)
                        else:
                            newList = [WordDict[word], agg]
                            WordDict[word] = newList
                    else:
                        WordDict[word] = agg
                
                AggList.append(agg)
                # if agg.tag == 'Ga01A01=':
                #     print 'A'
                SetParent(agg, WordDict)
            return True, AggList, WordDict
        else:
            raise Exception(u'Invalid Path type')
    except Exception:
        logging.error('Loading Cilin file failed.')
        return False, None, None
    
def __calcDist(path1, path2, WordDict):
    a = 0.65
    b = 0.8
    c = 0.9
    d = 0.96
    e = 0.5
    f = 0.1
    
    try:        
        # find the same 
        idx = -1
        for i in range(0, 8):
            if path1[i] != path2[i]:
                break
            idx = i
        
        sameroots = ''
        if idx >= 6:
            x = 5
            sameroots = path1[0:7]
        elif idx >= 4:
            x = 4
            sameroots = path1[0:5]
        elif idx >= 3:
            x = 3
            sameroots = path1[0:4]
        elif idx >= 1:
            x = 2
            sameroots = path1[0:2]
        elif idx >= 0:
            x = 1
            sameroots = path1[0:1]
        
        
        if idx == -1: # no tier
            return True, f
        
        k = None
        if x == 5: # that's just the same word
            if path1[-1] == '=': # similiar word
                return True, 1
            elif path1[-1] == '#': # same category
                return True, e
            else:
                return True, 0 # '@', single word
        else:
            k = 2 * (5 - x)
        
        # calculate n
        agg = WordDict[sameroots]
        n = len(agg.childs)        
        r = np.cos(float(n) * np.pi / 180) * (n - k + 1) / float(n)
        if x == 1: # tier 1 same
            return True, a * r
        elif x == 2: # tier 2 same
            return True, b * r
        elif x == 3: # tier 3 same
            return True, c * r
        elif x == 4: # tier 4 same
            return True, d * r
    except Exception:
        logging.error('Computing distance failed.')
        return False, None

def CalcDist(word1, word2, WordDict):
    a = WordDict[word1]
    b = WordDict[word2]
    if isinstance(a, Aggregation) and isinstance(b, Aggregation):
        return __calcDist(a.tag, b.tag, WordDict)
    
    max_valu = 0.0
    if not isinstance(a, Aggregation) and isinstance(b, Aggregation):
        for a_agg in a:
            a_tag = a_agg.tag
            max_valu = max(max_valu, __calcDist(a_tag, b.tag, WordDict))
    elif isinstance(a, Aggregation) and not isinstance(b, Aggregation):
        for b_agg in b:
            b_tag = b_agg.tag
            max_valu = max(max_valu, __calcDist(a.tag, b_tag, WordDict))
    else:
        for a_agg in a:
            a_tag = a_agg.tag
            for b_agg in b:
                b_tag = b_agg.tag
                max_valu = max(max_valu, __calcDist(a_tag, b_tag, WordDict))
    return max_valu
    
def PrintWordsList(agg):
    for (word, t) in agg.words.items():
        print word

if __name__ == '__main__':            
   state, AggList, WordDict = LoadCilin(dataset)
   if state == True:
       a = u'人民'
       b = u'同志'
       state, valu = CalcDist(a, b, WordDict)
       if state == True:
           print valu
      