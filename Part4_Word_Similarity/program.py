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
    #try:
    if isinstance(path, unicode) or isinstance(path, str):
        fin = codecs.open(path, mode='r',encoding='utf-8')
        for line in fin:
            # tag, attr = line.split(u' ', 1)
            tag, attr = line.split(u' ', 1)
            agg = Aggregation()
            agg.tag = tag
            
            for word in attr.split(u' '):
                # word=word.encode('utf-8')
                word = word.strip()
                agg.words[word] = 1
                WordDict[word] = agg
            
            AggList.append(agg)
            # if agg.tag == 'Ga01A01=':
            #     print 'A'
            SetParent(agg, WordDict)
        return True, AggList, WordDict
    else:
        raise Exception(u'Invalid Path type')
    #except Exception:
    #    logging.error('Loading Cilin file failed.')
    #    return False, None, None
    
def CalcDist(word1, word2, WordDict):
    a = 0.65
    b = 0.8
    c = 0.9
    d = 0.96
    e = 0.5
    f = 0.1
    
    try:
        path1 = WordDict[word1].tag
        path2 = WordDict[word2].tag
        
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
    
def PrintWordsList(agg):
    for (word, t) in agg.words.items():
        print word

if __name__ == '__main__':            
   state, AggList, WordDict = LoadCilin(dataset)
   if state == True:
       a = u'娘胎'
       b = u'穴位'
       state, valu = CalcDist(a, b, WordDict)
       if state == True:
           print valu
      