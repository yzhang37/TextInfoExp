#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
import logging
# 本地文件，用于中文简体和繁体互相转换使用
import langconv
reload(sys)
sys.setdefaultencoding('utf-8')
import traceback

base_path = os.getcwdu()
data_path = os.path.join(base_path, u'htl')

class ChineseConv(object):
    def __init__(self):
        self._simpconv = langconv.Converter('zh-hans')
        self._tradconv = langconv.Converter('zh-hant')
        
    def s2t(self, sentence):
        return self._tradconv.convert(sentence)
    
    def t2s(self, sentence):
        return self._simpconv.convert(sentence)

#  定义一个全局的中文简体/翻译转换工具    
cc = ChineseConv()

file_idx = 0
for base, dirs, files in os.walk(data_path):
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