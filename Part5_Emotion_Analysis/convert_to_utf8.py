#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
import logging
import cchardet
reload(sys)
sys.setdefaultencoding('utf-8')

base_path = os.getcwdu()
data_path = os.path.join(base_path, u'htl')

def detnconv(path):
    if os.path.isfile(path):
        fin = open(path, 'r')
        msg = fin.read()
        fin.close()
        res = cchardet.detect(msg)
        encoding = res['encoding']
        try:
            if encoding.strip().lower() == 'utf-8':
                return
            new_msg = msg.decode(res['encoding'], 'ignore')
            fin = open(path ,'w')
            fin.write(new_msg.encode('utf-8'))
            fin.close()
        except:
            logging.error(path)

file_idx = 0    
for base, dirs, files in os.walk(data_path):
    for file in files:
        if file != u'.DS_Store' and file != u'Desktop.ini':
            detnconv(os.path.join(base, file))
            file_idx += 1
            if file_idx % 100 == 0:
                print(u'处理 %d 个文件' % file_idx)
print(u'共计处理 %d 个文件' % file_idx)
