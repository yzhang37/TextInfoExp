# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 13:31:02 2017

@author: l
"""

import os;
import sys;
from bs4 import BeautifulSoup
import bs4.element

reload(sys)
sys.setdefaultencoding('utf-8')


basepath = os.getcwd().replace('\\', '/');
finpath = basepath + '/data/computer'
foutpath = basepath + '/output'


def fetchkeywordfromxml(filename):
    kw_list = []
    soup = BeautifulSoup(open(filename), u'html.parser')
    lists = soup.find_all("dt")
    for list in lists:
        if type(list) == bs4.element.Tag:
            if list.string != None:
                keyword = ""
                keyword = list.string.strip().replace(' ', '')
                if keyword.find(u'关键词') == 0:
                    nextitem = list
                    while nextitem.name != u'dd':
                        nextitem = nextitem.next_sibling
                    for a in nextitem.descendants:
                        keyword = a.string
                        if keyword != None and keyword.strip().replace(' ', '') != '':
                            keyword = keyword.strip().replace(' ', '')
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
            s = fetchkeywordfromxml(path)
        break