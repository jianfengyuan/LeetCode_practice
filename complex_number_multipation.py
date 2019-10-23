#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:58:07 2018

@author: jianfengyuan
"""
import re
a = '1+-2i'
pattern = re.compile('(\d)*\+([-]?\d*)i')
print(a)
m = pattern.search(a)
print(m.group(2))