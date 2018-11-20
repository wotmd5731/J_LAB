# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:43:07 2018

@author: JAE
"""

from ctypes import *
 
STD_OUTPUT_HANDLE = -11
 
class COORD(Structure):
    pass
 
COORD._fields_ = [("X", c_short), ("Y", c_short)]
 
def print_at(r, c, s):
    h = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    windll.kernel32.SetConsoleCursorPosition(h, COORD(c, r))
 
    c = s.encode("windows-1252")
    windll.kernel32.WriteConsoleA(h, c_char_p(c), len(c), None, None)

import time
import os
os.system('cls')

for x in range(10):
    for y in range(10):
        time.sleep(1)
        print_at(x, y, "   Hello   ")
