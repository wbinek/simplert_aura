# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:06:35 2018

@author: wojci
"""

import cython
from simpleRT.datatypes.MyVec3 cimport MyVec3

cdef class Source3D():
    cdef public MyVec3 position