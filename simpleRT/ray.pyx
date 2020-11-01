# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:10:08 2018

@author: wojci
"""
cimport cython
from cpython cimport bool
from simpleRT.datatypes.MyVec3 cimport MyVec3

import numpy as np
cimport numpy as np

cdef class Ray():

    def __init__(self):
        self.energy = np.array([1, 1, 1, 1, 1, 1])
        self.refl_order = 0
        self.receiver_hit = False
        
        self.position=MyVec3()
        self.direction=MyVec3()
        self.total_dist = 0
        
        self.status = RayStatus.OK
        
    cdef ray_lost(self):
        self.status = RayStatus.LOST
        
    cpdef bool is_lost(self):
        if self.status == RayStatus.LOST:
            return True
        return False
        