# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:59:58 2018

@author: wojci
"""
cimport numpy as np
cimport cython

cimport cython 
from cpython cimport bool

from simpleRT.datatypes.MyVec3 cimport MyVec3
from simpleRT.datatypes.MyArray3 cimport MyArray3

cdef np.ndarray calculate_rotation_matrix_leg(MyVec3 origin, MyVec3 target)

cdef MyArray3 calculate_rotation_matrix(MyVec3 origin, MyVec3 target)

cdef float ray_triangle_intersection(MyVec3 rpos, MyVec3 rdir, MyVec3 v0, MyVec3 v1, MyVec3 v2)

cdef MyVec3 move_ray(MyVec3 ray_position, MyVec3 ray_direction, float time)

cdef list ray_sphere_intersection(MyVec3 ray_pos, MyVec3 ray_dir, MyVec3 sphere_pos, float sphere_rad)

cdef MyVec3 random_direction_sphere()

cdef MyVec3 specular_reflection(MyVec3 ray_direction, MyVec3 normal)

cdef MyVec3 base_nexp_reflection(int n)

cdef MyVec3 nexp_reflection(MyVec3 ray_direction, MyVec3 normal, int n)

cdef MyVec3 base_reflection(ray, MyVec3 normal, float diffusion)
    
cdef MyVec3 phong_reflection(MyVec3 ray_direction, MyVec3 normal, float diffusion)
    
        
