# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:59:58 2018

@author: wojci
"""
import random as rnd
import numpy as np
cimport numpy as np

cimport cython 
from libc.math cimport sin, cos, abs, pi, sqrt, acos, pow
from cpython cimport bool

from simpleRT.datatypes.MyVec3 cimport MyVec3
from simpleRT.datatypes.MyArray3 cimport MyArray3

cdef MyArray3 I
global I
I = MyArray3()
I.eye()

cdef np.ndarray calculate_rotation_matrix_leg(MyVec3 origin, MyVec3 target):    
    cdef np.ndarray I = np.eye(3)   
    cdef MyVec3 cross = origin.cross(target)

    cdef np.ndarray v, rot_mat;
    
    if(cross.vectorLength()>0.0001):
        v = np.matrix([[0, -cross.z, cross.y],
                      [cross.z, 0, -cross.x],
                      [-cross.y, cross.x, 0]])
    
    
        rot_mat = I + v + v*v*(1/(1+origin.dot(target)))
       
    else:
        rot_mat = -I

    return rot_mat 

cdef MyArray3 calculate_rotation_matrix(MyVec3 origin, MyVec3 target):
    global I
    cdef MyVec3 cross = origin.cross(target)

    cdef MyArray3 v, rot_mat;
    
    if(cross.vectorLength()>0.0001):
        v = MyArray3()
        v.set_matrix(0, -cross.z, cross.y,
                    cross.z, 0, -cross.x,
                    -cross.y, cross.x, 0)
      
        # rot_mat = I + v + v*v*(1/(1+origin.dot(target)))
        rot_mat = (I.add_array(v)).add_array((v.mul_array(v)).mul_float(1/(1+origin.dot(target))))
     
    else:
        rot_mat = -I

    return rot_mat

cdef float ray_triangle_intersection(MyVec3 rpos, MyVec3 rdir, MyVec3 v0, MyVec3 v1, MyVec3 v2):
    cdef MyVec3 v1v0 = v1.sub_vec(v0)
    cdef MyVec3 v2v0 = v2.sub_vec(v0)
    cdef MyVec3 rov0 = rpos.sub_vec(v0)
    
    cdef MyVec3 n = v1v0.cross(v2v0)
    cdef MyVec3 q = rov0.cross(rdir)
    cdef float angle = rdir.dot(n)
    if abs(angle) < 0.000001:
        return -1.
    cdef float d = 1./angle
    cdef float u = d*(q.neg()).dot(v2v0)
    cdef float v = d*q.dot(v1v0)
    cdef float t = d*(n.neg()).dot(rov0)
    
    #t = min(u, min(v, min(1.0-u-v, t)));
    if u<0.0 or u>1.0 or v<0.0 or (u+v)>1.0:
        return -1.

    return t

cdef inline MyVec3 move_ray(MyVec3 ray_position, MyVec3 ray_direction, float time):
    return ray_position.add_vec(ray_direction.mul(time))

cdef list ray_sphere_intersection(MyVec3 ray_pos, MyVec3 ray_dir, MyVec3 sphere_pos, float sphere_rad):
    cdef MyVec3 oc = ray_pos.sub_vec(sphere_pos)
    cdef float b = oc.dot(ray_dir)
    cdef float c = oc.dot(oc) - sphere_rad*sphere_rad
    cdef float h = b*b-c
    if h<=0:
        return []
    return [-b-sqrt(h), -b+sqrt(h)]

cdef MyVec3 random_direction_sphere():
    cdef float r1 = rnd.random()
    cdef float r2 = rnd.random()
    
    cdef float x = 2*np.cos(2*np.pi*r1)*np.sqrt(r2*(1-r2))
    cdef float y = 2*np.sin(2*np.pi*r1)*np.sqrt(r2*(1-r2))
    cdef float z = 1-2*r2
    
    cdef MyVec3 dir = MyVec3()
    dir.x, dir.y, dir.z = x, y, z
      
    return dir

cdef MyVec3 specular_reflection(MyVec3 ray_direction, MyVec3 normal): 
    cdef MyVec3 v1 = normal.mul(normal.dot(ray_direction))
    cdef MyVec3 reflected = ray_direction.sub_vec(v1.mul(2))
    
    reflected = reflected.normalize()
    return reflected

cdef MyVec3 base_nexp_reflection(int n):
    cdef float r1 = rnd.random()
    cdef float r2 = rnd.random()
                
    cdef MyVec3 d = MyVec3()  
    cdef float x = cos(2.*pi*r1)*sqrt(1.-pow(r2, 2./(n+1.)))
    cdef float y = sin(2.*pi*r1)*sqrt(1.-pow(r2, 2./(n+1.)))
    cdef float z = pow(r2, 1./(n+1.))
        
    d.x, d.y, d.z = x, y, z
    return d

cdef MyVec3 nexp_reflection(MyVec3 ray_direction, MyVec3 normal, int n):
    
    if ray_direction.dot(normal)>0:
        normal = normal.neg()

    cdef MyVec3 specular = specular_reflection(ray_direction, normal)  
    cdef MyVec3 base_norm = MyVec3()
    base_norm.x, base_norm.y, base_norm.z = 0., 0., 1.
    
    cdef MyArray3 rot_matrix = calculate_rotation_matrix(base_norm, specular)
    cdef MyVec3 base_refl = base_nexp_reflection(n)

    cdef MyVec3 ref = rot_matrix.mul_vec(base_refl)
    ref = ref.normalize()
    
    if ref.dot(normal)<=0:
        ref = nexp_reflection(ray_direction, normal, n)
    return ref

cdef MyVec3 base_reflection(ray, MyVec3 normal, float diffusion):
    cdef float r1 = rnd.random()
    if r1>diffusion:
        return specular_reflection(ray, normal)
    else:
        return nexp_reflection(ray, normal, 1)
    
cdef MyVec3 phong_reflection(MyVec3 ray_direction, MyVec3 normal, float diffusion):
    cdef float r1 = rnd.random()
    cdef int n = int(pow(10, -1.7234170470604733*diffusion + 2.6245274102195886))
    if r1>diffusion:
        return nexp_reflection(ray_direction, normal, n)
    else:
        return nexp_reflection(ray_direction, normal, 1)
    
        
