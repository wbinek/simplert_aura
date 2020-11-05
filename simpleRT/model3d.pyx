# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:10:34 2018

@author: wojci
"""
import cython
cimport numpy as np
import numpy as np
from cpython.array cimport array, clone

from simpleRT.datatypes.MyVec3 cimport MyVec3
from simpleRT.helpers.objloader import OBJ

cdef class Face3D:
    
    def __init__(self, tuple objFace = None):   
        if objFace is None:
            self.normal_idx = -1
            self.mat_name = None
            return
        else:
            if(len(objFace[0]))>3:
                raise NotImplementedError('Face is not Triangulated, not supported yet')
            
            self.vertices[0] = int(objFace[0][0])-1
            self.vertices[1] = int(objFace[0][1])-1
            self.vertices[2] = int(objFace[0][2])-1
            self.normal_idx = objFace[1][1]-1
            self.mat_name = objFace[3]
            
cdef class Model3D:
    
    def __init__(self):
        self.materials = None
        
    cpdef loadOBJ(self, str path):
        raw_model = OBJ(path)
        
        self._faces = np.array([Face3D(face) for face in raw_model.faces])
        self.faces = self._faces

        cdef int i

        self._vertices = np.ndarray((len(raw_model.vertices),),dtype=MyVec3)
        self.vertices = self._vertices
        for i in range(len(raw_model.vertices)):
            v = MyVec3()
            v.x, v.y, v.z = raw_model.vertices[i][0], raw_model.vertices[i][1], raw_model.vertices[i][2]
            self.vertices[i]=v

        self._normals = np.ndarray((len(raw_model.normals),),dtype=MyVec3)
        self.normals = self._normals
        cdef np.ndarray norm
        for i in range(len(raw_model.normals)): 
            norm = np.array(raw_model.normals[i])/np.linalg.norm(np.array(raw_model.normals[i]))
            n = MyVec3()
            n.x, n.y, n.z = norm[0], norm[1], norm[2]
            self.normals[i] = n
        #self.normals = [np.array(normal)/np.linalg.norm(np.array(normal)) for normal in raw_model.normals]
        self.materials = raw_model.mtl
        
        for name, mat in self.materials.items():
            mat['absorption_coeff'] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            mat['diffusion_coeff'] = 0.1
            col = (np.array(mat['Kd'])*255).astype(int)
            mat['kd_hex'] =  '#%02x%02x%02x' % (col[0], col[1], col[2])
      
        
        
        