# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:10:34 2018

@author: wojci
"""
import cython
from simpleRT.datatypes.MyVec3 cimport MyVec3
cimport numpy as np

from simpleRT.helpers.objloader import OBJ
import numpy as np

cdef class Face3D:
    
    def __init__(self, tuple objFace = None):   
        if objFace is None:
            self.vertices = []
            self.normal_idx = -1
            self.mat_name = None
            return
        else:
            if(len(objFace[0]))>3:
                raise NotImplementedError('Face is not Triangulated, not supported yet')
            
            self.vertices = [v-1 for v in objFace[0]]
            self.normal_idx = objFace[1][1]-1
            self.mat_name = objFace[3]
            

cdef class Model3D:
    
    def __init__(self):
        self.vertices = []
        self.faces = None
        self.normals = []
        
        self.materials = None
        
    cpdef loadOBJ(self, str path):
        raw_model = OBJ(path)
        
        self.faces = np.array([Face3D(face) for face in raw_model.faces])
        for verti in raw_model.vertices:
            v = MyVec3()
            v.x, v.y, v.z = verti[0], verti[1], verti[2]
            self.vertices.append(v)
        #self.vertices = np.array(raw_model.vertices)
        for norm in raw_model.normals:
            norm = np.array(norm)/np.linalg.norm(np.array(norm))
            n = MyVec3()
            n.x, n.y, n.z = norm[0], norm[1], norm[2]
            self.normals.append(n)
        #self.normals = [np.array(normal)/np.linalg.norm(np.array(normal)) for normal in raw_model.normals]
        self.materials = raw_model.mtl
        
        for name, mat in self.materials.items():
            mat['absorption_coeff'] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            mat['diffusion_coeff'] = 0.1
            col = (np.array(mat['Kd'])*255).astype(int)
            mat['kd_hex'] =  '#%02x%02x%02x' % (col[0], col[1], col[2])
      
        
        
        