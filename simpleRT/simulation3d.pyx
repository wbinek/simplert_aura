# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:05:10 2018

@author: wojci
"""

import cython
import numpy as np
cimport numpy as np
from cpython cimport bool

import copy
import random as rnd
from rtree import index

from simpleRT.ray cimport Ray
from simpleRT.receiver3d cimport Receiver3D
from simpleRT.source3d cimport Source3D
from simpleRT.model3d cimport Model3D, Face3D
from simpleRT.datatypes.MyVec3 cimport MyVec3
from simpleRT.libmath cimport libmath

cpdef enum ReflectionModel:
    Base = 0
    Phong = 1
    
cpdef enum RaySphereIntersection:
    Full = 0
    Simple = 1
    
cpdef enum RTAlgorithm:
    Basic = 0
    NextEventEstimation = 1
    
cdef class SimulationRT():
    cdef public Model3D model3d
    cdef public Source3D source
    cdef public Receiver3D receiver

    cdef public float c, dv, sim_len
    cdef public list ray_set
    cdef public int no_lost, no_rays, fs, max_refl_ord
    cdef public bool brute_force
    cdef public ReflectionModel reflection_model
    cdef public RaySphereIntersection ray_sphere_int
    cdef public RTAlgorithm rt_algorithm
    cpdef public np.ndarray time, result, result_spatial
    cdef object kdTree

    def __init__(self, source=None, receiver=None, result=None, time = None, fs=44100):
        self.source = source
        self.receiver = receiver
        self.result = result
        self.time = time
        self.fs = fs
                 
        self.c = 343

    def __reduce__(self):
        return self.__class__, (self.source, self.receiver, self.result, self.time, self.fs)
  
    cpdef initialize_rays(self):
        ray_energy = (4*np.pi)/self.no_rays
        ray_position = self.source.position
        
        base_ray = Ray()
        base_ray.position = ray_position
        base_ray.energy = base_ray.energy*ray_energy
        
        self.ray_set=[]
        for i in range(self.no_rays):
            ray = copy.deepcopy(base_ray)
            ray.direction = libmath.random_direction_sphere()
            self.ray_set.append(ray)
            
    cpdef initialize_simulation_parameters(self, 
                                         no_rays=5000, 
                                         fs=44100, 
                                         sim_len=2, 
                                         max_refl_ord=5, 
                                         brute_force=False, 
                                         reflection_model=ReflectionModel.Base, 
                                         ray_sphere_intersection=RaySphereIntersection.Full,
                                         rt_algorithm = RTAlgorithm.Basic):
        self.no_rays = no_rays
        self.fs = fs
        self.sim_len = sim_len
        self.max_refl_ord = max_refl_ord
        self.brute_force = brute_force
        self.reflection_model = reflection_model
        self.ray_sphere_int = ray_sphere_intersection
        self.rt_algorithm = rt_algorithm
        
        self.dv = self.c/self.fs 
        self.time = np.arange(0,sim_len,1/fs)
        
        if not brute_force:
            self.initialize_rtree()

    cdef list rtree_generator(self):
        cdef int i
        cdef Face3D face
        cdef list objects=[]
        for i,face in enumerate(self.model3d.faces):
            v0 = self.model3d.vertices[face.vertices[0]]
            v1 = self.model3d.vertices[face.vertices[1]]
            v2 = self.model3d.vertices[face.vertices[2]]
            
            v0=np.array(v0.asArray())
            v1=np.array(v1.asArray())
            v2=np.array(v2.asArray())
                        
            stack = np.vstack((v0,v1,v2))
            mintri = np.min(stack,axis=0)
            maxtri = np.max(stack,axis=0)
            objects.append((i, np.hstack((mintri,maxtri)),None))
        return objects
        
    cdef initialize_rtree(self):
        p = index.Property()
        p.dimension = 3
        p.fill_factor=0.05
        p.leaf_capacity = 50
        
        self.kdTree = None
        objects = self.rtree_generator()
        self.kdTree = index.Index(objects,properties=p)
        
    cdef tuple brute_force_triangle_intersection(self, Ray ray, Face3D[:] faces, Face3D last_face, list indexes):
        cdef float tmin = np.Inf
        cdef float tp = -1
        cdef Face3D hit_face = None

        cdef int i, index
        cdef int lfaces = len(indexes)
        for i in range(lfaces):
            index = indexes[i]
            if faces[index] != last_face:            
                tp = libmath.ray_triangle_intersection(ray.position, ray.direction, 
                    self.model3d.vertices[faces[index].vertices[0]], 
                    self.model3d.vertices[faces[index].vertices[1]], 
                    self.model3d.vertices[faces[index].vertices[2]])
                           
                if(tp>0 and tp<tmin):
                    tmin = tp
                    hit_face = faces[index]

        return tmin, hit_face
    
    cdef tuple kd_tree_triang_intersection(self, Ray ray, Face3D last_face):
        cdef float tmin
        cdef Face3D hit_face = None
        
        cdef MyVec3 rpos = ray.position
        cdef MyVec3 rdir = ray.direction

        cdef int step=10
        cdef MyVec3 start, end, minpos, maxpos
        cdef list intersections
        
        cdef int i
        for i in range(10):
            start = rpos.add_vec(rdir.mul(i*step))
            end = rpos.add_vec(rdir.mul((i+1)*step))          
            minpos = start.minv(end)
            maxpos = start.maxv(end)
            
            intersections = list(map(int,self.kdTree.intersection((minpos.x,minpos.y,minpos.z,maxpos.x,maxpos.y,maxpos.z))))

            if intersections:
                tmin, hit_face = self.brute_force_triangle_intersection(ray, self.model3d.faces, last_face, intersections)

            if hit_face is not None:
                return tmin, hit_face

        return tmin, hit_face
        
    cdef run_ray_basic(self,Ray ray):
        cdef Face3D last_face = None
        cdef float t1,t2
        cdef float tmin
        cdef Face3D hit_face

        cdef int entry_sample, exit_sample, ds
        cdef float dif_coeff,  
        cdef np.ndarray abs_coeff
        for i in range(self.max_refl_ord):
                        
            #Find colision with receiver 
            t1 = np.Inf
            rec_hit = libmath.ray_sphere_intersection(ray.position, ray.direction ,self.receiver.position, self.receiver.radius)
            if rec_hit:
                t1 = rec_hit[0] #entry point
                t2 = rec_hit[1] #out_point
            
            #Find colision with model geometry
            if self.brute_force:
                tmin, hit_face = self.brute_force_triangle_intersection(ray, self.model3d.faces, last_face, list(range(len(self.model3d.faces))))
            else:         
                tmin, hit_face = self.kd_tree_triang_intersection(ray, last_face)
            
            
            # Test if model hit any face, if not it is lost
            if hit_face is None:
                ray.ray_lost()
                self.no_lost+=1
                return
            
            # Test if colision with model is after colision with receiver
            if tmin>t1 and t1>0:              
                entry_sample = int(min((ray.total_dist + t1)/self.dv, self.fs*self.sim_len))
                exit_sample = int(min((ray.total_dist + t2)/self.dv + 1, self.fs*self.sim_len))
                ds = exit_sample-entry_sample
                
                # If simlified intersection model is selected - using only meadle intersection sample
                if self.ray_sphere_int == RaySphereIntersection.Simple:
                    entry_sample = int((exit_sample-entry_sample)/2)
                    exit_sample = entry_sample+1
                    ds=1
                
                # add energy to result
                self.result[:,entry_sample:exit_sample] += np.transpose(np.tile(ray.energy,(ds,1)))
                #print((np.tile(ray.direction.asArray(),(6,1)).T * ray.energy).T)
                self.result_spatial[:,:,entry_sample:exit_sample] += np.tile(((np.tile(ray.direction.asArray(),(6,1)).T * ray.energy).T)[:,:,None],(1,1,ds))
                ray.receiver_hit = True
                
            # Move ray to face hit position, update energy
            dif_coeff = self.model3d.materials[hit_face.mat_name]['diffusion_coeff']
            abs_coeff = self.model3d.materials[hit_face.mat_name]['absorption_coeff']
            last_face = hit_face
            ray.position = libmath.move_ray(ray.position, ray.direction, tmin)
            ray.total_dist += tmin
            ray.energy*=(1-abs_coeff)
            
            # calculate reflection based on chosen reflection model
            if self.reflection_model == ReflectionModel.Base:
                ray.direction = libmath.base_reflection(ray.direction, self.model3d.normals[hit_face.normal_idx], dif_coeff)
            elif self.reflection_model == ReflectionModel.Phong:
                ray.direction = libmath.phong_reflection(ray.direction, self.model3d.normals[hit_face.normal_idx], dif_coeff)
            ray.refl_order+=1
            
    cdef run_ray_nee(self,ray):
        raise NotImplementedError

    cdef void run_ray(self, ray):
        if self.rt_algorithm == RTAlgorithm.Basic:
            self.run_ray_basic(ray)
        elif self.rt_algorithm == RTAlgorithm.NextEventEstimation:
            self.run_ray_nee(ray)

    cpdef run_simulation(self, callback = None, hold_result=False):
        if not hold_result:
            self.result = np.zeros((6, len(self.time)))
            self.result_spatial = np.zeros((6, 3, len(self.time)))

        self.no_lost=0
        cdef int i=0
        for ray in self.ray_set:
            self.run_ray(ray)
            i=i+1
            if callback:
                callback(i)

     
    @staticmethod
    def run_tests():  
        ray = Ray()
        rpos = MyVec3()
        rpos.x, rpos.y, rpos.z = 0,0,0
        ray.position=rpos
        
        rdir = MyVec3()
        rdir.x, rdir.y, rdir.z = 1,-0.1,0
        ray.direction=rdir
        
        v0 = MyVec3()
        v1 = MyVec3()
        v2 = MyVec3()
        v0.fromArray([2,-5,-5])
        v1.fromArray([2,5,-5])
        v2.fromArray([2,-5,5])
        dist = libmath.ray_triangle_intersection(ray.position, ray.direction, v0, v1, v2)
        print(dist)
        
        sphere = Receiver3D()
        sphere.position.fromArray([1, 0, 0]) 
        sphere.radius = 0.2
        res = libmath.ray_sphere_intersection(ray.position, ray.direction, sphere.position, sphere.radius)
        print(res)
        
        normal = MyVec3()
        normal.fromArray([-1,0,0])
        base = libmath.base_reflection(ray.direction, normal,0)
        specular = libmath.specular_reflection(ray.direction, normal)
        lambert = libmath.nexp_reflection(ray.direction, normal,5000)
        print(specular.asArray())
        print(lambert.asArray())