# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:49:36 2018

@author: wojci
"""
import numpy as np
from scipy.signal import find_peaks
from math import atan2,acos,sqrt,pi

def peak_detection(result):
    peaks, _ = find_peaks(result, height=0)
    pf_res = np.zeros(len(result))
    pf_res[peaks] = result[peaks]
    return pf_res

def cart2sph(x,y,z):
    r=sqrt(x**2+y**2+z**2)
    if r==0:
        return [0,0,0]
    azimuth=atan2(y,x)
    elevation=acos(z/r)-pi/2
    return [azimuth,elevation,r]

def rotation_relative2source(dirs,source_vector,azimuth=0,elevation=0):
    directions = dirs - source_vector[:,None].T
    
    directions[:,1] = directions[:,1]-elevation
    directions[:,1][directions[:,1]>90] = 180 - directions[:,1][directions[:,1]>90]
    directions[:,0][directions[:,0]>90] = directions[:,0][directions[:,0]>90] + 180
    
    directions[:,1][directions[:,1]<-90] = -180 - directions[:,1][directions[:,1]<-90]
    directions[:,0][directions[:,0]<-90] = directions[:,0][directions[:,0]<-90] + 180
    
    directions[:,0] = directions[:,0]-azimuth
    directions[:,0] = directions[:,0]%360
    
    return directions
    
    
    
    
    