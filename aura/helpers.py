# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:49:36 2018

@author: wojci
"""
import numpy as np
from scipy.signal import find_peaks

def peak_detection(result):
    peaks, _ = find_peaks(result, height=0)
    pf_res = np.zeros(len(result))
    pf_res[peaks] = result[peaks]
    return pf_res