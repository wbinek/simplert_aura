# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:20:51 2018

@author: wojci
"""
import pickle
import pandas as pd
import numpy as np
freq_idx = {'125Hz':0, '250Hz':1, '500Hz':2, '1000Hz':3, '2000Hz':4, '4000Hz':5}

def load_previous_materials(model3D, filename):  
    df = pickle.load( open( filename, "rb" ) )
    for i, row in df.iterrows():
        for j, column in row.iteritems():
            if j == "Diffusion":
                model3D.materials[row.name]['diffusion_coeff'] = column
            else:
                model3D.materials[row.name]['absorption_coeff'][freq_idx[j]] = column     
    return model3D
            
def save_materials(model3D, filename):
    df = pd.DataFrame([np.append(mat['absorption_coeff'],mat['diffusion_coeff']) 
                       for name, mat in model3D.materials.items()], 
                      columns = ['125Hz', '250Hz', '500Hz', '1000Hz', '2000Hz', '4000Hz', 'Diffusion'], 
                      index = model3D.materials.keys())
    pickle.dump(df, open( filename, "wb" ) )