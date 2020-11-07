#%%
#%matplotlib ipympl

import matplotlib.pyplot as plt
import numpy as np
#from ipywidgets import interact, widgets
#from IPython import display
#import qgrid
import pandas as pd
import math
import pickle
import os
import time
import cProfile

#%%
from simpleRT import Model3D

folder = 'modele'
basename = 'Tarnow-Jaskolka'
filename = basename + '.obj'
model3D = Model3D()
os.chdir(folder)
model3D.loadOBJ(filename)
os.chdir('..')
    
#%%
from simpleRT import Source3D, SimulationRT, Receiver3D
from simpleRT.datatypes.MyVec3 import MyVec3
from simpleRT.helpers import matloader as mat

simulation = SimulationRT()
simulation.model3d = model3D

source = Source3D()
simulation.source = source          
spos = MyVec3()
spos.fromArray([10, 1.5, -15.3])
source.position = spos

receiver = Receiver3D()
simulation.receiver = receiver
rpos = MyVec3()
rpos.fromArray([15,1.2,-19])
receiver.position = rpos

rec_rad=0.3
simulation.receiver.radius = rec_rad

model3D = mat.load_previous_materials(model3D, os.path.join(folder,basename+".mat"))
#mat.save_materials(model3D, basename+".mat")

#%%
from simpleRT import ReflectionModel, RaySphereIntersection

no_rays = 2000
fs = 44100
sim_length = 2
max_reflection_order = 30
brute_force = False
profile = False
reflection_model = ReflectionModel.Phong
intersection_model = RaySphereIntersection.Full

def callback(no_ray):
    if(no_ray % 100)==0:
        print(no_ray)

def run_simulations(sim):
    sim.initialize_simulation_parameters(no_rays, fs, sim_length, max_reflection_order, brute_force, reflection_model, intersection_model)
    sim.initialize_rays() #create array of rays, set ray energy and position in the source, generate random ray direction
    sim.run_simulation(callback)

start = time.time()
if profile:
    with cProfile.Profile() as pr:
        run_simulations(simulation)
        pr.dump_stats('profile_stats.prof')
else:
    run_simulations(simulation)

end = time.time()
print(simulation.no_lost)
print(end - start)

plt.figure()
plt.plot(simulation.time,simulation.result[0])

plt.figure()
zeros = np.zeros(len(simulation.result[0]))
Lp = 10*np.log10(simulation.result[0]/10e-12)
Lp[Lp==-np.inf]=0
plt.plot(simulation.time,Lp)

plt.show()

#%%
#pickle.dump( simulation, open( os.path.join(folder,basename+str(no_rays)+".res"), "wb" ))

# #%%
# if 'simulation' in locals():
#     simulation = pickle.load( open( os.path.join(folder,basename+str(no_rays)+".res", "rb" )) )

# #%%
# from scipy.signal import find_peaks
# from pyfilterbank import FractionalOctaveFilterbank

# def peak_detection(result):
#     peaks, _ = find_peaks(result, height=0)
#     pf_res = np.zeros(len(result))
#     pf_res[peaks] = result[peaks]
#     return pf_res

# def filter_octave(data, ofb, idx):
#     y, states = ofb.filter(data**(1/2))
#     return y[:,idx]

# #Initialize octave filter bank
# ofb = FractionalOctaveFilterbank(fs, order=4,nth_oct=1,  start_band=-3, end_band=2)
# print(ofb.center_frequencies)

# #Variable creation
# pf_filter = np.empty(simulation.result.T.shape, dtype=float)
# peak_det = np.empty(simulation.result.T.shape, dtype=float)

# #For each freq range detect peaks anf filter the echogram
# for idx in range(len(simulation.result)):
#     peak_det[:,idx] = peak_detection(simulation.result[idx])
#     pf_filter[:,idx] = filter_octave(peak_det[:,idx], ofb, idx)

# ir = np.sum(pf_filter,axis=1)

# #Plots
# plt.figure()
# plt.plot(simulation.time,pf_filter[:,5])

# plt.figure()
# plt.plot(simulation.time,ir)

# plt.figure()
# zeros = np.zeros(len(simulation.result[0]))
# Lp = 10*np.log10(ir**2/(2e-5)**2)
# Lp[Lp==-np.inf]=0
# plt.plot(simulation.time,Lp)
# plt.ylim(-60,10)

# #%%
# import soundfile as sf
# import sounddevice as sd

# #Load audio sample and convolve
# data, samplerate = sf.read('audio/mowa.wav')
# convolved = np.convolve(data, ir)

# #%%
# #Normalize audio and play
# convolved = 0.15*convolved/np.max(convolved)
# sd.play(convolved, fs)

# # %%
