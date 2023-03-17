# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:03:44 2023

@author: Owen
"""
#!/usr/bin/env python

# add framework path to Python's search path
import os, sys
hhgmax_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','..') # you need to adapt this!
sys.path.append(hhgmax_dir)

# import the pylewenstein module and numpy
import pylewenstein
import numpy as np
import pylab
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.interpolate import interp1d
from scipy import integrate
#%%
# set parameters (in SI units)

c = 299792458
eps0 = 8.8541878128*1e-12


wavelength_0 = 800*1e-9 # 800nm
fwhm_0 = 30*1e-15 # 30fs
ionization_potential = 21.5645*pylewenstein.e # 21.5645 eV (Ne)
E_0 = 2*1e-3 # 2mJ
spot_0 = 0.01*1e-2 # 0.01cm
peak_intensity_0 = 2*E_0/(fwhm_0*np.pi*spot_0**2)

wavelength_1 = 800*1e-19 
fwhm_1 = 30*1e-15
spot_1 = 0.01*1e-2


wavelength_2 = 400*1e-19
fwhm_2 = fwhm_0/np.sqrt(2)
spot_2 = 0.007*1e-2

# define time axis
T_1 = wavelength_0/c # one period of carrier
t = np.linspace(-20.*T_1,20.*T_1,2200*40+1)
T_2 = wavelength_1/c

def fitness_func(t,S,delay):
    E_1 = E_0*S 
    peak_intensity_1 = 2*E_1/(fwhm_1*np.pi*spot_1**2)
    E_2 = E_0*(1-S)
    E_2 = 0.2*(E_2/E_0)**2
    peak_intensity_2 = 2*E_2/(fwhm_2*np.pi*spot_2**2)

    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau_1 = fwhm_1/2./np.sqrt(np.log(np.sqrt(2)))
    Et_1 = np.exp(-(t/tau_1)**2) * np.cos(2*np.pi/T_1*t) * np.sqrt(2*peak_intensity_1/c/eps0)
    tau_2 = fwhm_2/2./np.sqrt(np.log(np.sqrt(2)))
    Et_2 = np.exp(-((t-delay)/tau_2)**2) * np.cos(2*np.pi/T_2*t) * np.sqrt(2*peak_intensity_2/c/eps0)
    Et = Et_1+Et_2
    
    # calculate central wavelength
    amp = np.abs(np.fft.fftshift(np.fft.fft(Et)))
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    freq_center = integrate.trapz(amp ** 2 * freqs, freqs) / integrate.trapz(amp ** 2, freqs)
    lambda_center = 2 * np.pi * c / freq_center
    
    # calculate dipole response and its spectrum using FFT
    d = pylewenstein.lewenstein(t, Et, ionization_potential, lambda_center)
    q = freqs[freqs >= 0] / (1. / T_1)
    y = abs(np.fft.fft(d)[freqs >= 0]) ** 2
    y = np.log10(y)
    
    
    #plt.plot(x,y)
    
    
    
    
    
#%%BO
#peak_I: peak itensity in W/cm^2
#FWHM: width in fs
#lambda_: wavelength in nm

pbounds = {'peak_I':(0,1e16),'FWHM':(0,200),'lambda_':(0,2000),'chirped_coeff':(0,2e28)}

optimizer = BayesianOptimization(
    f=fitness_func_chirped,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True)

optimizer.maximize(init_points=1,n_iter=100)

plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")

#%% plot the initil and best dipole responce result
#initial spectrum
params=optimizer.space.params[0]
params={'FWHM':params[0],
        'chirped_coeff':params[1],
        'lambda_':params[2],
        'peak_I':params[3]
        }
plot_spectrum_chirped(params)

#best
params=optimizer.max['params']
plot_spectrum_chirped(params,xmin=40,xmax=46)
#%% example
params={'FWHM':30,
        'lambda_':1000,
        'peak_I':1e14}
plot_spectrum_gaussian(params)