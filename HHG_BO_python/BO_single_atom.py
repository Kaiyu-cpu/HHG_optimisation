# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:38:49 2023

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

n = 43 # n is number of harmonic we want to optimise

def compare_I(n,q,I): # n is number of harmonic we want to optimise
    I = interp1d(q,I)
    #intensity of harmonic  - nearest neighbours'
    I_residual = I(n) - I(n-2) - I(n+2)
    return I_residual

def fitness_func(peak_I,FWHM,lambda_,n=n):
    
    # set parameters (in SI units)
    wavelength = lambda_*1e-9
    fwhm = FWHM*1e-15
    ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
    peakintensity = peak_I * 1e4 # W/m^2
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-20.*T,20.*T,200*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)
    
    #fft to get spectrum
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    I = abs(np.fft.fft(d))**2
    
    return compare_I(n,q,I)

def plot_spectrum(params):
    # set parameters (in SI units)
    wavelength = params['lambda_']*1e-9
    fwhm = params['FWHM']*1e-15
    ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
    peakintensity = params['peak_I'] * 1e4 # W/m^2
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-20.*T,20.*T,200*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)
                                
    pylab.figure(figsize=(5,4))
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    I = abs(np.fft.fft(d))**2
    pylab.semilogy(q[q>=0],I[q>=0])
    pylab.xlim((40,46))
    pylab.grid()
    pylab.show()

#%%BO


#peak_I: peak itensity in W/cm^2
#FWHM: width in fs
#lambda_: wavelength in nm
pbounds = {'peak_I':(0,1e23),'FWHM':(1,200),'lambda_':(700,1200)}

optimizer = BayesianOptimization(
    f=fitness_func,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True)

optimizer.maximize(init_points=1,n_iter=100)

plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")

#%% plot the initil and best dipole responce result
#initial spectrum
params=optimizer.space.params[0]
params={'FWHM':params[0],
        'lambda_':params[1],
        'peak_I':params[2]}
plot_spectrum(params)

#best
params=optimizer.max['params']
plot_spectrum(params)
#%%
params={'FWHM':30,
        'lambda_':1000,
        'peak_I':1e14}
plot_spectrum(params)