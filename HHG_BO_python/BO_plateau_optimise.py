# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:20:53 2023

@author: Owen
"""
# -*- coding: utf-8 -*-

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

def three_step(q,c1,m2,c2,c3,q1,q2):
       m2 = (c3-c1)/(q2-q1)
       c2 = c3-m2*q2
       fitting = c1*np.heaviside(-q+q1,0.5)+(m2*q+c2)*np.heaviside(q-q1,0.5)*np.heaviside(-q+q2,0.5)+c3*np.heaviside(q-q2,0.5)
       return fitting
   
def fitness_func_gaussian(peak_I,FWHM,lambda_):
    
    # set parameters (in SI units)
    wavelength = lambda_*1e-9
    fwhm = FWHM*1e-15
    ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
    peakintensity = peak_I * 1e4 # W/m^2
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-40.*T,40.*T,400*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)
    
    #fft to get spectrum
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    x = q[q>=0]
    y = abs(np.fft.fft(d)[q>=0])**2
    y = np.log10(y)
    
    omega_0 = 2*np.pi/T
    h_bar = 1.054571817e-34

    # remove low energies 
    temp = []
    for i in range(len(x)):
        if x[i]*h_bar*omega_0 <= ionization_potential:
            temp.append(i)
    
    x = np.delete(x,temp)
    y = np.delete(y,temp) 
    
    # Three step fitting
    from scipy.optimize import curve_fit
    
    initial_guesses = [-60,-1,-40,-80,30,60]
    fit, fit_cov = curve_fit(three_step,x,y,p0=initial_guesses)
    
    return (100+fit[0])*fit[4]

def plot_spectrum_gaussian(params,xmin=0,xmax=100):
    # set parameters (in SI units)
    wavelength = params['lambda_']*1e-9
    fwhm = params['FWHM']*1e-15
    ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
    peakintensity = params['peak_I'] * 1e4 # W/m^2
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-40.*T,40.*T,400*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)
    
    #fft to get spectrum and plot  
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    I = abs(np.fft.fft(d))**2                   
    
    pylab.figure(figsize=(5,4))
    pylab.semilogy(q[q>=0],I[q>=0])
    pylab.xlim((xmin,xmax))
    pylab.grid()
    pylab.show()

def fitness_func_chirped(peak_I, FWHM, lambda_, chirped_coeff):
    
    # set parameters (in SI units)
    wavelength = lambda_*1e-9
    fwhm = FWHM*1e-15
    ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
    peakintensity = peak_I * 1e4 # W/m^2
    b = chirped_coeff
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-40.*T,40.*T,400*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t+b*t**2) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)
    
    #fft to get spectrum
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    x = q[q>=0]
    y = abs(np.fft.fft(d)[q>=0])**2
    y = np.log10(y)
    
    omega_0 = 2*np.pi/T
    h_bar = 1.054571817e-34

    # remove low energies 
    temp = []
    for i in range(len(x)):
        if x[i]*h_bar*omega_0 <= ionization_potential:
            temp.append(i)
    
    x = np.delete(x,temp)
    y = np.delete(y,temp) 
    
    # Three step fitting
    from scipy.optimize import curve_fit
    
    initial_guesses = [-60,-1,-40,-80,30,60]
    fit, fit_cov = curve_fit(three_step,x,y,p0=initial_guesses)
    
    if fit[4]<=0:
        return 0.01
    else:
        return (100+fit[0])*fit[4]

def plot_spectrum_chirped(params,xmin=0,xmax=100):
    # set parameters (in SI units)
    wavelength = params['lambda_']*1e-9
    fwhm = params['FWHM']*1e-15
    ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
    peakintensity = params['peak_I'] * 1e4 # W/m^2
    b=params['chirped_coeff']
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-40.*T,40.*T,400*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t+b*t**2) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)
    
    #fft to get spectrum
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    x = q[q>=0]
    y = abs(np.fft.fft(d)[q>=0])**2
    y = np.log10(y)
    y_uncorrect=y
    
    omega_0 = 2*np.pi/T
    h_bar = 1.054571817e-34

    # remove low energies 
    temp = []
    for i in range(len(x)):
        if x[i]*h_bar*omega_0 <= ionization_potential:
            temp.append(i)
    
    x = np.delete(x,temp)
    y = np.delete(y,temp) 
    
    # Three step fitting
    from scipy.optimize import curve_fit
    
    initial_guesses = [-60,-1,-40,-80,30,60]
    fit, fit_cov = curve_fit(three_step,x,y,p0=initial_guesses)
    
    print(fit)

    plt.plot(q[q>=0],y_uncorrect)
    plt.plot(x,three_step(x,*fit))                
    pylab.xlim((xmin,xmax))
    pylab.grid()
    pylab.show()
    
def plot_driving_field_chirped(params,xmin=-5,xmax=5):
    # set parameters (in SI units)
    wavelength = params['lambda_']*1e-9
    fwhm = params['FWHM']*1e-15
    peakintensity = params['peak_I'] * 1e4 # W/m^2
    b=params['chirped_coeff']
    
    # define time axis
    T = wavelength/pylewenstein.c # one period of carrier
    t = np.linspace(-40.*T,40.*T,400*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t + b*t**2) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)
    
    pylab.plot(t,Et)
    pylab.xlim(xmin*1e-14,xmax*1e-14)
#%%BO


#peak_I: peak itensity in W/cm^2
#FWHM: width in fs
#lambda_: wavelength in nm
pbounds = {'peak_I':(1e13,1e15),'FWHM':(1,200),'lambda_':(200,2000),'chirped_coeff':(0,2e28)}

optimizer = BayesianOptimization(
    f=fitness_func_chirped,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True)

optimizer.maximize(init_points=1,n_iter=200)

plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
plt.show()

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
plot_spectrum_chirped(params,xmin=0,xmax=120)
plt.show()
#%% example
params={'FWHM':30,
        'lambda_':1000,
        'peak_I':1e14}
plot_spectrum_gaussian(params)