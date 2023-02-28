# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:28:04 2023

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

def three_step(q,c1,m2,c2,c3,q1,q2):
       m2 = (c3-c1)/(q2-q1)
       c2 = c3-m2*q2
       fitting = c1*np.heaviside(-q+q1,0.5)+(m2*q+c2)*np.heaviside(q-q1,0.5)*np.heaviside(-q+q2,0.5)+c3*np.heaviside(q-q2,0.5)
       return fitting
   
# set parameters (everything is SI units)
wavelength = 1000e-9 # 1000 nm
fwhm = 30e-15 # 30 fs
ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
peakintensity = 1e14 * 1e4 # 1e14 W/cm^2

# define time axis
T = wavelength/pylewenstein.c # one period of carrier
t = np.linspace(-20.*T,20.*T,200*40+1)

# define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)

# use module to calculate dipole response (returns dipole moment in SI units)
d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength)

# plot result
import pylab

'''
pylab.figure(figsize=(5,4))
q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
pylab.semilogy(q[q>=0], abs(np.fft.fft(d)[q>=0])**2)
pylab.xlim((0,100))
pylab.show()
'''

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
#pylab.xlim((xmin,xmax))
pylab.grid()
pylab.show()
    




