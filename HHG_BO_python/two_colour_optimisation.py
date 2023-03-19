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
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import curve_fit

def three_step(q,q1,q2,m1,c1,c3):
       m2 = (c3-m1*q1-c1)/(q2-q1)
       c2 = (m1-m2)*q1+c1
       l1=m1*q+c1
       l2=m2*q+c2
       l3=c3
       fitting = np.heaviside(-q+q1,0.5)*l1 + np.heaviside(q-q1,0.5)*np.heaviside(-q+q2,0.5)*l2 + np.heaviside(q-q2,0.5)*l3
       #print(m2)
       return fitting

def non_overlap_max_with_index(input_sequence, window_size=80):
    input_sequence=list(input_sequence)
    num_windows = len(input_sequence) // window_size
    max_values = []
    max_indices = []
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_values = input_sequence[start_idx:end_idx]
        max_value = max(window_values)
        max_index = start_idx + window_values.index(max_value)
        max_values.append(max_value)
        max_indices.append(max_index)
    return max_values, max_indices
#%%
# set parameters (in SI units)

c = pylewenstein.c
eps0 = pylewenstein.eps0
h_bar = 1.054571817e-34

wavelength_0 = 800*1e-9 # 800nm
fwhm_0 = 30*1e-15 # 30fs
ionization_potential = 21.5645*pylewenstein.e # 21.5645 eV (Ne)
E_0 = 2*1e-3 # 2mJ
spot_0 = 0.01*1e-2 # 0.01cm
peak_intensity_0 = 2*E_0/(fwhm_0*np.pi*spot_0**2)

wavelength_1 = 800*1e-9 
fwhm_1 = 30*1e-15
spot_1 = 0.01*1e-2


wavelength_2 = 400*1e-9
fwhm_2 = fwhm_0/np.sqrt(2)
spot_2 = 0.007*1e-2

# define time axis
T_1 = wavelength_0/c # one period of carrier
t = np.linspace(-40.*T_1,40.*T_1,1100*40+1)
T_2 = wavelength_2/c

def fitness_func(delay,t=t):
    S=0.5
    delay = delay*1e-15 # convert unit to fs
    # claculate the peak intensity of two waves
    E_1 = E_0*S 
    peak_intensity_1 = 2*E_1/(fwhm_1*np.pi*spot_1**2)
    E_2 = E_0*(1-S)
    E_2 = (0.2*(E_2/E_0)**2)*1e-3
    peak_intensity_2 = 2*E_2/(fwhm_2*np.pi*spot_2**2)
    #print(E_1,E_2)
    #print(peak_intensity_1,peak_intensity_2)

    # define net drving electric field
    tau_1 = fwhm_1/2./np.sqrt(np.log(np.sqrt(2)))
    Et_1 = np.exp(-(t/tau_1)**2) * np.cos(2*np.pi/T_1*t) * np.sqrt(2*peak_intensity_1/c/eps0)
    tau_2 = fwhm_2/2./np.sqrt(np.log(np.sqrt(2)))
    Et_2 = np.exp(-((t-delay)/tau_2)**2) * np.cos(2*np.pi/T_2*t) * np.sqrt(2*peak_intensity_2/c/eps0)
    Et = Et_1+Et_2
    #plt.plot(t,Et)
    #plt.show()
    
    # calculate central wavelength of drving field
    power_spectrum = np.abs(np.fft.fft(Et))**2
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    power_spectrum = power_spectrum[freqs>=0]
    freqs = freqs[freqs>=0]
    freq_center = np.sum(power_spectrum * freqs) / np.sum(power_spectrum)
    lambda_center = c / freq_center
    #print(lambda_center)
    #plt.plot(freqs,power_spectrum)
    
    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,lambda_center)
    
    #fft to get HHG spectrum
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T_1)
    x = q[q>=0]
    y = abs(np.fft.fft(d)[q>=0])**2
    y = np.log10(y)
    
    x_copy = x
    y_copy = y
    
    x_temp = x[20:]
    y_temp = y[20:]
    
    max_values, max_indices = non_overlap_max_with_index(y_temp,window_size=80)
    
    y_new = max_values
    x_new = x_temp[max_indices]
    
    # remove low energies 
    omega_0 = 2*np.pi/T_1
    energy = x * h_bar * omega_0
    indices = np.argwhere(energy > ionization_potential)
    low_boundary = indices[0][0]
    x = x[low_boundary:]
    y = y[low_boundary:]
    
    energy_new = x_new * h_bar * omega_0
    indices_new = np.argwhere(energy_new > ionization_potential)
    low_boundary_new = indices_new[0][0]
    x_new = x_new[low_boundary_new:140]
    y_new = y_new[low_boundary_new:140]
    
    # Three step fitting
    q1=0.0002856*lambda_center**2-0.1715*lambda_center+38.7331
    initial_guesses = [q1,q1+10,-0.1,-60,-80]
    fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
    plateau_length = fit[0]
    
    plt.plot(x,y,color='royalblue',lw=0.5)
    #plt.scatter(x_new,y_new)
    plt.plot(x,three_step(x,*fit),color='red',lw=3,alpha=0.6)
    plt.plot(x_copy[:low_boundary],y_copy[:low_boundary],linestyle='--',color='royalblue',lw=0.5)
    plt.xlim(0,100)
    plt.grid()
    plt.xlabel('harmonic orders')
    plt.ylabel('harmonic yeild (a.u.)')
    plt.title('Ne single atom dipole response')
    # Calculate the plateau area
    #y=10**y
    
    #area = np.trapz(y[x <= plateau_length], x[x <= plateau_length])
    
    #print(area)
    return plateau_length
#%% grid search
area=[]
for i in range (-50,50):
    area.append(fitness_func(i))
#%%
S=np.linspace(-50,50,100)
plt.plot(S,area)
plt.xlabel('delay time (fs)')
plt.ylabel('plateau length (harmonic orders)')
plt.grid()
plt.title('plateau length vs delay time')
    
#%%BO
#peak_I: peak itensity in W/cm^2
#FWHM: width in fs
#lambda_: wavelength in nm

pbounds = {'delay':(-50,50)} # delay in fs

optimizer = BayesianOptimization(
    f=fitness_func,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True)

optimizer.maximize(init_points=1,n_iter=300)

#%%
delays=[]
for i in range(301):
    delays.append(optimizer.space.params[i])
delays=np.array(delays)
iteration=np.linspace(0,300,301)
plt.ylabel('delay time (fs)')
plt.xlabel('iteration')
# set the thresholds and colors for coloring the bars
low_threshold = -10
high_threshold = 30
low_color = 'b'
high_color = 'r'
# create a list of colors for the bars based on their values
colors = [low_color if y_vals <= low_threshold else high_color if y_vals >= high_threshold else 'g' for y_vals in delays]
plt.scatter(iteration,delays,color=colors,alpha=0.5)
plt.grid()

#%%
plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
plt.xlabel('iteration')
plt.ylabel('plateau length (harmonic orders)')
plt.grid()
plt.title('Bayesion optimisation learning curve')
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