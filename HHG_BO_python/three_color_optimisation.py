# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:05:57 2023

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

def find_zero_level(x,y,q2):
    zero_level = np.mean(y[x>q2])
    return zero_level
    
#%%
# set parameters (in SI units)

c = pylewenstein.c
eps0 = pylewenstein.eps0
h_bar = 1.054571817e-34
ionization_potential = 21.5645*pylewenstein.e # 21.5645 eV (Ne)

wavelength_1 = 770*1e-9 # 770nm
#fwhm_1 = 30*1e-15 # 30fs

#wavelength_2 = 800*1e-9 
fwhm_2 = 40*1e-15 # 40fs
peak_intensity_2 = 1e18 # 1e14 W/cm2

wavelength_3 = 400*1e-9
fwhm_3 = 20e-15 # 20fs
peak_intensity_3 = 1e18 # 1e14 W/cm2
T_3 = wavelength_3/c

# define time axis
T_1 = wavelength_1/c # one period of carrier
t = np.linspace(-60.*T_1,60.*T_1,2600*60+1)

def fitness_func(delay_12,delay_13,fwhm_1,wavelength_2,t=t,plot=False,Print=False,title='',show_driving=False):
    print('para:', delay_12,delay_13,fwhm_1,wavelength_2)
    peak_intensity_1 = 4e19/fwhm_1 # 4e15 W/cm2 / fwhm
    
    #convert to SI unit
    fwhm_1 = fwhm_1*1e-15
    wavelength_2 = wavelength_2*1e-9
    delay_12 = delay_12*1e-15 
    delay_13 = delay_13*1e-15
    
    # define net drving electric field
    tau_1 = fwhm_1/2./np.sqrt(np.log(np.sqrt(2)))
    Et_1 = np.exp(-(t/tau_1)**2) * np.cos(2*np.pi/T_1*t) * np.sqrt(2*peak_intensity_1/c/eps0)
    
    T_2 = wavelength_2/c
    tau_2 = fwhm_2/2./np.sqrt(np.log(np.sqrt(2)))
    Et_2 = np.exp(-((t-delay_12)/tau_2)**2) * np.cos(2*np.pi/T_2*t) * np.sqrt(2*peak_intensity_2/c/eps0)
    
    tau_3 = fwhm_3/2./np.sqrt(np.log(np.sqrt(2)))
    Et_3 = np.exp(-((t-delay_13)/tau_3)**2) * np.cos(2*np.pi/T_3*t) * np.sqrt(2*peak_intensity_3/c/eps0)
    
    #Et=Et_1
    Et = Et_1+Et_2+Et_3
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
    y[y==0]=10e-85
    y = np.log10(y)
    #print(x.shape)
    x_copy = x
    y_copy = y
    
    x_temp = x
    y_temp = y

    max_values, max_indices = non_overlap_max_with_index(y_temp,window_size=240)
    
    y_new = max_values
    x_new = x_temp[max_indices]
    
    # remove low energies 
    omega_0 = 2*np.pi/T_1
    energy = x * h_bar * omega_0
    indices = np.argwhere(energy > ionization_potential)
    low_boundary = indices[0][0]
    x = x[low_boundary:]
    y = y[low_boundary:]
    '''
    energy_new = x_new * h_bar * omega_0
    indices_new = np.argwhere(energy_new > ionization_potential)
    low_boundary_new = indices_new[0][0]
    x_new = x_new[low_boundary_new:140]
    y_new = y_new[low_boundary_new:140]
    '''
    # Three step fitting
    q1=800/(fwhm_1*1e15)
    initial_guesses = [q1,q1+20,-0.1,-65,-85]
    fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
    #plateau_length=fit[0]
    
    # check the fitting
    q1,q2,m1,c1,c3 = fit
    m2 = (c3-m1*q1-c1)/(q2-q1)
    if abs(m1)<abs(m2) and m1<0 and m2<0 and q1>0 and q2<400:
        plateau_length = fit[0]
    else:
        search_space=[50,100,150,200,250,300,350]
        for i in search_space:
            initial_guesses = [i,i+20,-0.1,-65,-85]
            fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
            q1,q2,m1,c1,c3 = fit
            m2 = (c3-m1*q1-c1)/(q2-q1)
            if abs(m1)<abs(m2) and m1<0 and m2<0 and q1>0 and q2<400:
                plateau_length = fit[0]
                break
    
    print('fit:',fit)        
    # Calculate the plateau area
    zero_level=find_zero_level(x,y,fit[1])
    y=y-zero_level
    y_copy=y_copy-zero_level
    #y=10**y
    area = np.trapz(y[x <= plateau_length], x[x <= plateau_length])
    #area=area+10000
    
    if plot==True:   
        plt.plot(x[x<fit[1]],y[x<fit[1]],color='royalblue',lw=0.5,label='spectrum')
        plt.plot(x[x>=fit[1]],y[x>=fit[1]],linestyle='--',color='orange',lw=0.5,label='numerical noise')
        #plt.scatter(x_new,y_new)
        plt.plot(x[x<fit[1]],three_step(x[x<fit[1]],*fit)-zero_level,color='red',lw=3,alpha=0.6,label='fitted line')
        plt.plot(x_copy[:low_boundary],y_copy[:low_boundary],linestyle='--',color='orange',lw=0.5)
        #plt.xlim(0,100)
        plt.grid()
        plt.legend(loc='upper right')
        plt.xlabel('harmonic orders')
        plt.ylabel('harmonic yeild (a.u.)')
        plt.title(title)
    
    if Print==True:
        print('area = ',area)
        print('paras :',delay_12,delay_13,fwhm_1,wavelength_2)
    
    if show_driving == True:
        plt.plot(t,Et)
        plt.ylabel("E")
        plt.xlabel("t")
    
    return area

#%%BO
#peak_I: peak itensity in W/cm^2
#FWHM: width in fs
#lambda_: wavelength in nm

pbounds = {'fwhm_1':(4,30),'wavelength_2':(1200,2000),'delay_12':(-70,70),'delay_13':(-50,50)} # delay in fs

optimizer = BayesianOptimization(
    f=fitness_func,
    pbounds=pbounds,
    random_state=3,
    allow_duplicate_points=True)

optimizer.maximize(init_points=5,n_iter=300)


#%%
plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
plt.xlabel('iteration')
plt.ylabel('plateau area (in log space)')
plt.grid()
plt.title('Bayesian optimisation learning curve')
#%% plot the initil and best dipole responce result
#initial spectrum
params=optimizer.space.params[0]
fitness_func(params[0],params[1],params[2],params[3],plot=True,Print=True,title='initial')

#%%best
params=optimizer.max['params']
params=list(params.values())
fitness_func(params[0],params[1],params[2],params[3],plot=True,Print=True,title='best')
#plot_spectrum_chirped(params,xmin=40,xmax=46)

#%%
delay_12=[]
delay_13=[]
fwhm_1=[]
wavelength_2=[]
for i in range(305):
    delay_12.append(optimizer.space.params[i][0])
    delay_13.append(optimizer.space.params[i][1])
    fwhm_1.append(optimizer.space.params[i][2])
    wavelength_2.append(optimizer.space.params[i][3])
    
iteration=np.linspace(0,304,305)
#%%
plt.ylabel('wavelength_2')
plt.xlabel('iteration')
plt.scatter(iteration,wavelength_2)
plt.grid()

#%% example
params={'FWHM':30,
        'lambda_':1000,
        'peak_I':1e14}
plot_spectrum_gaussian(params)