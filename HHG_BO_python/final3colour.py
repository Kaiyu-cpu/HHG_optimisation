
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
import pandas as pd

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
me = 9.109e-31

wavelength_1 = 1600*1e-9 # 1600nm
fwhm_1 = 16*1e-15 # 16fs
#peak_intensity_1 = 2.47e18 # 2.47e14 W/cm2 

wavelength_2 = 800*1e-9 
fwhm_2 = 16*1e-15 # 40fs
#peak_intensity_2 = 0.73e18 # 0.73e14 W/cm2

wavelength_3 = 533*1e-9
fwhm_3 = 16e-15 # 20fs
#peak_intensity_3 = 0.31e18 # 0.31 e14 W/cm2
T_3 = wavelength_3/c

# define time axis
T_1 = wavelength_1/c # one period of carrier
t = np.linspace(-40.*T_1,40.*T_1,2000*40+1)
 
omega_0 = 2*np.pi/T_1
Up=(2*pylewenstein.e**2*3e18)/(c*eps0*me*4*omega_0**2)

q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T_1)
x = q[q>=0]

#remove low energy
energy = x * h_bar * omega_0
indices = np.argwhere(energy > ionization_potential + Up)
low_boundary = indices[0][0]

indices_cutoff = np.argwhere(energy > ionization_potential + 3.17*Up)
cutoff = indices_cutoff[0][0]

def fitness_func(I1, I2, I3, delay_12,delay_13, t=t, x=x, plot=False,Print=False,title='',show_driving=False,save=False):
    
    print('para:', I1,I2,I3,delay_12,delay_13)
    
    if (I1+I2+I3>3.5):
        return 0
    else:
        #convert to SI unit
        delay_12 = delay_12*1e-15 
        delay_13 = delay_13*1e-15
        peak_intensity_1 = I1*1e18
        peak_intensity_2 = I2*1e18
        peak_intensity_3 = I3*1e18
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
        #lambda_center=1600e-9
        d = pylewenstein.lewenstein(t,Et,ionization_potential,lambda_center)
        
        #fft to get HHG spectrum
        y = abs(np.fft.fft(d)[q>=0])**2
        y[y==0]=10e-85
        y = np.log10(y)
        #plt.plot(x,y)
        #print(x.shape)
        x_copy = x
        y_copy = y
        
        x_temp = x
        y_temp = y
    
        max_values, max_indices = non_overlap_max_with_index(y_temp,window_size=160)
        
        y_smooth = max_values
        x_smooth = x_temp[max_indices]
        
        y_smooth = np.array(y_smooth)
        x_smooth = np.array(x_smooth)
        
        # remove low energies 
        x = x[low_boundary:]
        y = y[low_boundary:]
        
        y_new = y_smooth[x_smooth>=30]
        x_new = x_smooth[x_smooth>=30]
        #plt.scatter(x_new,y_new)
        # Three step fitting
        q1=320
        initial_guesses = [q1,q1+60,-0.025,-65,-85]
        fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
        #plateau_length=fit[0]
        
        # check the fitting
        q1,q2,m1,c1,c3 = fit
        m2 = (c3-m1*q1-c1)/(q2-q1)
        if abs(m1)<abs(m2) and m1<0 and m2<0 and q1>0 and q2<400:
            plateau_length = fit[0]
        else:
            search_space=[200,250,300,310,330,340,350,360]
            for i in search_space:
                initial_guesses = [i,i+60,-0.025,-65,-85]
                fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
                q1,q2,m1,c1,c3 = fit
                m2 = (c3-m1*q1-c1)/(q2-q1)
                if abs(m1)<abs(m2) and m1<0 and m2<0 and q1>0 and q2<500:
                    plateau_length = fit[0]
                    break
        
        #print('fit:',fit)        
        # Calculate the plateau area
        
        #zero_level=find_zero_level(x,y,fit[1])
        zero_level=-75
        y=y-zero_level
        y_copy=y_copy-zero_level
        
        #cutoff_yield=y_copy[cutoff]
        mask = (310 <= x_copy) & (x_copy <= 320)
        cutoff_yield=np.mean(y_copy[mask])
        #y=10**y
        #area = np.trapz(10**y[x <= plateau_length], x[x <= plateau_length])
        #area=area+10000
        
        if plot==True:   
            plt.plot(x[x<fit[1]],y[x<fit[1]],color='royalblue',lw=0.5,label='spectrum')
            plt.plot(x[x>=fit[1]],y[x>=fit[1]],linestyle='--',color='orange',lw=0.5,label='numerical noise')
            #plt.scatter(x_new,y_new)
            plt.plot(x[x<fit[1]],three_step(x[x<fit[1]],*fit)-zero_level,color='red',lw=3,alpha=0.6,label='fitted line')
            plt.plot(x_copy[:low_boundary],y_copy[:low_boundary],linestyle='--',color='green',lw=0.5,label='low energy region')
            #plt.xlim(0,100)
            plt.grid()
            plt.legend(loc='upper right')
            plt.xlabel('harmonic orders')
            plt.ylabel('harmonic yeild (a.u.)')
            plt.title(title)
        
        if Print==True:
            print('cutoff)yield ',cutoff_yield)
            print('paras :',I1,I2,I3,delay_12,delay_13)
        
        if show_driving == True:
            plt.plot(t,Et)
            plt.ylabel("E")
            plt.xlabel("t")
            plt.xlim(-5.33e-15/2,5.33e-15/2)
        
        if save == True:
            length=len(x_copy)
            fit = np.pad(fit, (0, length - len(fit)), mode='constant', constant_values=np.nan)
            low_boundary_1 = np.pad(low_boundary, (0, length-1), mode='constant', constant_values=np.nan)
            cutoff_yield = np.pad(cutoff_yield, (0, length-1), mode='constant', constant_values=np.nan)
            params=[I1,I2,I3,delay_12,delay_13]
            params = np.pad(params, (0, length - len(params)), mode='constant', constant_values=np.nan)
            a = {'x':x_copy,
                'y':y_copy,
                'fit':fit,
                'low_boundary':low_boundary_1,
                'cutoff_yield':cutoff_yield,
                'params':params}
            df = pd.DataFrame(a)
            #print(df)
            df.to_csv('best_harmonic_yeild_data.csv')
        
        return cutoff_yield

#%%BO
#peak_I: peak itensity in W/cm^2
#FWHM: width in fs
#lambda_: wavelength in nm

pbounds = {'I1':(2,2.9),'I2':(0.2,1),'I3':(0.2,1),'delay_12':(-6,6),'delay_13':(-6,6)} # delay in fs

optimizer = BayesianOptimization(
    f=fitness_func,
    pbounds=pbounds,
    random_state=0,
    allow_duplicate_points=True)

optimizer.maximize(init_points=5,n_iter=1000)


#%%
plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "o")
plt.xlabel('iteration')
plt.ylabel('plateau area (in linear space)')
plt.grid()
plt.title('Bayesian optimisation learning curve')
#%% drop zeros
Area = optimizer.space.target
Area = Area[Area>0]
plt.plot(range(len(Area)),Area,'o')

#%% plot the initil and best dipole responce result
#initial spectrum
params=optimizer.space.params[0]
fitness_func(params[0],params[1],params[2],params[3],params[4],plot=True,Print=True,title='initial')

#%%best
params=optimizer.max['params']
params=list(params.values())
fitness_func(params[0],params[1],params[2],params[3],params[4],plot=True,Print=True,title='best',save=True)
#plot_spectrum_chirped(params,xmin=40,xmax=46)

#%%
delay_12=[]
delay_13=[]
wavelength_2=[]
for i in range(305):
    delay_12.append(optimizer.space.params[i][0])
    delay_13.append(optimizer.space.params[i][1])
    wavelength_2.append(optimizer.space.params[i][2])
    
iteration=np.linspace(0,304,305)
#%%
plt.ylabel('delay_13')
plt.xlabel('iteration')
plt.scatter(iteration,delay_13)
plt.grid()

#%% example
params={'FWHM':30,
        'lambda_':1000,
        'peak_I':1e14}
plot_spectrum_gaussian(params)

