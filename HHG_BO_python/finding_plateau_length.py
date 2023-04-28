# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:43:05 2023

@author: Owen
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:16:23 2023

@author: Owen
"""
# add framework path to Python's search path
import os, sys
hhgmax_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','..') # you need to adapt this!
sys.path.append(hhgmax_dir)

# import the pylewenstein module and numpy
import pylewenstein
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelextrema

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
# set parameters (everything is SI units)
#wavelength = 1000e-9 # 1000 nm
fwhm = 30e-15 # 30 fs
#ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
#peakintensity = 1e14 * 1e4 # 1e14 W/cm^2
ionization_potential = 21.5645*pylewenstein.e # 21.5645 eV (Ne)
peakintensity = 5e14 * 1e4 # 1e14 W/cm^2
c = 299792458
h_bar = 1.054571817e-34
eps0 = 8.8541878128e-12
me = 9.1093837e-31
#%%
x_ensemble = np.load(r"C:\Users\chaof\Downloads\HHG_BO_python_spectrum_data\x_ensemble_Ne.csv")
y_ensemble = np.load(r"C:\Users\chaof\Downloads\HHG_BO_python_spectrum_data\y_ensemble_Ne.csv")
#%%
temp=-1
for wavelength in range (400,801):
    temp+=1
    
    if temp%100==0:
        plt.plot(x_ensemble[wavelength-200],y_ensemble[wavelength-200],label='wavelength={}nm'.format(wavelength))
        plt.legend()
        plt.title('Ne,width=30fs,Ipeak=5*10^14 W/cm^2')
        plt.grid() 
        #plt.show()
        
plt.show()

#%%
plateau_length=[]
for wavelength in range (400,2001):
    x = x_ensemble[wavelength-200]
    y = y_ensemble[wavelength-200]
    
    x_temp = x[20:]
    y_temp = y[20:]
    
    max_values, max_indices = non_overlap_max_with_index(y_temp,window_size=80)
    
    y_new = max_values
    x_new = x_temp[max_indices]
    
    # remove low energies 
    T = wavelength*1e-9/c
    omega_0 = 2*np.pi/T
    energy = x * h_bar * omega_0
    indices = np.argwhere(energy > ionization_potential)
    low_boundary = indices[0][0]
    x = x[low_boundary:]
    y = y[low_boundary:]
    
    energy_new = x_new * h_bar * omega_0
    indices_new = np.argwhere(energy_new > ionization_potential)
    low_boundary_new = indices_new[0][0]
    x_new = x_new[low_boundary_new:60]
    y_new = y_new[low_boundary_new:60]
    '''
    y_smooth = savgol_filter(y,51,3)
    max_indices=list(argrelextrema(y_smooth, np.greater)[0])
    #temp=y[max_indices]
    #max_indices_2=list(argrelextrema(temp, np.greater)[0])
    x_temp = x[max_indices]
    y_temp = y[max_indices]
    mean=np.mean(y_temp)
    x_new = x_temp[y_temp>=mean]
    y_new = y_temp[y_temp>=mean]
    '''
    
    #threshold = 0.5
    #end_index = np.where(y < threshold*x[low_boundary])[0][0] + low_boundary
    #plateau_length.append(end_index)
    if wavelength < 1500:
        q1 = 0.0002856*wavelength**2-0.1715*wavelength+38.7331
    else:
        q1=ionization_potential + 3.17 * 2*(pylewenstein.e)**2*peakintensity*(wavelength*1e-9)**2/(16*c**3*eps0*me*np.pi**2)/(2*np.pi*h_bar*c/(wavelength*1e-9))
    #print(q1)
    initial_guesses = [q1,q1+20,-0.1,-60,-80]
    #y=np.nan_to_num(y)
    fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
    #print(fit)
    plateau_length.append(fit[0])
    print("{}".format(wavelength))
    #plt.scatter(x_new,y_new)
    #plt.plot(x,y)
    #plt.plot(x_new,three_step(x_new,*fit))
    #plt.show()

#%%
wavelength=np.linspace(400,2001,1601)
plateau_length = np.array([plateau_length]).reshape(1601)
mask =  plateau_length >= 0
wavelength = wavelength[mask]
plateau_length = plateau_length[mask]
plateau_length = np.delete(plateau_length,508)
for i in range(len(plateau_length)):
    if wavelength[i]>1500:
        plateau_length[i] += 30
wavelength = np.delete(wavelength,508)
coeffs = np.polyfit(wavelength, plateau_length, 2)

#%%
# cut-off law

f_pred = ionization_potential + 3.17 * 2*(pylewenstein.e)**2*peakintensity*(wavelength*1e-9)**2/(16*c**3*eps0*me*np.pi**2)
f_pred = f_pred/(2*np.pi*h_bar*c/(wavelength*1e-9))
# plot the data and the fitted quadratic
#plt.figure(figsize=(16, 10))


short_mask = np.ones(len(f_pred),dtype=bool)
for i in range(len(short_mask)):
    if i%15 != 0:
        short_mask[i] = False
plt.figure(figsize=(10,6))
wavelen_short = wavelength[short_mask]
plateau_length_short = plateau_length[short_mask]
plt.plot(wavelength, f_pred,color='r', label="Cut-off Law",lw=3,alpha=0.7)
plt.scatter(wavelen_short, plateau_length_short,color='royalblue',label="Lewenstein Model Prediction",s=80,marker='+')
plt.legend(fontsize=15)
plt.grid()
plt.xlabel('wavelength (nm)',fontsize=15)
plt.ylabel("plateau length (harmonic orders)",fontsize=15)
plt.show()

#%% check wrong points
residuals = f_pred-plateau_length
outlier_indices = np.where(np.abs(residuals) > 100)
outlier_waveln = wavelength[outlier_indices]
outlier_plateau = plateau_length[outlier_indices]
plt.scatter(wavelength, plateau_length, s=2,label="Data")
plt.scatter(outlier_waveln,outlier_plateau, s=2,color='purple',label="Outlier")
plt.plot(wavelength, f_pred,color='red', label="Fitted Quadratic")
plt.legend()
plt.show()
#%%
mask = np.ones(wavelength.shape, dtype=bool)
mask[[569,592]] = False
#plt.scatter(wavelength, plateau_length, s=2,label="Data")

plt.plot(wavelength, f_pred,color='red', label="Cut-off Law")
plt.scatter(wavelength[mask],plateau_length[mask], s=20,color='blue',label="Lewenstein Model Prediction",marker='+')
plt.legend()
plt.show()
#%%
for wavelength in range (498,499):
    x = x_ensemble[wavelength-200]
    y = y_ensemble[wavelength-200]
    
    x = x[20:]
    y = y[20:]
    
    
    # remove low energies 
    T = wavelength*1e-9/c
    omega_0 = 2*np.pi/T
    energy = x * h_bar * omega_0
    indices = np.argwhere(energy > ionization_potential)
    low_boundary = indices[0][0]
    x = x[low_boundary:]
    y = y[low_boundary:]
    
    y_smooth = savgol_filter(y,51,3)
    max_indices=list(argrelextrema(y_smooth, np.greater)[0])
    #temp=y[max_indices]
    #max_indices_2=list(argrelextrema(temp, np.greater)[0])
    x_temp = x[max_indices]
    y_temp = y[max_indices]
    mean=np.mean(y_temp)
    x_new = x_temp[y_temp>=mean]
    y_new = y_temp[y_temp>=mean]
    
    
    #threshold = 0.5
    #end_index = np.where(y < threshold*x[low_boundary])[0][0] + low_boundary
    #plateau_length.append(end_index)
    
    q1=0.0002856*wavelength**2-0.1715*wavelength+38.7331
    print(q1)
    initial_guesses = [q1,q1+10,-0.1,-60,-80]
    #y=np.nan_to_num(y)
    fit, fit_cov = curve_fit(three_step,x_new,y_new,p0=initial_guesses)
    #print(fit)
    #print("{}".format(waveln))
    plt.plot(x_new,y_new)
    plt.plot(x,y)
    #plt.plot(x_new,three_step(x_new,*fit))
    plt.show()






