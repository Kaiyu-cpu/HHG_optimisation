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

def three_step(q,q1,q2,m1,c1,c3):
       m2 = (c3-m1*q1-c1)/(q2-q1)
       c2 = (m1-m2)*q1+c1
       l1=m1*q+c1
       l2=m2*q+c2
       l3=c3
       fitting = np.heaviside(-q+q1,0.5)*l1 + np.heaviside(q-q1,0.5)*np.heaviside(-q+q2,0.5)*l2 + np.heaviside(q-q2,0.5)*l3
       #print(m2)
       return fitting

#%%
# set parameters (everything is SI units)
#wavelength = 1000e-9 # 1000 nm
fwhm = 30e-15 # 30 fs
#ionization_potential = 12.13*pylewenstein.e # 12.13 eV (Xe)
#peakintensity = 1e14 * 1e4 # 1e14 W/cm^2
ionization_potential = 21.5645*pylewenstein.e # 21.5645 eV (Ne)
peakintensity = 5e14 * 1e4 # 1e14 W/cm^2
#%%
# plot result

'''
pylab.figure(figsize=(5,4))
q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
pylab.semilogy(q[q>=0], abs(np.fft.fft(d)[q>=0])**2)
pylab.xlim((0,100))
pylab.show()
'''

'''
omega_0 = 2*np.pi/T
h_bar = 1.054571817e-34

# remove low energies 
temp = []
for i in range(len(x)):
    if x[i]*h_bar*omega_0 <= ionization_potential:
        temp.append(i)

x = np.delete(x,temp)
y = np.delete(y,temp) 
'''
x_ensemble=[]
y_ensemble=[]

for wavelength in range (200,2001):

    # define time axis
    T = wavelength*1e-9/pylewenstein.c # one period of carrier
    t = np.linspace(-20.*T,20.*T,2200*40+1)
    
    # define electric field: Gaussian with given peak intensity, FWHM and a cos carrier
    tau = fwhm/2./np.sqrt(np.log(np.sqrt(2)))
    Et = np.exp(-(t/tau)**2) * np.cos(2*np.pi/T*t) * np.sqrt(2*peakintensity/pylewenstein.c/pylewenstein.eps0)

    # use module to calculate dipole response (returns dipole moment in SI units)
    d = pylewenstein.lewenstein(t,Et,ionization_potential,wavelength*1e-9)
    
    #fft to get spectrum
    q = np.fft.fftfreq(len(t), t[1]-t[0])/(1./T)
    x = q[q>=0]
    y = abs(np.fft.fft(d)[q>=0])**2
    y = np.log10(y)
    
    #plt.plot(x,y)
    
    x_ensemble.append(x)
    y_ensemble.append(y)
    
    print('{}nm is done!'.format(wavelength))
#%%
np.save(r"C:\Users\chaof\Downloads\HHG_BO_python_spectrum_data\x_ensemble_Ne.csv",x_ensemble)
np.save(r"C:\Users\chaof\Downloads\HHG_BO_python_spectrum_data\y_ensemble_Ne.csv",y_ensemble)
#%%
x_ensemble = np.load(r"C:\Users\chaof\Downloads\HHG_BO_python_spectrum_data\x_ensemble_Ne.csv")
y_ensemble = np.load(r"C:\Users\chaof\Downloads\HHG_BO_python_spectrum_data\y_ensemble_Ne.csv")

#%%
temp=-1
for wavelength in range (800,2001):
    temp+=1
    
    if temp%200==0:
        plt.plot(x_ensemble[wavelength-200],y_ensemble[wavelength-200],label='wavelength={}nm'.format(wavelength))
        plt.legend()
        plt.title('Ne,width=30fs,Ipeak=5*10^14 W/cm^2')
        plt.grid() 
        #plt.show()
        
plt.show()


#%% Fitting to find the plateau extentsion 
c = 299792458
h_bar = 1.054571817e-34
plateau_length=[]
for wavelength in range (800,2001):
    x = x_ensemble[wavelength-200]
    y = y_ensemble[wavelength-200]
    
    T = wavelength*1e-9/c
    omega_0 = 2*np.pi/T
    
    # remove low energies 
    energy = x * h_bar * omega_0
    indices = np.argwhere(energy > ionization_potential)
    low_boundary = indices[0][0]
    x = x[low_boundary:]
    y = y[low_boundary:]
    
    # Three step fitting
    #q1=0.00015*wavelength**2
    q1 = 5.14e-4*wavelength**2-6.95e-1*wavelength+323
    initial_guesses = [q1,q1+50,-0.1,-60,-80]
    y=np.nan_to_num(y)
    fit, fit_cov = curve_fit(three_step,x,y,p0=initial_guesses)
    plateau_length.append(fit[0])
<<<<<<< Updated upstream
    #print("{}".format(wavelength))
    #plt.plot(x,y)
    #plt.plot(x,three_step(x,*fit))
=======
    print("{}".format(wavelength))
    plt.plot(x,y)
    plt.plot(x,three_step(x,*fit))
    
>>>>>>> Stashed changes

#%%
wavelength=np.linspace(800,2000,1201)
plt.plot(wavelength,plateau_length)
plt.xlabel("wavelength(nm)")
plt.ylabel("plateau length (au)")

p = np.polyfit(wavelength,plateau_length,2)



#%% fit a quadratic to the curve, get the right shape
# perform quadratic fit
plateau_length = np.array([plateau_length]).reshape(1201)
mask = plateau_length >= 0
wavelength = wavelength[mask]
plateau_length = plateau_length[mask]

coeffs = np.polyfit(wavelength, plateau_length, 2)

# generate a function from the fitted coefficients
f = np.poly1d(coeffs)

# plot the data and the fitted quadratic
plt.scatter(wavelength, plateau_length, s=2,label="Data")
plt.plot(wavelength, f(wavelength),color='red', label="Fitted Quadratic")
plt.legend()
plt.show()

#%% check wrong points
residuals = f(wavelength)-plateau_length
outlier_indices = np.where(np.abs(residuals) > 20)
outlier_waveln = wavelength[outlier_indices]
outlier_plateau = plateau_length[outlier_indices]
plt.scatter(wavelength, plateau_length, s=2,label="Data")
plt.scatter(outlier_waveln,outlier_plateau, s=2,color='purple',label="Outlier")
plt.plot(wavelength, f(wavelength),color='red', label="Fitted Quadratic")
plt.legend()
plt.show()

for waveln in [1500]:
    x = x_ensemble[int(waveln-200)]
    y = y_ensemble[int(waveln-200)]
    
    T = waveln*1e-9/c
    omega_0 = 2*np.pi/T
    
    # remove low energies 
    energy = x * h_bar * omega_0
    x = x[energy > ionization_potential]
    y = y[energy > ionization_potential]
    
    # Three step fitting
    q1=0.0002*waveln**2
    initial_guesses = [q1,q1+50,-0.1,-60,-80]
    y=np.nan_to_num(y)
    fit, fit_cov = curve_fit(three_step,x,y,p0=initial_guesses)
    print(fit)
    #plateau_length.append(fit[0])
    print("{}".format(waveln))
    plt.plot(x,y)
    plt.plot(x,three_step(x,*fit))
    plt.show()





#%% calculate the area -100 as the base (useless)
lower=20
upper=280
area=[]
A=(upper-lower)*100
for wavelength in range (200,2001):
    f = interp1d(x_ensemble[wavelength-200],y_ensemble[wavelength-200])
    area.append(A+integrate.quad(f, lower,upper)[0])
    print('{}nm is done!'.format(wavelength))
         
lambda_=np.linspace(200,2000,1801)
plt.plot(lambda_,area)
plt.xlabel('wavelength (nm)')
plt.ylabel('area')
plt.title('Area under {}th to {}th harmonics vs wavelength'.format(lower,upper))
    

























