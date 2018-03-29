import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import scipy.optimize

# Applying simple exponential fit to multiangle data
dls_data = np.loadtxt("100nm.dat", skiprows=1)
# Data is in columns based on angle, first row labels these columns

tau = dls_data[:,0]
intensity_20 = dls_data[:,1]
intensity_40 = dls_data[:,2]
intensity_60 = dls_data[:,3]
intensity_80 = dls_data[:,4]
intensity_100 = dls_data[:,5]
intensity_120 = dls_data[:,6]
intensity_140 = dls_data[:,7]
intensity_160 = dls_data[:,8]

def fitmodel(t, C, A, B):
    return A*np.e**(-C*t) + B

popt1, pcov1 = scipy.optimize.curve_fit(fitmodel, tau, intensity_20)
popt2, pcov2 = scipy.optimize.curve_fit(fitmodel, tau, intensity_40)
popt3, pcov3 = scipy.optimize.curve_fit(fitmodel, tau, intensity_60)
popt4, pcov4 = scipy.optimize.curve_fit(fitmodel, tau, intensity_80)
popt5, pcov5 = scipy.optimize.curve_fit(fitmodel, tau, intensity_100)
popt6, pcov6 = scipy.optimize.curve_fit(fitmodel, tau, intensity_120)
popt7, pcov7 = scipy.optimize.curve_fit(fitmodel, tau, intensity_140)
popt8, pcov8 = scipy.optimize.curve_fit(fitmodel, tau, intensity_160)

# Plot linear relationship between q^2 and fit constant Gamma to find D

gamma = np.array([popt1[0], popt2[0], popt3[0], popt4[0], popt5[0], 
                  popt6[0], popt7[0], popt8[0]])

#Load in the angles
dls_data = np.loadtxt("100nm.dat", usecols=(1,2,3,4,5,6,7,8))
theta = dls_data[0,:]
theta = theta*(np.pi/180) # scattering angle in radians

lam = 658e-9 # wavelength in meters
n = 1.333 # refractive index

q = ((4*np.pi * n) /(lam)) * np.sin(theta/2)

def straight_line_model(x, A, B):
    '''
    Model function for a straight-line fit with y-intercept A and slope B.
    '''
    return A + B * x

popt, pcov = scipy.optimize.curve_fit(straight_line_model, q**2, gamma)

# The slope gives the value for diffusion coefficient D

# To find the radius, use the relationship D = kT/(6*pi*eta*R)

k = 1.38065e-23 # Boltzmann's constant, J/K 
T = 293.08 # Kelvin, roomtemp
eta = 10.016e-4 # viscosity of water, Pa s at temperature of 293.08 Kelvin

D = popt[1]

R = k*T / (6*np.pi*eta*D)
print "The calculated hydrodynamic radius: ", R