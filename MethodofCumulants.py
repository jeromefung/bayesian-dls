import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import scipy
import scipy.optimize
import scipy.misc
import scipy.stats

# Import the data as an array
dls_data = np.loadtxt("CorrelationDLS.asc", skiprows=27)
tau = dls_data[:,0]
intensity_ac = dls_data[:,1]

# Using the expansion
# g^2 = B + beta* e^{-2*Gamma\tau}*(1 + mu_2/2! *tau^2 - mu_3/3!*tau^3 )

#2nd Order Cumulant Fit
def fitmodel_2nd_order(t, B, beta, Gamma, mu_2):
    return B + beta*np.e**(-2*Gamma*t)* (1 + (mu_2/2 *t**2))

popt1, pcov1 = scipy.optimize.curve_fit(fitmodel_2nd_order, tau, intensity_ac)

# 3rd Order Cumulant Fit
def fitmodel_3rd_order(t, B, beta, Gamma, mu_2, mu_3):
    return B + beta*np.e**(-2*Gamma*t)* (1 + (mu_2/2 *t**2) - (mu_3/6 * t**3))

popt2, pcov2 = scipy.optimize.curve_fit(fitmodel_3rd_order, tau, intensity_ac)

# Use these fits to solve for the hydrodynamic radius
# Data taken at a scattering angle of 90 degrees with a 632.8 nm HeNe laser.

lam = 632.8e-9 # wavelength in meters
theta = np.pi # scattering angle in radians
n = 1.33200 # refractive index

q = (4*np.pi * n) / (lam) * np.sin(theta/2)

Gamma1 = popt1[2] # exponetial constant from 2nd order fit
D1 = (Gamma1 / q**2) /0.001 # Diffusion constant with conversion from ms to s

Gamma2 = popt2[2] # exponetial constant from 3rd order fit
D2 = (Gamma2 / q**2) /0.001

R1 = k*T / (6*np.pi*eta*D1)
print "The hydrodynamic radius from the 2nd order fit: ", R1

R2 = k*T / (6*np.pi*eta*D2)
print "The hydrodynamic radius from the 3rd order fit: ", R2