import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize

# Import the data as an array
dls_data = np.loadtxt("CorrelationDLS.asc", skiprows=27)

# First few rows include time, temp, etc.
# data is now in 2 colums: delay time and intensity autocorrelation function

# Unpack imported data
tau = dls_data[:,0]
intensity_ac = dls_data[:,1]

# Fitting data to model g(t) = A*e**(-C*t) + B
def fitmodel(t, C, A, B):
    return A*np.e**(-C*t) + B

popt, pcov = scipy.optimize.curve_fit(fitmodel, tau, intensity_ac)
print ("best-fit parameters: ", popt)
print pcov

popt_uncertainties = np.sqrt(np.diag(pcov))
print popt_uncertainties

frac_uncertainties = (popt_uncertainties / popt)
print frac_uncertainties

# Use this constant to determine the diffusion constant D
# Data taken at a scattering angle of 90 degrees with a 632.8 nm HeNe laser

lam = 632.8e-9 # wavelength in meters
theta = np.pi # scattering angle in radians
n = 1.33200 # refractive index of water

q = (4*np.pi * n) / (lam) * np.sin(theta/2)

C = popt[0] # exponetial constant from fit
# C = 2*D*q^2
D = (C / (2*q**2)) /0.001 # Diffusion constant

# To find the hydrodynamic radius, use the relationship D = kT/(6*pi*eta*R)

k = 1.38065e-23 # Boltzmann's constant, J/K (joules per kelvin)
T = 297.94999 # Kelvin, from textfile
eta = 8.9e-4 # viscosity of water, Pa s (pascal seconds)

R = k*T / (6*np.pi*eta*D)
R_unc = R * frac_uncertainties[0]
print "The hydrodynamic radius found from the fit is", str(R)
print "The uncertainity of R is +-", str(R_unc)