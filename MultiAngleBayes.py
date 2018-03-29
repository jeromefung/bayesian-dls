import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import scipy.optimize
%matplotlib inline 
import emcee
import seaborn as sns
import pandas as pd

#Load in data
dls_data = np.loadtxt("100nm.dat", skiprows=1)

# Unpack the data
tau = dls_data[:,0]
intensity_20 = dls_data[:,1]
intensity_40 = dls_data[:,2]
intensity_60 = dls_data[:,3]
intensity_80 = dls_data[:,4]
intensity_100 = dls_data[:,5]
intensity_120 = dls_data[:,6]
intensity_140 = dls_data[:,7]
intensity_160 = dls_data[:,8]

#Load in the angles
dls_data = np.loadtxt("100nm.dat", usecols=(1,2,3,4,5,6,7,8))
phi = dls_data[0,:]
phi = phi*(np.pi/180) # scattering angle in radians

# Set a uniform uncertainity on the data as 5%
sig_g_20 = intensity_20*0.05
sig_g_40 = intensity_40*0.05
sig_g_60 = intensity_60*0.05
sig_g_80 = intensity_80*0.05
sig_g_100 = intensity_100*0.05
sig_g_120 = intensity_120*0.05
sig_g_140 = intensity_140*0.05
sig_g_160 = intensity_160*0.05

#Construct empty arrays to fill with all the data
all_angles= []
all_intensity =[]
all_sig = []
all_tau = []

# Construct an array that has each angle repeated for each data point
# Ex: [ 20, 20, 20, ....., 40, 40, 40, ....., 60, 60, 60, ....]
for j in range(0, 8):
    for i in range(0, len(intensity_20)):
        all_angles = np.append(all_angles, phi[j])

# Construct an array of all the intensity autocorrelation data        
list_of_data = [intensity_20, intensity_40, intensity_60, intensity_80, 
                intensity_100, intensity_120, intensity_140, intensity_160]
for item in list_of_data:        
    for i in range(0, len(item)):     
        all_intensity = np.append(all_intensity, item[i])

# Construct an array of all the errors for all the ac data
list_of_error = [sig_g_20, sig_g_40, sig_g_60, sig_g_80, sig_g_100, 
                 sig_g_120, sig_g_140, sig_g_160]
for item in list_of_error:        
    for i in range(0, len(item)):     
        all_sig = np.append(all_sig, item[i])

# Construct an array of the delay time tau that repeats for each data set
# Ex: (0.01, 0.1, 1, .... 0.01, 0.1, 1, .... )
for j in range(0, 8):
    for i in range(0, len(tau)):
        all_tau = np.append(all_tau, tau[i])
        
# Now define the log of the liklihood function 
# Assume that the prior is equal probablity
def log_prior(theta):
    # returns log of prior probability distribution
    # First unpack the model parameters, where B is baseline, D is the decay 
    # constant, and each A is the amplitude associated with 1 angle measurement
    B, D, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8 = theta 
    
    # Set a uniform prior, but within boundaries. Both D and B must be positive
    if 0.0 < D  and 0.0 < B < 10.0:
        return 0.0  # Since the probability is 1, returns 0.
    else:
        return -np.inf # Since the probability is 0, returns - infinity.
    
    
def log_likelihood(theta, tau, phi, g, sig_g):
    # returns the log of the likelihood function

    # tau: delay time 
    # g: measurements (autocorrelation function)
    # sig_g: uncertainties on measured data, set to be +- 5% of the value
    
    # Unpack the model parameters
    B, D, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8 = theta
    
    # Using the model A_i*np.e**(-D* (4 pi n / lam * sin(phi/2) )**2 *tau) + B
    # define the log of the likelihood function as
    # ln (L) = K - 1/2 * Sum [(y- function)^2 / sigma^2] 
    # ln (L) = K - 1/2 Chi^2   
    
    n = 1.333 # refractive index
    
    # To account for the different amplitudes, construct an array of all values
    # Each repeats for the length of the associated data set
    A = np.array([])
    
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_1)
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_2)   
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_3)
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_4)
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_5)
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_6)
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_7)
    for i in range(0, len(intensity_20)):
        A = np.append(A, A_8)

    # Define the angle dependent portion
    m = (4*np.pi * n * np.sin(phi/2))**2 
    
    residual = (g - A*np.e**(-D*m*tau) - B)**2
    chi_square = np.sum(residual/(sig_g**2))
    
    # the constant K is determined by the Gaussian function 
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*sig_g**2))) 
    
    return constant - 0.5*chi_square


def log_posterior(theta, tau, phi, g, sig_g):
    # returns log of posterior probability distribution
    
    # Unpack the model parameters
    B, D, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8 = theta
    
    # Bayes Theorem: Posterior = Prior * likelihood
    # Ln (Posterior) = Ln (Prior ) + Ln (Likelihood)
    return log_prior(theta) + log_likelihood(theta, tau, phi, g, sig_g)
    
# the model has 10 parameters; we'll use 50 walkers and 500 steps each with emcee
ndim = 10
nwalkers = 50
nsteps = 500

# set up the walkers in a "Gaussian ball" around the least-squares estimate
# The least squares fit estimate said that the amplitude is about 0.37
# C is about 18.35, and the baseline is usually set as 1.
ls_result = [1, 18.35, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37] 
            # B, D, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8
starting_positions = [ls_result + 1e-4*np.random.randn(ndim) for i in 
                        range(nwalkers)]

# set up the sampler object using emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                            args=(all_tau, all_angles, all_intensity, all_sig))

# run the sampler and use iPython's %time directive to tell us how long it took
%time sampler.run_mcmc(starting_positions, nsteps)
print('Done')

# Plot the walkers to see them converge on a value
fig, (ax_B, ax_D, ax_A_1, ax_A_2, ax_A_3, ax_A_4,
      ax_A_5, ax_A_6, ax_A_7, ax_A_8) = plt.subplots(10)
ax_B.set(ylabel='B')
ax_D.set(ylabel='D')
ax_A_1.set(ylabel='A_1')
ax_A_2.set(ylabel='A_2')
ax_A_3.set(ylabel='A_3')
ax_A_4.set(ylabel='A_4')
ax_A_5.set(ylabel='A_5')
ax_A_6.set(ylabel='A_6')
ax_A_7.set(ylabel='A_7')
ax_A_8.set(ylabel='A_8')
for i in range(10):
    sns.tsplot(sampler.chain[i,:,0], ax=ax_B)
    sns.tsplot(sampler.chain[i,:,1], ax=ax_D)
    sns.tsplot(sampler.chain[i,:,2], ax=ax_A_1)
    sns.tsplot(sampler.chain[i,:,3], ax=ax_A_2)
    sns.tsplot(sampler.chain[i,:,4], ax=ax_A_3)
    sns.tsplot(sampler.chain[i,:,5], ax=ax_A_4)
    sns.tsplot(sampler.chain[i,:,6], ax=ax_A_5)
    sns.tsplot(sampler.chain[i,:,7], ax=ax_A_6)
    sns.tsplot(sampler.chain[i,:,8], ax=ax_A_7)
    sns.tsplot(sampler.chain[i,:,8], ax=ax_A_8)
    
# Because it take a lot of steps before the walkers settle into the most likely 
# points, cut the samples to include only values after 300 stepes
samples = sampler.chain[:,300:,:]

# Reshape the samples into a 1D array
traces = samples.reshape(-1, ndim).T

# create a pandas DataFrame with labels to allow for calculations and graphing
parameter_samples = pd.DataFrame({'B': traces[0], 'D': traces[1], 'A_1': traces[2], 'A_2': traces[3], 'A_3': traces[4],
                                 'A_4': traces[5], 'A_5': traces[6], 'A_6': traces[7], 'A_7': traces[8], 'A_8': traces[9]})

# calculate the most likely value and the uncertainties using pandas
q = parameter_samples.quantile([0.16,0.50,0.84], axis=0)

# Finally, solve for the hydrodynamic radius

k = 1.38065e-23 # Boltzmann's constant, J/K
T = 293.08 # Kelvin, roomtemp
eta = 10.016e-4 # viscosity of water, Pa s at temperature of 293.08 Kelvin

# Solve for true diffusion coeffient, since we fit to D/lam^2
# D = D / lam^2

lam = 658e-9 # wavelength of laser in meters
D = 20.407227*lam**2

R = k*T / (6*np.pi*eta*D)
print "The hydrodynamic radius: ", R