# Import the data as an array
dls_data = np.loadtxt("CorrelationDLS.asc", skiprows=27)
# Unpack imported data
tau = dls_data[:,0]
intensity_ac = dls_data[:,1]
# Assume intensity autocorrelation function has a constant uncertainty of +-5% 
sig_y = intensity_ac*0.05

# Now define the log of the liklihood function using uniform prior
def log_prior(theta):
    # returns log of prior probability distribution
    A, C = theta # unpack the model parameters
    
    # Set a uniform prior, but within boundaries. 
    if 0 < A < 10.0 and 0.0 < C < 500.0:
        return 0.0  # Since the probability is 1, this returns 0.
    else:
        return -np.inf # Since the probability is 0
        
def log_likelihood(theta, x, y, sig_y):
    # returns the log of the likelihood function

    # theta: model parameters (specified as a tuple)
    # x: angles
    # y: measured tau
    # sig_y: uncertainties on measured data, set to be +- 5% of the value
    
    A, C = theta # unpack the model parameters
    
    # Using the model A*np.e**(-C*t), define the log of the likelihood function 
    # ln (L) = K - 1/2 * Sum [(y- function)^2 / sigma^2] 
    # ln (L) = K - 1/2 Chi^2
    # Based on derivation in Hogg, Bovy, and Lang paper
    
    residual = (y - A*np.e**(-C*x))**2
    chi_square = np.sum(residual/(sig_y**2))
    
    # the constant K is determined by the Gaussian function 
    constant = np.sum(np.log(1/np.sqrt(2.0*np.pi*sig_y**2)))
    return constant - 0.5*chi_square

def log_posterior(theta, x, y, sig_y):
    # returns log of posterior probability distribution
    A, C = theta
    
    # Bayes Theorem: Posterior = Prior * likelihood
    # Ln (Posterior) = Ln (Prior ) + Ln (Likelihood)
    return log_prior(theta) + log_likelihood(theta, x, y, sig_y)

# the model has 2 parameters; use 50 walkers and 500 steps each
ndim = 2
nwalkers = 50
nsteps = 500

# set up the walkers in a "Gaussian ball" around the least-squares estimate
# The least squares fit estimate: A = 0.9244, C = 33.155
ls_result = [0.92445056, 33.15515352] # A, C
starting_positions = [ls_result + 
                        1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# set up the sampler object using emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                    args=(tau, intensity_ac, sig_y))
%time sampler.run_mcmc(starting_positions, nsteps)

fig, (ax_A, ax_C) = plt.subplots(2)
ax_A.set(ylabel='A')
ax_C.set(ylabel='C')

for i in range(10):
    sns.tsplot(sampler.chain[i,:,0], ax=ax_A)
    sns.tsplot(sampler.chain[i,:,1], ax=ax_C)
# It takes about 100 steps for the walkers to settle
# Trim the data to include only steps after 100
samples = sampler.chain[:,100:,:]

# reshape the samples into a 1D array where the colums are A, C
traces = samples.reshape(-1, ndim).T
parameter_samples = pd.DataFrame({'A': traces[0], 'C': traces[1]})