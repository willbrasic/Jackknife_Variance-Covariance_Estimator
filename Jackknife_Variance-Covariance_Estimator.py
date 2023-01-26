""" 
This program simulates data and then uses the jackknife resampling method
to compute the estimator of the variance-covariance matrix for
the OLS estimates of a linear regression model. 

"""

__author__ = "William Brasic"
__email__ =  "wbrasic@arizona.edu"


"""

Preliminaries

"""

# importing necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
import statistics
import random 
from scipy.stats import t,f

# arrays are printed up to 6 digits and without scientific notation
np.set_printoptions(precision = 6)
np.set_printoptions(suppress = True)


# creating regression functin to use in jackknife function
def regression(data):
    """
    This function runs a regression and returns the estimated parameters

    Args:
        data (numpy array): array of outcome and covariate data

    Returns:
        dataframe: regression parameters
    """
    X = data[:, 1:]
    y = data[:, 0]
    return sm.OLS(y, sm.add_constant(X)).fit().params

# creating jackknife function
def jackknife(data, regressionFunction):
    """
    This function implements the leave-one-out estimator

    Args:
        data (array): array of outcome and covariate data
        fn (function): _description_

    Returns:
        Dictionary: estimates of beta 1000 times
    """
    # instantiating dictionaries to add data to
    jackknifeSample = {}
    jackknifeEstimates = {}
    # looping from i=0 to the length of the data (1000)
    for i in range(len(data)):
        # delete row ith row from dataframe
        jackknifeSample[i] = np.delete(arr = data, obj = i, axis = 0)
        # OLS estimates without ith row
        jackknifeEstimates[i] = regressionFunction(jackknifeSample[i])
    # returning the estimates for leave-one-out regression
    return jackknifeEstimates

# setting seed for reproducibility
np.random.seed(1024)

# number of observations
n = 1000

# numer of covariates
k = 4

# parameters
beta0, beta1, beta2, beta3, beta4 = 1, 0.13, -0.03, -0.02, -0.02

# random number of males (1) and females (0) with 0.5 probability chance of each
numOfMales = sum(np.random.RandomState(seed = 1024).choice(a = [1, 0], 
                                                           p = [0.5, 0.5], 
                                                           size = n))
numOfFemales = n - numOfMales

# generating male education data
maleEducation = np.random.choice(a = [10, 12, 14, 16, 18, 20], 
                                 p = [0.1, 0.3, 0.2, 0.35, 0.03, 0.02], 
                                 size = numOfMales)

# generating female education data
femaleEducation = np.random.choice(a = [10, 12, 14, 16, 18, 20], 
                                   p = [0.05, 0.3, 0.15, 0.42, 0.06, 0.02], 
                                   size = numOfFemales)

# merging the two created arrays
x1 = np.concatenate((maleEducation, femaleEducation))

# creating x3
x3 = np.random.RandomState(1024).choice(a = [1, 0], p = [0.2, 0.8], size = n)

# error. each error comes from a differently scaled normal distribution 
# where the variance is dependent on the individual's x1 value. 
# Hence, errors are heteroskedastic. 
e = np.random.normal(loc = 0, scale = np.sqrt(225/(x1**2)))

# creating dataframe
df = pd.DataFrame({'x1': x1, 'e': e, 'x3': x3})

# if index < 500 the observation is male. Otherwise, the observation is female
df['x2'] = df.index.map(lambda x: 1 if x <= numOfMales else 0)

# shuffling the rows in a random fashion using the entire dataset and resetting the index
# import for machine learning; not necessay here
df = df.sample(frac = 1, random_state = 1024).reset_index(drop = True)

# creating y
df['y'] = beta0 + beta1*df['x1'] + beta2*df['x2'] + beta3*df['x1']*df['x2'] + beta4*df['x3'] + e

# creating x1*x2
df['x1x2'] = df['x1']*df['x2']

# reorganizing dataframe
df = df[['y', 'x1', 'x2', 'x1x2', 'x3', 'e']]

# creating the model allowing for the heteroskedasticity present because of the DGP
model = sm.formula.ols(formula = 'y ~ x1 + x2 + x1x2 + x3', data = df).fit()
#print(model.summary())

# getting ols estimates mannualy
X = np.array([[1]*n, df['x1'], df['x2'], df['x1x2'], df['x3']]).T
betaHat = np.linalg.inv(X.T @ X) @ X.T @ df['y']

# getting variance-covariance matrix manually
eHat = df['y'] - X @ betaHat
vcMatrix = np.linalg.inv(X.T @ X)*sum(eHat**2)/(n-k-1)

# obtaining standard error of betaHat
seBetaHat = np.array([math.sqrt(vcMatrix[i,j])
                      for i in range(k+1) for j in range(k+1) if i == j])

# obtaining t stats for H_0: beta = 0 vs. H_A: beta != 0
tStatBetaHat = np.array(betaHat/seBetaHat)

# two sided p-value is such that 2*F(T > |t|) = 2*(1 - F(T <= |t|))
pValues = 2*(1 - t.cdf(x = abs(tStatBetaHat), df=n-k-1))

# 95% confidence intervals
betaCI = np.array([(betaHat[i] - abs(t.ppf(q = 0.025, df = n-k-1))*seBetaHat[i],
                    betaHat[i] + abs(t.ppf(q = 0.025, df = n-k-1))*seBetaHat[i]) 
                   for i in range(k+1)])

# creating data as numpy array. This fixes a problem I was running into
# when trying to use normal dataframe with jackknife function
df1 = df.iloc[:,0:5].to_numpy()

# call jackknife function
jackknifeResults = jackknife(data = df1, regressionFunction = regression)

# obtaining estimates for each paramter from leave-one-out-estimation
# e.g, jackBetaHat[0] corresponds to estimates for beta0 while jackBetaHat[4]
# corresponds to estimates for beta4
jackBetaHat = np.array([[jackknifeResults[i][j] \
    for i in range(len(jackknifeResults.items()))] for j in range(5)])

# obtaining the mean of each estimate
betaHatBar = np.array([np.mean(jackBetaHat[i]) for i in range(len(jackBetaHat))])

# jackknife variance estimator of estimator of beta
# scaled by (n-1)/n as suggested by Hansen
varJackBetaHat = ((n-1)/n)*np.array([[sum((jackBetaHat[j] - betaHatBar[j])*(jackBetaHat[i] - betaHatBar[i]))
                                      for j in range(len(betaHatBar))] for i in range(len(betaHatBar))])

# standard error of variance estimate of estimator of beta
seJackBetaHat = np.array([math.sqrt(varJackBetaHat[i][i]) 
                          for i in range(len(varJackBetaHat))])

# obtaining t stats for H_0: beta = 0 vs. H_A: beta != 0
tStatJackBetaHat = betaHat/seJackBetaHat

# two sided p-value is such that 2*F(T > |t|) = 2*(1 - F(T <= |t|))
pValuesJackBetaHat = 2*(1 - t.cdf(x = abs(tStatJackBetaHat), df=n-k-1))

# 95% confidence intervals
jackBetaCI = np.array([(betaHat[i] - abs(t.ppf(q = 0.025, df = n-k-1))*seJackBetaHat[i],
                        betaHat[i] + abs(t.ppf(q = 0.025, df = n-k-1))*seJackBetaHat[i]) 
                        for i in range(k+1)])
