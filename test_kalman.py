# import math functions
from math import *
import matplotlib.pyplot as plt
import numpy as np

REAL_SENSOR_NOISE_SIG = 5.
SIM_PREDICTION_SIG = 7.

# measurements for mu and motions, U
# demo values (simulates real world measurements)
measurements = [5.6, 6., 6.1, 5.7, 5.8, 5.6,
                6., 6.1, 5.7, 5.8, 5.6, 6., 6.1, 5.7, 5.8]
# no motion, i.e. x_{t+1} = x_t. This can also be changed to new resampled mean from sim
motions = [0., 0., 0.0, 0., 0., 0., 0., 0.0, 0., 0., 0., 0., 0.0, 0., 0.]

# initial parameters
measurement_sig = REAL_SENSOR_NOISE_SIG  # real robot pose sensor noise
motion_sig = SIM_PREDICTION_SIG # noise for prediction or resampled distribution noise from simulation

mu = 75. # initial prediction mean
sig = 1. # initial prediction distribution noise

PLOT = True

# gaussian function
def f(mu, sigma2, x):
    ''' f takes in a mean and squared variance, and an input x
       and returns the gaussian value.'''
    coefficient = 1.0 / sqrt(2.0 * pi *sigma2)
    exponential = exp(-0.5 * (x-mu) ** 2 / sigma2)
    return coefficient * exponential

# the update function
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    # print(mean1, var1, mean2, var2)
    # input
    new_var = 1/(1/var2 + 1/var1)
    return [new_mean, new_var]


# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2

    return [new_mean, new_var]


if PLOT:
    plt.ion()
    x_axis = np.arange(-20, 20, 0.1)


## TODO: Loop through all measurements/motions
# this code assumes measurements and motions have the same length
# so their updates can be performed in pairs
for n in range(len(measurements)):
    # measurement update, with uncertainty
    # motion update, with uncertainty
    mu, sig = predict(mu, sig, motions[n], motion_sig)
    print('Predict: [{}, {}]'.format(mu, sig))
    mu, sig = update(mu, sig, measurements[n], measurement_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    motion_sig-=0.5

    if PLOT:
        plt.cla()
        g = []
        for x in x_axis:
            g.append(f(mu, sig, x))

        plt.plot(x_axis,g)
        plt.draw()
        plt.pause(0.1)
# print the final, resultant mu, sig
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))

plt.draw()
plt.pause(0)