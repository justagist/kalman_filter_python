# import math functions
# from math import *
import matplotlib.pyplot as plt
import numpy as np
from kalman_filter import KalmanFilter

REAL_SENSOR_NOISE_SIG = 5.
SIM_PREDICTION_SIG = 7.

# measurements for mu and motions, U
# demo values (simulates real world measurements)
measurements = [5.6, 6., 6.1, 5.7, 5.8, 5.6,
                6., 6.1, 5.7, 5.8, 5.6, 6., 6.1, 5.7, 5.8]
# no motion, i.e. x_{t+1} = x_t. This can also be changed to new resampled mean from sim

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
    coefficient = 1.0 / np.sqrt(2.0 * np.pi *sigma2)
    exponential = np.exp(-0.5 * (x-mu) ** 2 / sigma2)
    return coefficient * exponential



if __name__ == "__main__":
        
    if PLOT:
        plt.ion()
        x_axis = np.arange(-20, 20, 0.1)
        # plt.draw()
        # plt.pause(0.1)
        # input()
    kf = KalmanFilter(dim=1, x=mu, P=sig, R=measurement_sig, Q=motion_sig)

    ## TODO: Loop through all measurements/motions
    # this code assumes measurements and motions have the same length
    # so their updates can be performed in pairs

    for n in range(len(measurements)):
        # motion update, with uncertainty
        mu, sig = kf.predict(Q=motion_sig)
        print('Predict: [{}, {}]'.format(mu, sig))
        # measurement update, with uncertainty
        mu, sig = kf.update(y=measurements[n], R=measurement_sig)
        print('Update: [{}, {}]'.format(mu, sig))
        measurement_sig -= 0.25 # Assumes noise reduces; keep this constant by commenting out this line if needed.

        if PLOT:
            plt.cla()
            g = []
            for x in x_axis:
                g.append(f(mu, sig, x).flatten())

            plt.plot(x_axis,g)
            plt.title("Step: {}".format(n+1))
            plt.ylim([0,0.5])
            
            plt.draw()
            plt.pause(0.1)
    # print the final, resultant mu, sig
    print('\n')
    print('Final result: [{}, {}]'.format(mu, sig))

    plt.draw()
    plt.pause(0)
