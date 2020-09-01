# import math functions
# from math import *
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

REAL_SENSOR_NOISE_SIG = 5.
SIM_PREDICTION_SIG = 7.

# measurements for mu and motions, U
# demo values (simulates real world measurements)
measurements = np.random.normal([0,0],[0.1,0.1],[20,2])
# no motion, i.e. x_{t+1} = x_t. This can also be changed to new resampled mean from sim

# initial parameters
measurement_sig = np.eye(2)*REAL_SENSOR_NOISE_SIG  # real robot pose sensor noise
# noise for prediction or resampled distribution noise from simulation
motion_sig = np.eye(2)*SIM_PREDICTION_SIG

mu = [75.,-50]  # initial prediction mean
sig = np.diag([2.,2.])  # initial prediction distribution noise

PLOT = True

def create_ellipse(ax, mean, cov, n_std=10.0, facecolor='none'):

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=[0,0,1,0.1], lw=1,
                      edgecolor='b'
                )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # print (scale_x)
    mean_x = mean[0,0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1,0]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    # print(ax.transData)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

if __name__ == "__main__":

    if PLOT:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12,12))
    #     x_axis = np.arange(-20, 20, 0.1)
        # plt.draw()
        # plt.pause(0.1)
        # input()
        
    kf = KalmanFilter(dim=2, x=mu, P=sig, R=measurement_sig, Q=motion_sig)

    ## TODO: Loop through all measurements/motions
    # this code assumes measurements and motions have the same length
    # so their updates can be performed in pairs

    for n in range(measurements.shape[0]):
        # motion update, with uncertainty
        mu, sig = kf.predict(Q=motion_sig)
        print('Predict: {} \n{}\n\n'.format(mu, sig))
        # measurement update, with uncertainty
        mu, sig = kf.update(y=measurements[n], R=measurement_sig)
        print('Update: {} \n{}\n\n'.format(mu, sig))
        # print (mu)
        measurement_sig -= np.eye(2)*0.25

        if PLOT:
            plt.cla()
            ax.axvline(c='grey', lw=1)
            ax.axhline(c='grey', lw=1)
            # g = []
            # for x in x_axis:
            #     g.append(f(mu, sig, x).flatten())

            # ax.plot(np.linspace(-1,1,10),np.linspace(-1,1,10))
            ax.scatter([mu[0,0]],[mu[1,0]])
            # # print (ax)
            create_ellipse(ax, mu, sig)
            # plt.add_patch()

            ax.set_title("Step: {}".format(n+1))
            ax.set_ylim([-30,30])
            ax.set_xlim([-30,30])

            plt.draw()
            plt.pause(0.1)
    # print the final, resultant mu, sig
    print('\n')
    print('Final result: {} \n{}\n\n'.format(mu, sig))

    plt.draw()
    plt.pause(0)

