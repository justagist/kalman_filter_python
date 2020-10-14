# import math functions
# from math import *
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D

REAL_SENSOR_NOISE_SIG = 5.
SIM_PREDICTION_SIG = 7.

# measurements for mu and motions, U
# demo values (simulates real world measurements)
measurements = np.random.normal([0,0,0],[0.1,0.1,0.1],[20,3])
# no motion, i.e. x_{t+1} = x_t. This can also be changed to new resampled mean from sim

# initial parameters
measurement_sig = np.eye(3)*REAL_SENSOR_NOISE_SIG  # real robot pose sensor noise
# noise for prediction or resampled distribution noise from simulation
motion_sig = np.eye(3)*SIM_PREDICTION_SIG

mu = [75.,-50, 200]  # initial prediction mean
sig = np.diag([2.,2.,2.])  # initial prediction distribution noise

PLOT = True

def create_ellipsoid(ax, mean, cov, n_std=10.0, facecolor='none'):

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(np.linalg.inv(cov))
    radii = 1.0/np.sqrt(s)
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            # print mu
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + mu.flatten()


    # ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    return ax

if __name__ == "__main__":

    if PLOT:
        plt.ion()
        # fig, ax = plt.subplots(figsize=(12,12))
        fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
        ax = fig.add_subplot(111, projection='3d')
    #     x_axis = np.arange(-20, 20, 0.1)
        # plt.draw()
        # plt.pause(0.1)
        # input()
        
    kf = KalmanFilter(dim=3, x=mu, P=sig, R=measurement_sig, Q=motion_sig)

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
        measurement_sig -= np.eye(3)*0.25 # Assumes noise reduces; keep this constant by commenting out this line if needed.

        if PLOT:
            plt.cla()
            ax.axvline(c='grey', lw=1)
            ax.axhline(c='grey', lw=1)
            # g = []
            # for x in x_axis:
            #     g.append(f(mu, sig, x).flatten())

            # ax.plot(np.linspace(-1,1,10),np.linspace(-1,1,10))
            ax.scatter([mu[0,0]],[mu[1,0]],[mu[2,0]])
            # # print (ax)
            create_ellipsoid(ax, mu, sig)
            # plt.add_patch()

            ax.set_title("Step: {}".format(n+1))
            ax.set_ylim([-2,2])
            ax.set_xlim([-2,2])
            ax.set_zlim([-2,2])

            plt.draw()
            plt.pause(0.1)
    # print the final, resultant mu, sig
    print('\n')
    print('Final result: {} \n{}\n\n'.format(mu, sig))

    plt.draw()
    plt.pause(0)

