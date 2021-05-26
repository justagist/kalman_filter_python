import numpy as np
import filterpy.kalman as FPyKal


class KalmanFilter(FPyKal.KalmanFilter):
    """
    Using filterpy KalmanFilter instead. Wrapping to support with original class's parameters for compatibility with old code.

    !! All new code should directly use filterpy.kalman.KalmanFilter instead of this class. !!
    """

    def __init__(self, dim=3, x=None, P=None, A=None, H=None, Q=None, R=None):
        """
        Initialise the KF object with the provided parameters

        :param dim: state dimensionality, defaults to 3
        :type dim: int, optional
        :param x: initial state, defaults to origin of the state space
        :type x: [float]*dim (OR) np.ndarray ([dim,1]), optional
        :param P: Initial covariance (noise) in state estimate, defaults to identity
        :type P: np.ndarray or [[float]], shape:([dim,dim]), optional
        :param A: state transition matrix, defaults to identity (zero motion)
        :type A: np.ndarray or [[float]], shape:([dim,dim]), optional
        :param H: mapping from state space to measurement space, defaults to identity
        :type H: np.ndarray or [[float]], shape:([y_dim,dim]), optional
        :param Q: Prediction noise (covariance), defaults to identity
        :type Q: np.ndarray or [[float]], shape:([dim,dim]), optional
        :param R: sensor noise (covariance), defaults to identity
        :type R: np.ndarray or [[float]], shape:([y_dim,y_dim]), optional
        """

        FPyKal.KalmanFilter.__init__(self, dim_x=dim, dim_z=dim)
        if x is not None:
            self.x = x
        if P is not None:
            self.P = P if isinstance(P,np.ndarray) else self.P*P
        self.H = np.asarray(H).reshape(
            [dim, dim]) if H is not None else np.eye(dim)
        self.Q = np.asarray(Q).reshape(
            [dim, dim]) if Q is not None else np.eye(dim)
        self.R = np.asarray(R).reshape(
            [dim, dim]) if R is not None else np.eye(dim)
        self.F = np.asarray(A).reshape(
            [dim, dim]) if A is not None else np.eye(dim)

    def predict(self, A=None, Q=None):
        """
        Motion update (prediction step) [NOTE: No control update]

        :param A: State transition function, defaults to previously defined value
        :type A: np.ndarray or [[float]], shape:([dim,dim]), optional
        :param Q: [description], defaults to previously defined value
        :type Q: np.ndarray or [[float]], shape:([dim,dim]), optional
        :return: predicted state mean, predicted state covariance
        :rtype: [np.ndarray, np.ndarray]
        """
        FPyKal.KalmanFilter.predict(self, F=A, Q=Q)
        return self.x.copy().reshape([-1,1]), self.P.copy()

    def update(self, y, R=None, H=None):
        """
        Kalman update step

        :param y: new measurement
        :type y: np.ndarray or [[float]], shape:([y_dim,1]), optional
        :param R: sensor noise (covariance), defaults to previously defined value
        :type R: np.ndarray or [[float]], shape:([y_dim,y_dim]), optional
        :param H: mapping from state space to measurement space, defaults to previously defined value
        :type H: np.ndarray or [[float]], shape:([y_dim,dim]), optional
        :return: updated state mean, updated state covariance
        :rtype: [np.ndarray, np.ndarray]
        """
        FPyKal.KalmanFilter.update(self, z=y, R=R, H=H)
        return self.x.copy().reshape([-1,1]), self.P.copy()

if __name__ == "__main__":


    REAL_SENSOR_NOISE_SIG = 5.
    SIM_PREDICTION_SIG = 7.
    # measurements for mu and motions, U
    # demo values (simulates real world measurements)
    measurements = [5.6, 6., 6.1, 5.7, 5.8, 5.6,
                    6., 6.1, 5.7, 5.8, 5.6, 6., 6.1, 5.7, 5.8]
    # no motion, i.e. x_{t+1} = x_t. This can also be changed to new resampled mean from sim
    motions = [0., 0., 0.0, 0., 0., 0., 0., 0.0, 0., 0., 0., 0., 0.0, 0., 0.] # change these values to give motion

    # initial parameters
    measurement_sig = REAL_SENSOR_NOISE_SIG  # real robot pose sensor noise
    # noise for prediction or resampled distribution noise from simulation
    motion_sig = SIM_PREDICTION_SIG

    mu = 75.  # initial prediction mean
    sig = 1.  # initial prediction distribution noise

    kf = KalmanFilter(dim=1, x=mu, P=sig, R=measurement_sig, Q=motion_sig)

    for n in range(len(measurements)):
        # motion update, with uncertainty
        mu, sig = kf.predict(Q=motion_sig)
        print('Predict: [{}, {}]'.format(mu, sig))
        # measurement update, with uncertainty
        mu, sig = kf.update(y=measurements[n], R=measurement_sig)
        print('Update: [{}, {}]'.format(mu, sig))
        motion_sig -= 0.5 # Assumes noise reduces; keep this constant by commenting out this line if needed.
