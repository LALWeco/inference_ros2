import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking keypoints in image space.

    The 4-dimensional state space

        x, y, vx, vy

    contains the keypoint position (x, y) and velocities.

    Object motion follows a constant velocity model. The keypoint location
    (x, y) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, odom_std_weight=1.0/40):
        ndim, dt = 2, 1.0  # 2D state: x,y position

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Increase process uncertainty for smoother tracking
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
        # Weight for odometry uncertainty
        self._odom_std_weight = odom_std_weight

    def initiate(self, measurement):
        """Create track from unassociated keypoint measurement.

        Parameters
        ----------
        measurement : ndarray
            Keypoint coordinates (x, y)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (4 dimensional) and covariance matrix (4x4
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement[:2]  # Only take x,y coordinates
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Increase initial uncertainty for velocities
        std = [
            2 * self._std_weight_position,  # x position uncertainty
            2 * self._std_weight_position,  # y position uncertainty
            10 * self._std_weight_velocity, # x velocity uncertainty
            10 * self._std_weight_velocity, # y velocity uncertainty
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, odom_vx=0, odom_vy=0, odom_uncertainty=None):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 4 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 4x4 dimensional covariance matrix of the object state at the
            previous time step.
        odom_vx : float
            Robot's x velocity in pixels per frame from odometry
        odom_vy : float
            Robot's y velocity in pixels per frame from odometry
        odom_uncertainty : Optional[tuple]
            Tuple of (σvx, σvy) representing uncertainty in odometry velocities.
            If None, uses default uncertainty based on odom_std_weight.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        # For stationary objects, expected velocity is opposite of camera motion
        mean[2:4] = [odom_vx, odom_vy]
        
        # Position uncertainty
        std_pos = [
            self._std_weight_position,
            self._std_weight_position,
        ]
        
        # Velocity uncertainty based on odometry uncertainty
        if odom_uncertainty is not None:
            σvx, σvy = odom_uncertainty
        else:
            σvx = σvy = self._odom_std_weight
            
        std_vel = [σvx, σvy]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (4 dimensional array).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        # Increase measurement noise to trust predictions more
        std = [
            self._std_weight_position,
            self._std_weight_position,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance, odom_vx=0, odom_vy=0, odom_uncertainty=None):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx4 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx4x4 dimensional covariance matrices of the object states at the
            previous time step.
        odom_vx : float
            Robot's x velocity in pixels per frame from odometry
        odom_vy : float
            Robot's y velocity in pixels per frame from odometry
        odom_uncertainty : Optional[tuple]
            Tuple of (σvx, σvy) representing uncertainty in odometry velocities.
            If None, uses default uncertainty based on odom_std_weight.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        # For stationary objects, expected velocity is opposite of camera motion
        mean[:, 2:4] = np.array([odom_vx, odom_vy])
        
        # Position uncertainty 
        std_pos = [
            self._std_weight_position * np.ones_like(mean[:, 0]),
            self._std_weight_position * np.ones_like(mean[:, 0]),
        ]
        
        # Velocity uncertainty based on odometry uncertainty
        if odom_uncertainty is not None:
            σvx, σvy = odom_uncertainty
            std_vel = [
                σvx * np.ones_like(mean[:, 0]),
                σvy * np.ones_like(mean[:, 0]),
            ]
        else:
            std_vel = [
                self._odom_std_weight * np.ones_like(mean[:, 0]),
                self._odom_std_weight * np.ones_like(mean[:, 0]),
            ]
        
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (4 dimensional).
        covariance : ndarray
            The state's covariance matrix (4x4 dimensional).
        measurement : ndarray
            The 2 dimensional measurement vector (x, y) containing
            the keypoint position.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=True, metric="maha"):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. The
        chi-square distribution has 2 degrees of freedom since we only use
        keypoint positions.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (4 dimensional).
        covariance : ndarray
            Covariance of the state distribution (4x4 dimensional).
        measurements : ndarray
            An Nx2 dimensional matrix of N measurements, each in
            format (x, y) representing keypoint position.
        only_position : Optional[bool]
            Always True for keypoint tracking since we only use position.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        measurements = measurements[:, :2]  # Only use x,y coordinates

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")
