import lap
import numpy as np
import scipy
from scipy.spatial.distance import cdist

from bytetracker import kalman_filter

def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q

def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def keypoint_distance(atracks, btracks):
    """
    Compute cost based on Euclidean distance between keypoints
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))
    
    if isinstance(atracks[0], np.ndarray):
        akeypoints = atracks
        bkeypoints = btracks
    else:
        akeypoints = np.stack([track.keypoint for track in atracks])  # N x 2
        bkeypoints = np.stack([track.keypoint for track in btracks])  # M x 2
    
    # Ensure 2D arrays
    if akeypoints.ndim == 1:
        akeypoints = akeypoints.reshape(1, -1)
    if bkeypoints.ndim == 1:
        bkeypoints = bkeypoints.reshape(1, -1)
    
    cost_matrix = cdist(akeypoints, bkeypoints)
    return cost_matrix

def fuse_score(cost_matrix, detections):
    """
    Fuse cost matrix with detection scores
    """
    if cost_matrix.size == 0:
        return cost_matrix
    dist_sim = np.exp(-cost_matrix/100)  # Convert distances to similarities
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = dist_sim * det_scores
    cost_matrix = -np.log(fuse_sim + 1e-10)  # Convert similarities back to costs
    return cost_matrix

def fuse_motion(kf, cost_matrix, tracks, detections, only_position=True, lambda_=0.98):
    """
    Fuse cost matrix with Kalman filter motion predictions
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2  # Always 2 for keypoint tracking
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.array([det.keypoint for det in detections])
    
    # Ensure measurements is 2D array
    if measurements.ndim == 1:
        measurements = measurements.reshape(1, -1)
        
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, 
            only_position=True, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
