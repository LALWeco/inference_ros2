import numpy as np
import torch
from bytetracker import matching
from bytetracker.basetrack import BaseTrack, TrackState
from bytetracker.kalman_filter import KalmanFilter

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, detection_data, score, cls):
        super().__init__()  # Add this line to properly initialize the base class
        
        # Extract keypoint from detection data (last 3 numbers, ignore last one)
        self.keypoint = np.array([detection_data[-3], detection_data[-2]], dtype=float).reshape(2)
        
        # Store original detection for output
        self._detection = np.asarray(detection_data[:4], dtype=float).reshape(4)
        
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        # Add track history buffer (store last N points)
        self.track_history = []
        self.history_len = 30  # Keep last 30 points
        # Add previous track history for restoration
        self.prev_track_history = []

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[2:4] = 0  # Zero out velocity if not tracked
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.keypoint = self.mean[:2]  # Update keypoint from Kalman state

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][2:4] = 0  # Zero out velocity if not tracked
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                stracks[i].keypoint = mean[:2]  # Update keypoint from Kalman state

    def activate(self, kalman_filter, frame_id, odom_vx=0, odom_vy=0):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # Initialize mean with opposite of camera motion for stationary objects
        mean, covariance = self.kalman_filter.initiate(self.keypoint)
        mean[2:4] = [odom_vx, odom_vy]  # Set initial velocity opposite to camera motion
        self.mean = mean
        self.covariance = covariance

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        # Initialize track history with first point
        self.track_history = [self.keypoint.copy()]

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track with new keypoint detection"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.keypoint
        )
        self.keypoint = new_track.keypoint
        self._detection = new_track._detection  # Update bbox for output
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

        # Restore previous history and add current point
        if len(self.prev_track_history) > 0:
            # Calculate time gap
            time_gap = frame_id - self.end_frame
            if time_gap < 30:  # Only restore if gap is not too large
                self.track_history = self.prev_track_history
            else:
                self.track_history = []
        self.track_history.append(self.keypoint.copy())
        
        # Clear previous history
        self.prev_track_history = []

    def update(self, new_track, frame_id):
        """Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Update with new keypoint detection
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.keypoint
        )
        self.keypoint = new_track.keypoint
        self._detection = new_track._detection  # Update bbox for output
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        
        # Update track history
        self.track_history.append(self.keypoint.copy())
        if len(self.track_history) > self.history_len:
            self.track_history.pop(0)

    def mark_lost(self):
        """Store track history before marking as lost"""
        self.state = TrackState.Lost
        self.end_frame = self.frame_id
        # Store current history before losing track
        self.prev_track_history = self.track_history.copy()

    @property
    def tlwh(self):
        """Get bounding box for output compatibility"""
        return self._detection.copy()

    @property
    def tlbr(self):
        """Get bounding box in tlbr format for output compatibility"""
        ret = self._detection.copy()
        return ret

class BYTETracker(object):
    def __init__(self, track_thresh=0.45, track_buffer=25, match_thresh=0.8, frame_rate=30, odom_std_weight=1.0/40):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_buffer = track_buffer

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter(odom_std_weight=odom_std_weight)

    def update(self, dets, _, odom_vx=0, odom_vy=0, odom_uncertainty=None):
        """Update tracks using keypoint detections
        dets: Detections with format [x1,y1,x2,y2,score,cls,kx,ky,_]
              where kx,ky are keypoint coordinates
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Get complete detection data including keypoints
        det_data = dets.numpy() if isinstance(dets, torch.Tensor) else dets  # Convert to numpy if tensor
        scores = det_data[:, 4]
        classes = det_data[:, 5]

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = det_data[inds_second]
        dets = det_data[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(det_data, s, c) 
                for (det_data, s, c) in zip(dets, scores[remain_inds], classes[remain_inds])
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # for det in strack_pool:
        #     print(det.track_id, det.mean)
        # # Print mean of track mean states
        print("Mean of track mean states:")
        print(np.mean([det.mean for det in strack_pool], axis=0))

        # Predict the current location with KF
        for track in strack_pool:
            track.mean, track.covariance = self.kalman_filter.predict(
                track.mean, 
                track.covariance,
                odom_vx=odom_vx,
                odom_vy=odom_vy,
                odom_uncertainty=odom_uncertainty
            )
            track.keypoint = track.mean[:2]  # Update keypoint from Kalman state
        
        # Match using keypoint distance
        dists = matching.keypoint_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(det_data, s, c)
                for (det_data, s, c) in zip(
                    dets_second, 
                    scores[inds_second],
                    classes[inds_second]
                )
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i] for i in u_track 
            if strack_pool[i].state == TrackState.Tracked
        ]
        
        dists = matching.keypoint_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks"""
        detections = [detections[i] for i in u_detection]
        dists = matching.keypoint_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, odom_vx, odom_vy)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # Use keypoint distance for duplicate removal
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # Return activated tracks directly
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks based on keypoint distance"""
    pdist = matching.keypoint_distance(stracksa, stracksb)
    pairs = np.where(pdist < 10.0)  # Distance threshold in pixels
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
