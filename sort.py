# sort.py

from filterpy.kalman import KalmanFilter
import numpy as np
from numba import jit

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array(list(zip(x, y)))
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

@jit
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox.reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(bbox)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        return self.kf.x

    def get_state(self):
        return self.kf.x[:4].reshape((1, 4))

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, bboxes):
        for tracker in self.trackers:
            tracker.predict()

        matched, unmatched_detections, unmatched_trackers = self.associate_detections_to_trackers(bboxes)

        for m in matched:
            self.trackers[m[1]].update(bboxes[m[0]])

        for i in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(bboxes[i]))

        # Remove lost trackers
        self.trackers = [self.trackers[i] for i in range(len(self.trackers)) if i not in unmatched_trackers]

        return np.array([self.trackers[i].get_state().flatten() for i in range(len(self.trackers))])

    def associate_detections_to_trackers(self, detections):
        if len(self.trackers) == 0:
            return [], [], list(range(len(detections)))

        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, detection in enumerate(detections):
            for t, tracker in enumerate(self.trackers):
                iou_matrix[d, t] = iou(detection, tracker.predict())

        matched_indices = linear_assignment(-iou_matrix)

        unmatched_detections = []
        unmatched_trackers = []

        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        for t in range(len(self.trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m)

        return matches, unmatched_detections, unmatched_trackers
