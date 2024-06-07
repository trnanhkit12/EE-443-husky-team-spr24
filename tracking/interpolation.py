import os
import os.path as osp
import sys
import time

import numpy as np
import ultralytics.trackers as trackers

# interpolation missing frames using kalman filter 
def kalman_interpolation(tracklets):
    # interpolate missing detection of a given track
    kf = trackers.utils.kalman_filter.KalmanFilterXYWH()
    for i, t in enumerate(tracklets):
        mean = None
        covariance = None
        bout = []
        tout = []
        for i, box in enumerate(t.boxes):
            # initial measurment to be passed to kalman filter
            measurment = [box[0], box[1], box[2], box[3]]
            if (i == 0):
                mean, covariance = kf.initiate(measurment)
            elif (t.times[i] == t.times[i-1]+1):
                # update and predict the mean and covariance when there is no gap
                mean, covariance = kf.predict(mean, covariance)
                mean, covariance = kf.update(mean, covariance, measurment)
            else:
                # calculate the number of missing frames
                gapcount = t.times[i] - t.times[i-1] - 1
                print(f'start: box = {t.boxes[i-1]}, time = {t.times[i-1]}')
                for j in range(gapcount):
                    # create new bounding box for a missing frame
                    mean, covariance = kf.predict(mean, covariance)
                    gbox = [mean[0], mean[1], mean[2], mean[3]]
                    gtime = t.times[i-1]+j+1
                    if (gapcount < 50):
                        bout.append(gbox)
                        tout.append(gtime)
                    print(f'added: box = {gbox}, time = {gtime}')
                print(f'end: box = {t.boxes[i]}, time = {t.times[i]}')

                mean, covariance = kf.predict(mean, covariance)
                mean, covariance = kf.update(mean, covariance, measurment)
                        
            bout.append(box)
            tout.append(t.times[i])

        t.boxes = bout
        t.times = tout

    return tracklets
        

# interpolate the missing frames using a linear framework
def linear_interpolation(tracklets):
    # interpolate missing detection of a given track
    for i, t in enumerate(tracklets):
        bout = []
        tout = []
        for i, box in enumerate(t.boxes):
            if (i == 0):
                pass
            else:
                # calculate the number of missing frames
                gapcount = t.times[i] - t.times[i-1] - 1
                if (gapcount < 50):
                    deltaT = t.times[i] - t.times[i-1]
                    gbox = t.boxes[i-1]
                    gtime = t.times[i-1]
                    incBox = (t.boxes[i] - t.boxes[i-1])/deltaT
                    for j in range(gapcount):
                        # create bounding box for each missing frame
                        gbox = gbox + incBox
                        gtime = gtime + 1
                        bout.append(gbox)
                        tout.append(gtime)

            bout.append(box)
            tout.append(t.times[i])

        t.boxes = bout
        t.times = tout

    return tracklets
