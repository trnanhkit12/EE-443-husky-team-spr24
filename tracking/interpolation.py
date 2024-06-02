import os
import os.path as osp
import sys
import time

import numpy as np
import ultralytics.trackers as trackers

def kalman_interpolation(detection, embedding):
    # interpolate missing detection of a given track
    personSet = np.unique(detection[:, 1])
    dout = None
    eout = None
    kf = trackers.utils.kalman_filter.KalmanFilterXYWH()
    for person in personSet:
        mean = None
        covariance = None
        idx = detection[:, 1]==person
        dset = detection[idx]
        eset = embedding[idx]
        last_frame = None
        print(f'person = {person}')
        for i, d in enumerate(dset):
            measurment = d[[3, 4, 5, 6]]
            if (last_frame is None):
                mean, covariance = kf.initiate(measurment)
                last_frame = d
            elif ((last_frame[2]+1)==d[2]):
                last_frame = d
                mean, covariance = kf.update(mean, covariance, measurment)
            else:
                gapcount = d[2]-last_frame[2] - 1
                if (gapcount > 20):
                    last_frame = None
                else:
                    for j in range (gapcount.astype(int)):
                        mean, covariance = kf.predict(mean, covariance)
                        interp = last_frame
                        interp[2] = interp[2] + 1
                        interp[3] = mean[0]
                        interp[4] = mean[1]
                        interp[5] = mean[2]
                        interp[6] = mean[3]
                        dout = np.vstack([dout, interp])
                        eout = np.vstack([eout, eset[i,:]])
                    last_frame = d
                    
            if dout is None:
                dout = d
                eout = eset[0, :]
            else:
                dout = np.vstack([dout, d])
                eout = np.vstack([eout, eset[i,:]])
                
    return (dout, eout)

def linear_interpolation(detection, embedding):
    # interpolate missing detection of a given track
    personSet = np.unique(detection[:, 1])
    dout = None
    eout = None
    for person in personSet:
        idx = detection[:, 1]==person
        dset = detection[idx]
        eset = embedding[idx]
        last_frame = None
        print(person)
        for i, d in enumerate(dset):
            if (last_frame is None) or ((last_frame[2]+1)==d[2]):
                last_frame = d
            else:
                interp = last_frame
                deltaT = d[2] -last_frame[2]
                incX = (d[3]-last_frame[3])/deltaT
                incY = (d[4]-last_frame[4])/deltaT
                incW = (d[5]-last_frame[5])/deltaT
                incH = (d[6]-last_frame[6])/deltaT
                gapcount = d[2]-last_frame[2]-1
                if (gapcount > 20):
                    last_frame = None
                else:
                    for j in range (gapcount.astype(int)):
                        interp[2] = interp[2] + 1
                        interp[3] = interp[3] + incX
                        interp[4] = interp[4] + incY
                        interp[5] = interp[5] + incW
                        interp[6] = interp[6] + incH
                        dout = np.vstack([dout, interp])
                        eout = np.vstack([eout, eset[i, :]])
                    last_frame = d
            if dout is None:
                dout = d
                eout = eset[0, :]
            else:
                dout = np.vstack([dout, d])
                eout = np.vstack([eout, eset[i, :]])
                
    return (dout, eout)
