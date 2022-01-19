import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from utils.metrics import plot_pr_curve, plot_mc_curve, compute_ap

def road_ap_per_class(recalllist, prelist, lenth):
    # unique_classes = np.array(list(prelist.keys()))
    unique_classes = []
    for k in recalllist:
        if recalllist[k]:
            unique_classes.append(k)
    unique_classes = np.array(unique_classes)
    nc = len(unique_classes)  # number of classes, number of detections
    ap, p, r = np.zeros((nc, lenth)), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        # AP from recall-precision curve
        p[ci][:len(prelist[c])] = np.array(prelist[c])
        r[ci][:len(recalllist[c])] = np.array(recalllist[c])
        ap[ci], mpre, mrec = compute_ap(recalllist[c], recalllist[c])

    f1 = 2 * p * r / (p + r + 1e-16)
    i = f1.mean(0).argmax()  # max f1 index
    if i == 0:
        i = r.mean(0).argmax()  # max r index
    return p[:, i], r[:, i], ap, unique_classes.astype('int32')
