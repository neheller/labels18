import numpy as np
import copy
import math
import cv2
import multiprocessing
from scipy.interpolate import interp1d

import time
from datetime import timedelta

floor = -128
ceiling = 128

"""Helpful changes to the input data"""
def preprocess(X):
    X[X<floor] = floor
    X[X>ceiling] = ceiling
    return X

"""Unhelpful changes to the ground truth labels"""
lis_frequency = 0.0716433
psd_frequency = 0.0044771

variances = {
    "lis": {
        "schop": 5,
        "mchop": 10.5,
        "lchop": 15.5,
        "snat": 14,
        "mnat": 52,
        "lnat": 138,
        "srnd": 0.05,
        "mrnd": 0.10,
        "lrnd": 0.15
    },
    "psd": {
        "schop": 1.2,
        "mchop": 2.7,
        "lchop": 4,
        "snat": 0.8,
        "mnat": 4.8,
        "lnat": 11.8,
        "srnd": 0.05,
        "mrnd": 0.10,
        "lrnd": 0.15
    }
}

#Utility class to help adjust points in polar space
class Vector2():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    def __mul__(self, other):
        return Vector2(self.x * other, self.y * other)

    def magnitude(self):
        return (self.x**2 + self.y**2) ** 0.5

    def unitV(self):
        if self.magnitude() == 0:
            return Vector2(0,0)
        return Vector2(self.x / self.magnitude(), self.y / self.magnitude())

    def dotP(self, other):
        return (self.x * other.x) + (self.y * other.y)

    @staticmethod
    def randomVector():
        return Vector2(random.uniform(-1, 1), random.uniform(-1, 1))

    def __str__(self):
        return "x = " + str(self.x) + " y = " + str(self.y)

#produces a smooth looping function from 0 to 2pi by samping 5 points and interpolating
def smooth_function(center, variance):
    x = np.linspace(0, 2*math.pi, num = 6, endpoint = True)
    y = np.random.normal(center, variance**0.5, 5)
    y = np.append(y, y[0])
    return interp1d(x, y, kind='cubic')

#perturbs a uint8 binary image
def perturb_nat(image, variance):
    center = 0
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #these are only needed for RBG images
    # ret, thresh = cv2.threshold(gray, 20, 255, 0)
    contour_image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    output = np.zeros(np.shape(contour_image))
    if len(contours) == 0: #blank image
        return output

    for contour in contours:
        f = smooth_function(center, variance)
        size = len(contour)

        br = cv2.boundingRect(contour)
        cx = br[0] + br[2] / 2
        cy = br[1] + br[3] / 2
        center_point = Vector2(cx, cy)

        for count, point in enumerate(contour):
            x = (count / size) * 2 * math.pi
            distance_to_move = f(x)
            current_point = Vector2(point[0][0], point[0][1])
            direction = current_point - center_point
            direction = direction.unitV()
            offset = direction * distance_to_move
            contour[count][0] = [current_point.x - offset.x, current_point.y - offset.y]

        cv2.fillPoly(output, pts=[contour], color=1)
    return output


# Assumes no more than 5 chunks in a row
def perturb_row(rbool, tot, rnd):
    if tot < 0.5:
        return rbool
    rng = np.arange(0,512)
    begin0 = 0
    while(not rbool[begin0]):
        begin0 = begin0 + 1
    end0 = begin0 + 1
    while(rbool[end0]):
        end0 = end0 + 1
    new_begin0 = begin0 + rnd[0]
    new_end0 = end0 + rnd[1]
    # length is end - begin
    if (end0 - begin0 - tot >= 0):
        return np.logical_and(rng>=new_begin0, rng<new_end0)
    # looking for a second chunk
    begin1 = end0 + 1
    while(not rbool[begin1]):
        begin1 = begin1 + 1
    end1 = begin1 + 1
    while(rbool[end1]):
        end1 = end1 + 1
    new_begin1 = begin1 + rnd[2]
    new_end1 = end1 + rnd[3]
    if (end0 - begin0 + end1 - begin1 - tot >= 0):
        return np.logical_or(
            np.logical_and(rng>=new_begin0, rng<new_end0),
            np.logical_and(rng>=new_begin1, rng<new_end1)
        )
    # looking for a third chunk
    begin2 = end1 + 1
    while(not rbool[begin2]):
        begin2 = begin2 + 1
    end2 = begin2 + 1
    while(rbool[end2]):
        end2 = end2 + 1
    new_begin2 = begin2 + rnd[4]
    new_end2 = end2 + rnd[5]
    if (end0 - begin0 + end1 - begin1 + end2 - begin2 - tot >= 0):
        return np.logical_or(
            np.logical_or(
                np.logical_and(rng>=new_begin0, rng<new_end0),
                np.logical_and(rng>=new_begin1, rng<new_end1)
            ),
            np.logical_and(rng>=new_begin2, rng<new_end2)
        )
    # looking for a fourth chunk
    begin3 = end2 + 1
    while(not rbool[begin3]):
        begin3 = begin3 + 1
    end3 = begin3 + 1
    while(rbool[end3]):
        end3 = end3 + 1
    new_begin3 = begin3 + rnd[6]
    new_end3 = end3 + rnd[7]
    if (end0 - begin0 + end1 - begin1 + end2 - begin2 + end3 - begin3 - tot >= 0):
        return np.logical_or(
            np.logical_or(
                np.logical_and(rng>=new_begin0, rng<new_end0),
                np.logical_and(rng>=new_begin1, rng<new_end1)
            ),
            np.logical_or(
                np.logical_and(rng>=new_begin2, rng<new_end2),
                np.logical_and(rng>=new_begin3, rng<new_end3),
            )
        )
    # looking for a fifth chunk
    begin4 = end3 + 1
    while(not rbool[begin4]):
        begin4 = begin4 + 1
    end4 = begin4 + 1
    while(rbool[end4]):
        end4 = end4 + 1
    new_begin4 = begin4 + rnd[7]
    new_end4 = end4 + rnd[8]
    return np.logical_or(
        np.logical_or(
            np.logical_or(
                np.logical_and(rng>=new_begin0, rng<new_end0),
                np.logical_and(rng>=new_begin1, rng<new_end1)
            ),
            np.logical_or(
                np.logical_and(rng>=new_begin2, rng<new_end2),
                np.logical_and(rng>=new_begin3, rng<new_end3),
            )
        ),
        np.logical_and(rng>=new_begin4, rng<new_end4)
    )

def natural_perturb(Y, variance):
    for i in range(0, 20):
        Y[i] = perturb_nat(Y[i], variance)
    return Y

def choppy_perturb(Y, variance):
    rnds = np.random.normal(0, variance, (20*512, 10))
    unrolled_bools = np.reshape(Y, (20*512, 512)) > 0.5
    tots = np.sum(unrolled_bools, axis=1)
    for i in range(50, 20*512-50):
        unrolled_bools[i,:] = perturb_row(
            unrolled_bools[i,:], tots[i], rnds[i,:]
        )
    perturbed_bools = np.reshape(unrolled_bools, (20, 512, 512))
    return perturbed_bools.astype(np.int32)

# I use pr_foo1bar to denote p(foo|bar)
def random_perturb(Y, p, f):
    pr_flip1rare = p
    pr_flip1comm = f/(1-f)*p
    rnds = np.random.random(Y.shape)
    pos2neg_flips = np.logical_and(np.less(rnds, pr_flip1rare), Y)
    neg2pos_flips = np.logical_and(np.less(rnds, pr_flip1comm), 1-Y)
    Y = Y - pos2neg_flips + neg2pos_flips
    return Y


def perturb(Y, mode="control", ds="lis", f=None):
    if mode == "control":
        return Y
    else:
        y = Y[:,:,:,0]
    variance = variances[ds][mode]
    if mode[1:] == "chop":
        y = choppy_perturb(y, variance)
    elif mode[1:] == "nat":
        y = natural_perturb(y.astype(np.uint8), variance)
    elif mode[1:] == "rnd":
        if f is None:
            if (ds == "lis"):
                f = lis_frequency
            else:
                f = psd_frequency
        y = random_perturb(y.astype(np.uint8), variance, f)

    ret = np.zeros(np.shape(Y))
    ret[:,:,:,0] = y
    ret[:,:,:,1] = 1-y
    return ret
