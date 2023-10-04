"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
# student: Petter Byström
import numpy as np



def manhattan_dist(v1, v2):
    return float(sum([abs(v1[i]-v2[i]) for i in range(len(v1))]))

def hamming_dist(v1, v2):
    return float(np.count_nonzero(v1!=v2))
    
def euclidean_dist(v1, v2):
    return float(np.sqrt(np.sum(np.square(v1 - v2))))
  

def chebyshev_dist(v1, v2):
    return float(max([abs(v1[i]-v2[i]) for i in range(len(v1))]))

def minkowski_dist(v1, v2, d):
    return float(np.sum( abs(v1-v2)**d) ** (1.0/d))
