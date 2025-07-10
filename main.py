
import os
import surprise_mars_cyclegan as search
import pickle
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
import random 
import numpy as np


GaGan = search.GaGan()
GaGan.begin_server("D:/MarsEnv2/WindowsNoEditor/Mars.exe", 'PhysXCar')



for i in range(1, 11):    
    GaGan.searchAlgo(100, 12, i)
# GaGan.searchAlgo(100, 12, 1, kde)