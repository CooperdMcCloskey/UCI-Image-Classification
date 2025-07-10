import math

import matplotlib.pyplot as plt
import tensorflow as tf

from data import data_import

data_import.createKey('data/2022_11_09_BonitaCanyon1')

ds = data_import.createDataset(['data/2022_11_09_BonitaCanyon1'])