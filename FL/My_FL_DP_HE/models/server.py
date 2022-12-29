import os
import argparse, json
import time
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import pysnooper
from functools import reduce
from joblib import Parallel, delayed


tf.compat.v1.enable_eager_execution()

print(tf.__version__)
print(tf.executing_eagerly())
print(torch.cuda.is_available())





