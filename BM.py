import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd
from os.path import join as join

movies = pd.read_csv(join('ml-1m', 'movies.dat'), sep='::', header=None, encoding='latin-1')
users = pd.read_csv(join('ml-1m', 'users.dat'), sep='::', header=None, encoding='latin-1')
ratings = pd.read_csv(join('ml-1m', 'ratings.dat'), sep='::', header=None, encoding='latin-1')

print(users)
