import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd
from os.path import join as join

# import the dataset
movies = pd.read_csv(join('ml-1m', 'movies.dat'), sep='::', header=None, encoding='latin-1')
users = pd.read_csv(join('ml-1m', 'users.dat'), sep='::', header=None, encoding='latin-1')
ratings = pd.read_csv(join('ml-1m', 'ratings.dat'), sep='::', header=None, encoding='latin-1')

# preparing the training and test sets
training_set = pd.read_csv(join('ml-100k', 'u1.base'), delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv(join('ml-100k', 'u1.test'), delimiter='\t')
test_set = np.array(test_set, dtype='int')

user_count = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
movie_count = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# convert data into an array with users = rows and movies = columns
def format(data):
    new_data = []
    for user_id in range(1, user_count + 1):
        # get all ratings by this user
        movie_id = data[:, 1][data[:, 0] == user_id]
        rating_id = data[:, 2][data[:, 0] == user_id]
        ratings = np.zeros(movie_count)
        ratings[movie_id - 1] = rating_id
        new_data.append(list(ratings))
        return new_data


training_set = format(training_set)
test_set = format(test_set)
print(training_set)

# convert data to torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# convert ratings into binary liked/didn't like
training_set[training_set == 0] = -1  # didn't rate
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set > 2] = 1
test_set[test_set == 0] = -1  # didn't rate
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set > 2] = 1

print(training_set)
print(test_set)

# architect the NN





















