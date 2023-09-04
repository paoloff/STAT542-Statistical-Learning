import pandas as pd
import random

# Randomly split the data in file AMES.csv into two csv files:
#    - train, containing 75% of the data
#    - test, containing 25% of the data
#
# Both train.csv and test.csv files are saved in the current folder

def split(filename, seed_n = 0):

    random.seed(seed_n)

    dataset = pd.read_csv(filename)
    N = len(dataset)
    indices = list(range(N))
    train_indices = random.sample(indices, k=int(0.75*N))
    test_indices = list(set(indices)-set(train_indices))
    train = dataset.iloc[train_indices]
    test = dataset.iloc[test_indices]

    train.to_csv("./train.csv")
    test.to_csv("./test.csv")
    
    return 
