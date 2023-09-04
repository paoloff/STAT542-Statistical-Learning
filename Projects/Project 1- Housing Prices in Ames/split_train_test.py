import pandas as pd
import random

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
