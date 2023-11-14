import pandas as pd
import numpy as np

# Using train/test split from split_1
train_test_df = pd.read_csv('project3_splits.csv')
test_ind = np.array(list(train_test_df['split_1']))-1
train_ind = np.array(list(set(list(range(0,50000)))-set(test_ind)))

# Splitting data and saving
df = pd.read_table("alldata.tsv")
sentiment_list = list(df['sentiment'])
testdf = df.iloc[test_ind]
traindf = df.iloc[train_ind]
testdf.to_csv('test.tsv', sep="\t")
traindf.to_csv('train.tsv', sep="\t")