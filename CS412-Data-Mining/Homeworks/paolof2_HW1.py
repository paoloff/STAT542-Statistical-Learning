import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("automobile.csv")
sub_df = df[['curb-weight', 'horsepower','city-mpg', 'highway-mpg','price']]
pca = PCA(2)
pca.fit(sub_df)
print(pca.components_)