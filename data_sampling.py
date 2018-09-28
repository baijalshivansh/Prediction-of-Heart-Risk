#Heart Risk
#Data extraction and Sampling

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
 
 
 # importing the dataset and random under sampling and random oversampling
dataset = pd.read_csv('train_2v.csv')
dataset=dataset.dropna()
dataset=dataset.drop(dataset.query('stroke == 0').sample(frac=.4).index)
d1=dataset.loc[dataset['stroke'] == 1]
d1 = d1.loc[np.repeat(d1.index.values,5)]
d1 = d1.reset_index(drop=True)
frames=[dataset, d1]
result=pd.concat(frames)
result = result.reset_index(drop=True)
df = result
from random import shuffle
df = df.sample(frac=1).reset_index(drop=True)

x = df.iloc[:, 1:11]
y = df.iloc[:, 11]