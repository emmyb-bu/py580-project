import pandas as pd
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import glob
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#Importing data into Pandas defs
transverse_files = glob.glob(r'C:\Users\19545\Downloads\PankajFinal\TransverseCurrent\*', recursive=True)
transverse_dfs = [pd.read_csv(f, header=None, sep="\t", index_col=False) for f in transverse_files]

hall_files = glob.glob(r'C:\Users\19545\Downloads\PankajFinal\HallVoltage\*', recursive=True)
hall_dfs = [pd.read_csv(f, header=None, sep="\t", index_col=False) for f in hall_files]

master_transverse = pd.concat(transverse_dfs,ignore_index=True)
master_hall = pd.concat(hall_dfs,ignore_index=True)

#Importing labels into Pandas dfs
transverse_labels = glob.glob(r'C:\Users\19545\Downloads\PankajFinal\TransverseLabels\*', recursive = True)
transverse_dfs_labels = [pd.read_csv(f, header=None, sep="\t", index_col=False) for f in transverse_labels]

hall_labels = glob.glob(r'C:\Users\19545\Downloads\PankajFinal\HallLabels\*', recursive = True)
hall_dfs_labels = [pd.read_csv(f, header=None, sep="\t", index_col=False) for f in hall_labels]

master_tlabels = pd.concat(transverse_dfs_labels,ignore_index=True)
master_hlabels = pd.concat(hall_dfs_labels,ignore_index=True)

#Adjusting Indices to reflect concat
count = 0
for idx in master_tlabels.iloc[:, 0]:
    true = (idx - 1)+math.floor(count/100)*995
    master_tlabels.iloc[count, 0] = true
    count = count + 1
count = 0
for idx in master_hlabels.iloc[:, 0]:
    true = (idx - 1)+math.floor(count/100)*995
    master_hlabels.iloc[count, 0] = true
    count = count + 1

#Selecting for Labeled Data
idx_tlabels = master_tlabels.iloc[:,0].values
idx_hlabels = master_hlabels.iloc[:,0].values

labeled_transverse = master_transverse.drop(index=master_transverse.index.difference(idx_tlabels))
labeled_hall = master_hall.drop(index=master_hall.index.difference(idx_hlabels))

#Keras Time
x_train, x_rem, y_train, y_rem = train_test_split(labeled_transverse, master_tlabels.iloc[:,1], train_size=0.7, shuffle = True)
x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.5, shuffle = True)

model = Sequential()
model.add(Dense(600, activation='relu', input_shape=(600,)))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(11, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=math.floor(len(x_train)/2), verbose = 1)
