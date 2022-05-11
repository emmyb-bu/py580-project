import pandas as pd
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Maximum
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

#Creating new column to match output of ML
count = 0
temparray = []
newcolumn = []
for idx in master_tlabels.iloc[:, 1]:
    temparray = []
    if idx == '-':
        temparray = np.zeros(11)
        temparray[10] = 1
        newcolumn.append(temparray)
    else:
        temparray = np.zeros(11)
        temparray[int(idx)] = 1
        newcolumn.append(temparray)
newcolumn = np.asarray(newcolumn).astype('float32')
tlabels_asarray = newcolumn

count = 0
temparray = []
newcolumn = []
for idx in master_hlabels.iloc[:, 1]:
    temparray = []
    if idx == '-':
        temparray = np.zeros(11)
        temparray[10] = 1
        newcolumn.append(temparray)
    else:
        temparray = np.zeros(11)
        temparray[int(idx)] = 1
        newcolumn.append(temparray)
newcolumn = np.asarray(newcolumn).astype('float32')
hlabels_asarray = newcolumn

#Selecting for Labeled Data
idx_tlabels = master_tlabels.iloc[:,0].values
idx_hlabels = master_hlabels.iloc[:,0].values

labeled_transverse = master_transverse.drop(index=master_transverse.index.difference(idx_tlabels))
labeled_hall = master_hall.drop(index=master_hall.index.difference(idx_hlabels))

#Keras Time
x_train, x_test, y_train, y_test = train_test_split(labeled_transverse, tlabels_asarray, train_size=0.7, shuffle = True)
#x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.5, shuffle = True)

model = Sequential()
model.add(Dense(600, activation='relu', input_shape=(600,), name = 'x'))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(11, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
history = model.fit(x_train, y_train,validation_split = .5, epochs=50, batch_size=math.floor(len(x_train)/2), verbose = 1)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
