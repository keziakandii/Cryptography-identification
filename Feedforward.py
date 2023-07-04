import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv("3des_runs_1.csv") #Read all data of the file in CSV format
val_data = pd.read_csv('3des_runs_1.csv')
data1 = np.mat(data) #Data matrixization
data2 = np.mat(val_data)
pre_raw = 30 #The number of rows where the predicted value is required
# print (data1)
def load_data_train():
    data_train = data1[:,:] # Columns 4 to 47 in rows 1 to 26 of the data are taken as training data
    data_val = data2[:,:]
    #print(data_train)
    return data_train,data_val

def load_data_pre():
    data_pre = data1[pre_raw, :].astype('float64') #Columns 4 to 46 of row 31 are input data
    data_mean = data_pre.mean() #average
    data_std = data_pre.std() #Find the standard deviation
    data_pre = (data_pre - data_mean) / data_std #standardization
    return data_pre

def load_data_real():
    data_real = data1[pre_raw, ]  #Take out column 31 of the data in row 47
    return data_real

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation



def Train_Model(data_train,data_val):
    #modelfile = './modelweight' #This location holds the weights during model training
    #y_mean_std = "./y_mean_std.txt" # Save the data in the standardization process, which is required for later data restoration
    data_train = np.matrix(data_train).astype('float64')
    data_val = np.matrix(data_val).astype('float64')
    data_mean = np.mean(data_train, axis=0)#Average the columns
    data_std = np.std(data_train, axis=0)#Calculates the standard deviation for each column
    # data_train = (data_train - data_mean) / data_std
    print(1)
    x_train = data_train[:, 0:(data_train.shape[1] - 1)] #All data (except the last column) as input x
    y_train = data_train[:, data_train.shape[1] - 1] #The last column of all data is used as output y
    x_val = data_val[:, 0:(data_val.shape[1]-1)]
    y_val = data_val[:, data_val.shape[1]-1]
    #print(x_train)
    #print(y_train)
    #Model training
    model = Sequential()
    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer="uniform"))
    model.add(Activation('relu'))
    model.add(Activation('relu'))
    model.add(Dense(1, input_dim=x_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    history=model.fit(x_train, y_train, epochs=1000, batch_size=x_train.shape[0],validation_data=(x_val,y_val))
    #model.save_weights(modelfile) #Save the model weights
    y_mean = data_mean[:, data_train.shape[1] - 1]
    y_std = data_std[:, data_train.shape[1] - 1]
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs,acc,'b',label='Training accuracy')
    plt.plot(epochs,val_acc,'r',label='validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.legend()
    plt.show()
    print("Training is complete")

Train_Model(data,val_data)

#BP neural networks
