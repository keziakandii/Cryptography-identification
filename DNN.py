import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'feature_runs.csv')
#Convert the Species column from letters to numbers
#data['Species']=pd.factorize(data.Species)[0]
#X Don't want the zero column, don't want the last column First: represents all rows Second: represents columns
X = data.iloc[:,:-1].values
#Y only needs a specific column
Y = data.label.values
train_x,test_x,train_y,test_y=train_test_split(X,Y)
#Converting data to Tensor LongTensor is equivalent to int64
train_x = torch.from_numpy(train_x).type(torch.float32)
train_y = torch.from_numpy(train_y).type(torch.int64)
test_x = torch.from_numpy(test_x).type(torch.float32)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)

#The data is only 150 lines, so the batch is also a little smaller
batch = 5
no_of_batches = len(data)//batch
epochs = 200

#TensorDataset()Tensors can be packaged and merged
train_ds = TensorDataset(train_x,train_y)
#If you want the model to not pay attention to the order of the training set data, use the out-of-order order
train_dl = DataLoader(train_ds,batch_size=batch,shuffle=True)
test_ds = TensorDataset(test_x,test_y)
#There is no need to use out-of-order for the test set to avoid increased workload
test_dl = DataLoader(test_ds,batch_size=batch)

#Create a model
#Inheritance nn. Module this class and customize the model
class Model(nn.Module):
    #Define an initialization method
    def __init__(self):
        #Inherits all properties of the parent class
        super().__init__()
        #(Initialize the first layer input to hidden layer 1) 4 input features 32 output features
        self.liner_1 = nn.Linear(21,32)
        #(Initialize the second layer hidden layer 1 to hidden layer 2) 32 input features 32 output features
        self.liner_2 = nn.Linear(32,32)
        #(Initialize the third layer (hidden layer 2 to output layer) 32 input features 3 output features
        self.liner_3 = nn.Linear(32,16)
        self.liner_4 = nn.Linear(16,7)
        #Using F does not require initializing the activation layer

    #Defining def forward calls these layers to handle inputs
    def forward(self,input):
        #Make a call to input on the first layer and activate
        x = F.relu(self.liner_1(input))
        #Make a call to input on the second layer and activate
        x = F.relu(self.liner_2(x))
        #The multi-classification task is not activated, and some people write it as shown below, activated with softmax
        #x = F.softmax(self.liner_3(x))
        #First, nn. The input to the CrossEntropyLoss() function is an inactive output
        #Second, multinomial classifications can be activated using the softmax function
        #Then, using the softmax() function for activation is to map the result onto a probability distribution from 0 to 1
        #Finally, if you do not activate softmax, the maximum output value is still the value with the highest probability
        x = F.relu(self.liner_3(x))
        x = self.liner_4(x)
        return x

model = Model()
#Loss function
loss_fn = nn.CrossEntropyLoss()

def accuracy(y_pred,y_true):
    #Torch.argmax converts numbers into real predictions
    y_pred = torch.argmax(y_pred,dim=1)
    acc = (y_pred == y_true).float().mean()
    return acc

#It is easy to observe the change in values as the training progresses
train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]

def get_model():
    #Get this model
    model = Model()
    #Optimization function optimizes all variables of the model, i.e. model.parameters()
    opt = torch.optim.Adam(model.parameters(),lr=0.0001)
    return model,opt

model,optim = get_model()

for epoch in range(epochs):
    for x,y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # The gradient is set to 0
        optim.zero_grad()
        # Backpropagation solves gradients
        loss.backward()
        # optimize
        optim.step()
    # No gradient calculations are required
    with torch.no_grad():
        epoch_accuracy = accuracy(model(train_x),train_y)
        epoch_loss = loss_fn(model(train_x), train_y).data
        epoch_test_accuracy = accuracy(model(test_x),test_y)
        epoch_test_loss = loss_fn(model(test_x), test_y).data
        print('epoch: ',epoch,'train_loss: ',round(epoch_loss.item(),4),'train_accuracy: ',round(epoch_accuracy.item(),4),
             'test_loss: ',round(epoch_test_loss.item(),4),'test_accuracy: ',round(epoch_test_accuracy.item(),4)
              )
        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_accuracy)

plt.plot(range(1,epochs+1),train_loss,label='train_loss')
plt.plot(range(1,epochs+1),test_loss,label='test_loss')
plt.plot(range(1,epochs+1),train_acc,label='train_acc')
plt.plot(range(1,epochs+1),test_acc,label='test_acc')
plt.show()
#CNN
