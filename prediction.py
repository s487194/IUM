import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F

class ANN_Model(nn.Module):
    def __init__(self,input_features=82,hidden1=20,hidden2=20,out_features=3):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self, x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x

data = pd.read_csv("Sales.csv")

data["Profit_Category"] = pd.cut(data["Profit"], bins=[-np.inf, 500, 1000, np.inf], labels=[0, 1, 2])
bike = data.loc[:, ['Customer_Age', 'Customer_Gender', 'Country','State', 'Product_Category', 'Sub_Category', 'Profit_Category']] 
bikes = pd.get_dummies(bike, columns=['Country', 'State', 'Product_Category', 'Sub_Category', 'Customer_Gender'])
X = bikes.drop('Profit_Category', axis=1).values
y = bikes['Profit_Category'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

model = torch.load("classificationn_model.pt")
model.eval()  

with torch.no_grad():
    y_pred = model(X_test)

_, predicted = torch.max(y_pred.data, 1)

np.savetxt("predictions1.txt", predicted.numpy(), fmt='%d')
