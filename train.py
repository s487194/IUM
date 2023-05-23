import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


# Wczytanie danych
print(os.getcwd())
data = pd.read_csv("./Sales.csv")

# Przygotowanie danych
data["Profit_Category"] = pd.cut(data["Profit"], bins=[-np.inf, 500, 1000, np.inf], labels=[0, 1, 2])
bike = data.loc[:, ['Customer_Age', 'Customer_Gender', 'Country','State', 'Product_Category', 'Sub_Category', 'Profit_Category']]  
bikes = pd.get_dummies(bike, columns=['Country', 'State', 'Product_Category', 'Sub_Category', 'Customer_Gender'])
X = bikes.drop('Profit_Category', axis=1).values
y = bikes['Profit_Category'].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
#### Tworzenie tensorów
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

#### Model

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


torch.manual_seed(20)
model=ANN_Model()
model.parameters

def calculate_accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        accuracy = correct / total * 100
    return accuracy

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
final_losses = []
accuracy_list = []

for i in range(epochs):
    i = i + 1
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss)
    
    train_accuracy = calculate_accuracy(model, X_train, y_train)
    test_accuracy = calculate_accuracy(model, X_test, y_test)
    print(f"Epoch: {i}, Loss: {loss.item()}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model,"classificationn_model.pt")