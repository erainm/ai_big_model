#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     ：ai_big_model
@File        ：mobile_price_category_predict.py
@Create at   ：2025/8/11 15:19
@version     ：V1.0
@Author      ：erainm
@Description : 手机价格分类预测（基础版）
'''
# 导包
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. define function,perform data load and progressing
def get_data_loader(batch_size,data_url):
    # 1. load data
    data = pd.read_csv(data_url)
    # 2. know data
    print("data shape-->",data.shape)
    print("data column-->",data.columns)
    print("previous data-->",data.head())
    print("data info-->",data.info())

    # exit()
    # 3. processing data
    # 3.1 gain feature value x(1~20 column) and target value y(21 column)
    x, y = data.iloc[:, :-1],data.iloc[:,-1]
    # 3.2 type convert(later used for tensor),feature value convert float,target value convert integer
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 3.3 dataset divide
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=88)
    # 3.4 construct dataset,finally trainSet(dataloader) and testSet(dataloader)
    # first,convert numpy dataset to tensor,then,encapsulate into tensor dataset
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.from_numpy(y_valid.values))
    # second,convert tensor dataset encapsulate into data loader,and set batch size and whether to disturb the data
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size = batch_size, shuffle=False)
    # 4. return result: dataloader,dataset length,input feature dim,output feature dim(0,1,2,3)
    # unique() explain: de duplication,get only 0,1,2,3
    return train_dataloader,valid_dataloader, x_train.shape[1], len(y.unique())

# 2. define class,construct neural network
class PhonePriceModel(torch.nn.Module):
    # overwrite init method and forward method
    def __init__(self,input_num, output_num):
        super.__init__()
        # define net structure,neural network hide layer
        self.linear1 = torch.nn.Linear(20, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.out = torch.nn.Linear(256,4)

    # overwrite forward method(neuron inside,weighted summation and activation function)
    # default hide layer use relu activation function
    def forward(self, x):
        # weighted summation --> activation function(hide layer default use relu function)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(2))

        # output layer
        x = self.out(x)
        # return weighted summation result,not predict result
        return x

# 3. train model
def train_model(train_dataloader, phone_price_model, epochs):
    # 1. get data(parameter have data)
    # 2. get model(parameter have model)
    # 3. create multi-class loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # 4. create optimizer
    optimizer = torch.optim.SGD(phone_price_model.parameters(), lr=0.01)
    # 5. loop train model,outer loop turns, inner loop batch
    for epoch in range(epochs):
        # define init parameters
        total_loss, batch_cnt, start = 0.00, 0, time.time()
        for batch_x, batch_y in train_dataloader:
            # forward propagation:from input to output:predict value and loss value
            # model predict(bottom used forward,used weighted summation)
            y = phone_price_model(batch_x)
            # loss calculation
            loss = loss_fn(y, batch_y)
            # summation loss value and batch value
            total_loss += loss.item()
            batch_cnt += 1
            # Backpropagation,from output to input:grad calculation and parameter update
            # grad clear
            optimizer.zero_grad()
            # grad calculation
            loss.backward() # auto differential
            optimizer.step() # optimizer update parameter:w = w_old - lr * grad

        # this step,one round end,summation loss and batch
        epoch_loss = total_loss / batch_cnt
        print(f"round {epoch +1},runtime:{time.time() - start:.2f} seconds,loss value is --> {epoch_loss:.2f}")
    # 6. save trained model parameter dictionary
    torch.save(phone_price_model.state_dict(),"C:/Users/erain/Desktop/ai_big_model/mobile_price_category_predict/data/mobile_price_class_predict.pth")

# 4. model evaluate
def eval_model(valid_dataloader, input_num, output_num):
    # 1. get data((parameter have data))
    # 2. create new model object,load trained model parameter dictionary,for evaluate
    phone_price_model = PhonePriceModel(input_num, output_num)
    phone_price_model.load_state_dict(torch.load("C:/Users/erain/Desktop/ai_big_model/mobile_price_category_predict/data/mobile_price_class_predict.pth"))
    # 3. define variable,record true sample num
    correct = 0
    # 4. batch evaluate process
    for batch_x, batch_y in valid_dataloader:
        # 4.1 model predict
        y = phone_price_model(batch_x)
        # 4.2 argmax() max get predict result
        y_predict = torch.argmax(y, dim=1)
        # get predict ture num
        correct += (y_predict == batch_y).sum()

    # 5. calculation predict accuracy
    print(f"predict accuracy: {(correct / len(valid_dataloader.dataset)):.4f}")

# program main entrance
if __name__ == '__main__':
    # define function,load data and progressing
    # set batch_size，and pass to get_data_loader() function
    batch_size = 8
    train_dataloader, valid_dataloader, input_num, output_num = get_data_loader(batch_size=batch_size,data_url="C:/Users/erain/Desktop/ai_big_model/mobile_price_category_predict/data/手机价格预测.csv")

    # define class,construct neural network
    phone_price_model = PhonePriceModel(input_num, output_num)

    # train model(forward/reverse spread)
    epochs = 50
    train_model(train_dataloader, phone_price_model, epochs)

    # model evaluate
    eval_model(valid_dataloader, input_num, output_num)