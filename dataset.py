# -*- coding: utf-8 -*-
'''
author:pumpkin king
'''
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from config import Config
import torch
import random

class Dataset:
    def __init__(self, config):
        self.config = config
        self.x, self.y = self.read_data()
    

    def read_data(self):                # 读取初始数据
 
        init_data = pd.read_csv(self.config.train_data_path,header=0)

        init_data['species'] = init_data['species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

        print(init_data.head)
        print("init_data.columns.tolist():",init_data.columns.tolist())

        x = init_data.drop(["species"],axis=1).values
        y = init_data["species"].values
        # print("x:",x)
        # print("y:",y)
        print("type(x):",type(x))


        return x,y


    def custom_train_test_split(self,x, y, test_size, is_shuffle):
 
        train_size = int((1 - test_size) * len(x))

        data_index=list(range(0,len(x)))    

        if(is_shuffle):
             random.shuffle(data_index)

        x_train = x[data_index[:train_size]]
        y_train = y[data_index[:train_size]]
    
        x_test = x[data_index[train_size:]]
        y_test = y[data_index[train_size:]]

        return x_train, x_test, y_train, y_test

    def get_train_and_valid_data(self):
        
        train_x, valid_x, train_y, valid_y = train_test_split(self.x, self.y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=True)   # 划分训练和验证集
        
        print(type(train_x))
    
        train_x = torch.FloatTensor(train_x)
        valid_x = torch.FloatTensor(valid_x)
        train_y = torch.LongTensor(train_y)
        valid_y = torch.LongTensor(valid_y)

  

        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):


        # 这里拿了训练集进行了测试，根据需要换成自己的测试集，或者读取其他csv文件
        train_x, valid_x, train_y, valid_y = train_test_split(self.x, self.y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=False)   # 划分训练和验证集
        
        train_x = torch.FloatTensor(train_x)
        valid_x = torch.FloatTensor(valid_x)
        train_y = torch.LongTensor(train_y)
        valid_y = torch.LongTensor(valid_y)

        if return_label_data:
            return valid_x,valid_y

        return  valid_x



    

np.random.seed(Config.random_seed)  # 设置随机种子，保证可复现
data_g = Dataset(Config)