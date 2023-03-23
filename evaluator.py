# -*- coding: utf-8 -*-
'''
author:pumpkin king
'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def predict(Net,config, test_X):
    # 获取测试数据
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(config.input_dim,config.output_dim).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()

    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X = model(data_X)
        print(pred_X)
        pred_X = torch.max(pred_X, 1)[1] 
        result = torch.cat((result, pred_X))

    return result.detach().cpu().numpy()




def predict_custom(Net,config, test_X):
    # 获取测试数据
    # test_set = TensorDataset(test_X)
    # test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(config.input_dim,config.output_dim).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()

    #for _data in test_loader:
    data_X = test_X.to(device)
    pred_X = model(data_X)
    print(pred_X)
    pred_X = torch.max(pred_X,0)[1]
    print(pred_X)
    #result = torch.cat((result, pred_X))

    return pred_X
