# -*- coding: UTF-8 -*-

'''
author:pumpkin king
'''

#import pandas as pd
import numpy as np
#import os

# import time
# import logging
# from logging.handlers import RotatingFileHandler
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

from trainner import train
from evaluator import predict,predict_custom
from utils import load_logger

from dataset import Dataset
from config import Config



from model.net_classifier import Net_Classifier
import torch


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 可复现
        data_original = Dataset(config)
        Net = Net_Classifier

        if config.model==1:
            Net = Net_Classifier
        elif config.model == 2:
            Net=Net_Classifier
        elif config.model == 3:
            Net=Net_Classifier   

        if config.phase=="train":
            print("The soothsayer will train")
            train_X, valid_X, train_Y, valid_Y = data_original.get_train_and_valid_data()
            train(Net,config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.phase=="predict":
            print("The soothsayer will predict")
            test_X, test_Y = data_original.get_test_data(return_label_data=True)
            print(test_X.shape)

            pred_result = predict(Net,config, test_X)       
            print(pred_result)

            print("test_Y:",test_Y)
            from sklearn.metrics import precision_score
            ps = precision_score(test_Y, pred_result, average='micro') 
            print(ps)
      
        if config.phase=="predict_custom":
            print("The soothsayer will predict custom data")

            test_X=torch.FloatTensor([5.1,	3.5,	1.4,	0.2]) #0
            #test_X=torch.FloatTensor([5.9,	3,	5.1,	1.8]) #2

            pred_result = predict_custom(Net,config, test_X)     
      
            print(pred_result)


       
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse


    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", default="predict", type=str, help="train or predict")
    parser.add_argument("-m", "--model", default=1, type=int, help="1:Net_Classifier 2:Net_Classifier")
    args = parser.parse_args()


    c = Config()
    for key in dir(args):
        if not key.startswith("__"):
            setattr(c, key, getattr(args, key))   # 将属性值赋给Config


    main(c)
    #python main.py -p "train" -m 1
    #python main.py -p "predict" -m 1

