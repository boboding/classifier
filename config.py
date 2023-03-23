# -*- coding: utf-8 -*-
'''
author:pumpkin king
'''

import os
import time

class Config:

    # 训练参数
    phase="train" # or predict
    load_model=False

    train_data_rate = 0.8      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.2     # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 256
    learning_rate = 0.001
    epoch = 1000               # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 50                # 训练多少epoch，验证集没提升就停掉
    random_seed = 1            # 随机种子，保证可复现


    # 框架参数
    model_name="model.pth"

    # 网络配置
    input_dim  = 4    # 输入维度 有多少个特征列 本例中表格数据有4列表示特征，最后一列表示类别
    output_dim = 3    # 输出维度 一共分了多少个类别


    # 路径参数


    train_data_path = "./data/iris.csv"


    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False         
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if phase=="train" and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + "/"
        os.makedirs(log_save_path)

    model=3 #模型类型