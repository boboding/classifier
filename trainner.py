# -*- coding: utf-8 -*-
'''
author:pumpkin king
'''
import torch

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
def train(Net,config, logger, train_and_valid_data):

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data

    print(type(train_X))
    print(type(train_Y))
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size,shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size,shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = Net(config.input_dim,config.output_dim).to(device)
    if config.load_model:                # 加载原模型参数
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0

    train_loss_hist = []

    test_loss_hist = []


    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train() #训练模式
        train_loss_array = []
  
        for i, _data in enumerate(train_loader):

            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()             

            pred_Y = model(_train_X)   

            # print("pred_Y.shape:",pred_Y.shape)
            # print("_train_Y.shape:",_train_Y.shape)
        
            loss = criterion(pred_Y, _train_Y)  # 计算loss
            loss.backward()

           
            optimizer.step()            
            train_loss_array.append(loss.item())
            global_step += 1


        model.eval() #预测模式
        valid_loss_array = []

        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = model(_valid_X)


            loss = criterion(pred_Y, _valid_Y)
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)

        train_loss_hist.append(train_loss_cur)
        test_loss_hist.append(valid_loss_cur)

        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))


        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:        # 早停机制
                logger.info(" The training stops early in epoch {}".format(epoch))
                break

    plt.ylim(0, 1)
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")

    plt.legend()
    plt.show()        