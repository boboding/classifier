# -*- coding: UTF-8 -*-

import torch.nn.functional as F
import torch.nn as nn

class Net_Classifier(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Net_Classifier,self).__init__()
        self.input_layer    = nn.Linear(input_dim,128)
        self.hidden_layer  = nn.Linear(128,64)
        self.output_layer   = nn.Linear(64,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer(out))
        out =  self.output_layer(out)

        return out
    



