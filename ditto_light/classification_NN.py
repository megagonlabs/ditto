# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:05:20 2021

@author: lydia
"""
import torch
import torch.nn as nn
   
class classification_NN(nn.Module):
    def __init__(self,
                 inputs_dimension,
                 num_hidden_lyr=2,
                 dropout_prob=0.5,
                 bn=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        
        output_dim = 1 # 0 for unmatch; 1 for match
        #layer_channels = list(range(inputs_dimension,output_dim,-1*int((inputs_dimension-output_dim)/(1+num_hidden_lyr))))
        layer_channels = [inputs_dimension] * (1+num_hidden_lyr)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(layer_channels[i], layer_channels[i + 1])
                                    for i in range(num_hidden_lyr)])))
        
        self.activation = nn.ReLU()
        
        self.layer_out =  nn.Linear(layer_channels[-1], output_dim)
        self.weight_init(self.layer_out)
        
        if bn:
            self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim) for dim in layer_channels[1:]])
        
    def weight_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
        return m
    
    def forward(self, data):
        """ forward propagate input
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        output = data
        for i, layer in enumerate(self.layers):
            output = self.activation(self.bn[i](layer(output)))
        #return torch.softmax(self.layer_out(self.dropout(output)), dim =1)
        #return torch.softmax(self.layer_out((output)), dim =1)
        return self.layer_out(output)
