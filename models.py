#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy
import random

from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err

curr_node_id = 0

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    def eval(self):
        #evaluation part
        for i in range(self.num_nodes):
            self.layer_net[i] = numpy.dot(self.input_vals, numpy.transpose(self.weight[i])) + self.bias
            self.layer_out[i] = sig(self.layer_net[i])
            

    def backprop(self, other):
        #use backpropagation method to update weights
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                if self.layer_num == 1:
                    self.weight[i][j] = self.weight[i][j] - (LEARNING_RATE * other.weight_delta[0][i] * self.input_vals[j] * other.weight[0][i] * inv_sig(self.layer_out[i]))
                elif self.layer_num == 2:
                    self.weight_delta[i][j] = inv_sig(self.layer_out[i]) * inv_err(self.layer_out[i], other)
                    self.weight[i][j] = self.weight[i][j] - (LEARNING_RATE * self.weight_delta[i][j] * self.input_vals[j])
                

class cfile():
    def __init__(self, name, mode = 'r'):
        #self = file.__init__(self, name, mode)
        self.fh = open(name, mode)

    def w(self, string):
        self.fh.write(str(string) + '\n')
        return None

    def close(self):
        self.fh.close()


# In[ ]:





# In[ ]:





# In[ ]:




