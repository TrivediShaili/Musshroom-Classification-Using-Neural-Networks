#!/usr/bin/env python
# coding: utf-8

# In[5]:


import math


def sig(x):
    #use logistic function as activation function
    return (float(1) / (float(1 + math.exp(-x)))) # formula of the Sigmoid/Logistic Activation

def inv_sig(x):
    #derivative of the output of neruon with respect to its input
    return (1 - sig(x)) * sig(x) # Derivative of the Logistic Function
    
def err(o, t):
    #squared error function, o is the actual output value and t is the target output 
    return 0.5 * ((t - o) ** 2)   #Formula for the Mean Square Error Method

def inv_err(o, t):
    #derivative of squared error function with respect to o
    return (o - t) # Derivation of the Mean Square Error Function


# In[ ]:




