#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy

c_Names=['label', 'cap-shape','cap-surface','cap-color','bruises','odor',
          'gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape',
          'stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
          'stalk-color-above-ring','stalk-color-below-ring','veil-type',
          'veil-color','ring-number','ring-type','spore-print-color','population',
          'habitat']
    
df = pandas.read_csv("agaricus-lepiota.data", header=None, na_values='?', names=c_Names) #To read the csv file using pandas
df.fillna(0) # convert all ? to 0's
# Converting the categorical data into binary
df = pandas.get_dummies(df)
df.reset_index(drop=True, inplace=True) # removing the first column which displays the line number
    
#Split the data into training(60), validation(20) and test(20) datasets
training, validate, testing = numpy.split(df, [int(.6*len(df)), int(.8*len(df))])
    
    
# Save the data to the respective files.
training.to_csv('training.txt', header=False, index = False)
print("Successfully Created training.txt")
validate.to_csv('validation.txt', header=False, index = False)
print("Successfully Created validation.txt")
testing.to_csv('testing.txt', header=False, index = False)
print("Successfully Created testing.txt")


# In[ ]:




