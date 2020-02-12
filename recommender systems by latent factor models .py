#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
k = 20
lamb = 0.1
iters = 40
learning = 0.01

file = "c:/cs246/ratings.train.txt"
myfile = open(file, 'r')

q = {}
p = {}

#initialize all q and p
for line in myfile:
    items = line.strip().split("\t")
    q_row = int(items[0])
    p_row = int(items[1])
    if q_row in q:
        pass
    else:
        q[q_row] = np.random.rand(k) * np.sqrt(5.0/float(k))
    if p_row in p:
        pass
    else:
        p[p_row] = np.random.rand(k) * np.sqrt(5.0/float(k))


# In[7]:


# start to train the data
error_record = []
for i in range(iters):
    data = open(file, 'r')
    for line in data:
        rat = line.strip().split("\t")
        q_idx = int(rat[0])
        p_idx = int(rat[1])
        rate = int(rat[2])

        qi = q[q_idx]
        pu = p[p_idx]
    
        A = 2.0 * (rate - np.dot(qi, pu.T))

        qi_update = qi + learning * (A * pu - 2.0 * lamb * qi)
        pu_update = pu + learning * (A * qi - 2.0 * lamb * pu)
        q[q_idx] = qi_update
        p[p_idx] = pu_update
        
    # calculate error
    error = 0.0
    data = open(data, 'r')
    for line in data:
        rat = line.strip().split("\t")
        q_idx = int(rat[0])
        p_idx = int(rat[1])
        rate = int(rat[2])

        qi = q[q_idx]
        pu = p[p_idx]
        pu_T = pu.reshape(k, 1)
        error += (rate - np.dot(qi, pu_T)) ** 2
    for q_key in q:
        error += np.sum(q[q_key] * q[q_key])
    for p_key in p:
        error += np.sum(p[p_key] * p[p_key])
    # record error in each iter
    error_scalar = error.reshape(())
    error_record.append(error_scalar)


# In[8]:


x = np.arange(0, iters, 1) + 1
y = error_record
plt.plot(x, y, "-o")
plt.xlabel("# of Iteration")
plt.ylabel("Error")
plt.show()


# In[ ]:




