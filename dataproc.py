
# coding: utf-8

# In[1]:


import glob
import os

QA = 20
train = {}
for i in range(QA):  
    if (i+1) not in train:
        train[i+1] = {}
    
    search_test = './tasks_1-20_v1-2/en/qa'+str(i+1)+'_*_test.txt'
    search_train = './tasks_1-20_v1-2/en/qa'+str(i+1)+'_*_train.txt'
    for filename in glob.glob(search_train):
        print(filename)    
        with open(filename) as f:
            Flag = False
            counter = 0
            for line in f:
                lines = line.split('\t')
                if(len(lines)==1):
                    if(Flag==True):
                        counter+=1
                        Flag = False
                    if counter in train[i+1]:
                        if 'memory' in train[i+1][counter]:
                            train[i+1][counter]['memory'].append(line)
                        else:
                            train[i+1][counter]['memory'] = [line]
                    else:
                        train[i+1][counter] = {}
                        train[i+1][counter]['memory'] = [line]
                else:
                    train[i+1][counter]['query'] = lines[0]
                    train[i+1][counter]['answer'] = lines[1]
                    train[i+1][counter]['fact'] = lines[2]
                    Flag = True
                    


# In[2]:


import glob
import os

QA = 20
test = {}
for i in range(QA):  
    if (i+1) not in test:
        test[i+1] = {}
    
    search_test = './tasks_1-20_v1-2/en/qa'+str(i+1)+'_*_test.txt'
    search_train = './tasks_1-20_v1-2/en/qa'+str(i+1)+'_*_train.txt'
    for filename in glob.glob(search_test):
        print(filename)    
        with open(filename) as f:
            Flag = False
            counter = 0
            for line in f:
                lines = line.split('\t')
                if(len(lines)==1):
                    if(Flag==True):
                        counter+=1
                        Flag = False
                    if counter in test[i+1]:
                        if 'memory' in test[i+1][counter]:
                            test[i+1][counter]['memory'].append(line)
                        else:
                            test[i+1][counter]['memory'] = [line]
                    else:
                        test[i+1][counter] = {}
                        test[i+1][counter]['memory'] = [line]
                else:
                    test[i+1][counter]['query'] = lines[0]
                    test[i+1][counter]['answer'] = lines[1]
                    test[i+1][counter]['fact'] = lines[2]
                    Flag = True
                    


# In[4]:


import pickle

with open('train.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

