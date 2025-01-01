#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[41]:


import warnings
warnings.filterwarnings("ignore")


# In[42]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[43]:


f = open("5year.arff", 'r')


# In[44]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[45]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[46]:


y[0]


# In[47]:


answers = {} # Your answers


# In[48]:


def accuracy(predictions, y):
    corrects = sum([(predi == yi) for (predi, yi) in zip(predictions, y)])
    return corrects/len(y)


# In[49]:


def BER(predictions, y):
    TP = sum([predi and yi for (predi, yi) in zip(predictions, y)])
    TN = sum([(predi==False and yi==False) for (predi, yi) in zip(predictions, y)])
    FP = sum([(predi==True and yi==False) for (predi, yi) in zip(predictions, y)])
    FN = sum([(predi==False and yi==True) for (predi, yi) in zip(predictions, y)])
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    return 1-0.5*(TPR+TNR)


# In[50]:


### Question 1


# In[51]:


mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)


# In[52]:


acc1 = accuracy(pred, y)
ber1 = BER(pred, y)


# In[53]:


answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate


# In[54]:


answers['Q1']


# In[55]:


assertFloatList(answers['Q1'], 2)


# In[56]:


### Question 2


# In[57]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)


# In[58]:


acc2 = accuracy(pred, y)
ber2 = BER(pred, y)


# In[59]:


answers['Q2'] = [acc2, ber2]


# In[60]:


answers['Q2']


# In[61]:


assertFloatList(answers['Q2'], 2)


# In[62]:


### Question 3


# In[63]:


random.seed(3)
random.shuffle(dataset)


# In[64]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[65]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[66]:


len(Xtrain), len(Xvalid), len(Xtest)


# In[67]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

pred = mod.predict(Xtrain)
berTrain = BER(pred, ytrain)
pred = mod.predict(Xvalid)
berValid = BER(pred, yvalid)
pred = mod.predict(Xtest)
berTest = BER(pred, ytest)


# In[68]:


answers['Q3'] = [berTrain, berValid, berTest]


# In[69]:


answers['Q3']


# In[70]:


assertFloatList(answers['Q3'], 3)


# In[71]:


### Question 4


# In[74]:


Ctest = 0.0001
berList = []
ber5 = None
bestC = None

while Ctest <= 10000:
    model = linear_model.LogisticRegression(C=Ctest, class_weight='balanced')
    model.fit(Xtrain, ytrain)
    predictValid = model.predict(Xvalid)
    berValid = BER(predictValid, yvalid)
    #print("C= " + str(Ctest) + ", BER= ", str(berValid))
    if ber5 == None or berValid < ber5:
        ber5 = berValid
        bestC = Ctest
    berList += [berValid]
    Ctest = Ctest*10
    


# In[75]:


answers['Q4'] = berList


# In[76]:


answers['Q4']


# In[77]:


assertFloatList(answers['Q4'], 9)


# In[78]:


### Question 5


# In[79]:


model = linear_model.LogisticRegression(C=bestC, class_weight='balanced')
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
ber5 = BER(pred, ytest)


# In[80]:


answers['Q5'] = [bestC, ber5]


# In[81]:


answers['Q5']


# In[82]:


assertFloatList(answers['Q5'], 2)


# In[83]:


### Question 6


# In[84]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[85]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[86]:


dataTrain[0]


# In[123]:


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    usersPerItem[d['book_id']].add(d['user_id'])
    itemsPerUser[d['user_id']].add(d['book_id'])
    reviewsPerUser[d['user_id']].append(d)
    reviewsPerItem[d['book_id']].append(d)
    ratingDict[(d['user_id'], d['book_id'])] = d['rating']


# In[ ]:





# In[124]:


def Jaccard(s1, s2):
    #assuming 2 lists of same type?
    num = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return num/denom


# In[125]:


def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[ ]:





# In[126]:


answers['Q6'] = mostSimilar('2767052', 10)


# In[127]:


answers['Q6']


# In[128]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)


# In[129]:


### Question 7


# In[130]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)
    
ratingMean = sum([d['rating'] for d in dataset]) / len(dataset) #incase user hasn't rated anything


# In[131]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[136]:


def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        #print(str(d))
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2]) #R-Raverage
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2])) #Sim(i,j)
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[137]:


preds = [predictRating(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
mse7 = MSE(preds, labels)


# In[138]:


answers['Q7'] = mse7


# In[139]:


answers['Q7']


# In[140]:


assertFloat(answers['Q7'])


# In[141]:


### Question 8


# In[142]:


def predictRatingU(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerItem[item]:
        #print(str(d))
        u2 = d['user_id']
        if u2 == user: continue
        ratings.append(d['rating'] - userAverages[u2]) #R-Raverage
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[u2])) #Sim(u,v)
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[143]:


preds = [predictRatingU(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
mse8 = MSE(preds, labels)


# In[144]:


answers['Q8'] = mse8


# In[145]:


answers['Q8']


# In[146]:


assertFloat(answers['Q8'])


# In[ ]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




