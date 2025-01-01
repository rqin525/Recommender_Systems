#!/usr/bin/env python
# coding: utf-8

# In[152]:


get_ipython().system('pip install scikit-surprise')


# In[34]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
#from sklearn import svm
import numpy
import string
import random
import string
#from sklearn import linear_model
import warnings
from surprise import SVDpp, Reader, Dataset
from surprise.model_selection import train_test_split
warnings.filterwarnings("ignore")


# In[62]:


get_ipython().system('pip install implicit')


# In[84]:


get_ipython().system('pip install tensorflow')


# In[63]:


import pandas as pd
from implicit import bpr


# In[7]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[8]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[9]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[10]:


# Some data structures that will be useful


# In[11]:


allRatings = []
userIDs = {}
bookIDs = {}
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    if not l[0] in userIDs: userIDs[l[0]] = len(userIDs)
    if not l[1] in bookIDs: bookIDs[l[1]] = len(bookIDs)


# In[12]:


len(allRatings)


# In[13]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
usersPerBook = defaultdict(set) # Maps an item to the users who rated it
booksPerUser = defaultdict(set) # Maps a user to the items that they rated
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    usersPerBook[b].add(u)
    booksPerUser[u].add(b)


# In[14]:


ratingsValid[0]


# In[15]:


##################################################
# Read prediction                                #
##################################################


# In[16]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[17]:


### Question 1


# In[18]:


unseenPerUser = defaultdict(list)
#uniqueUser = []
for u in userIDs: #for every user   
    for b in bookIDs: #find a book
            if u not in booksPerUser or b not in booksPerUser[u]: #the user hasn't seen
                unseenPerUser[u].append(b)


# In[19]:


ratingsValidNeg = []
numpy.random.seed(0)
for u,b,r in ratingsValid:
    ratingsValidNeg.append((u,b,1)) #has read
    unseenList = unseenPerUser[u]
    bookNum = numpy.random.randint(len(unseenList))
    ratingsValidNeg.append((u, unseenList[bookNum], 0)) #has not read
random.shuffle(ratingsValidNeg)


# In[20]:


predictions = []
ytest = [v[2] for v in ratingsValidNeg]
for l in ratingsValidNeg:
    if l[1] in return1:
        predictions.append(1)
    else:
        predictions.append(0)


# In[21]:


### Question 2


# In[108]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
threshold = totalRead/4*3
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > threshold: break


# In[23]:


### Question 3/4


# In[24]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[25]:


########################################## predict read #########################################


# In[69]:


Xui = scipy.sparse.lil_matrix((len(userIDs), len(bookIDs)))
for d in allRatings:
    Xui[userIDs[d[0]],bookIDs[d[1]]] = 1
    
Xui_csr = scipy.sparse.csr_matrix(Xui)


# In[70]:


model = bpr.BayesianPersonalizedRanking(factors = 5)
model.fit(Xui_csr)


# In[73]:


recommended = model.recommend(0, Xui_csr[0])


# In[86]:


recommended


# In[101]:


# predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     maxSim=0
# #     if u in userIDs and b in bookIDs and len(ratingsPerUser[u])>1:
# #         ids, score = model.recommend(userIDs[u], Xui_csr[userIDs[u]], N=len(ratingsPerUser[u])/2)
# #         for i,s in zip(ids, score):
# #             if i==bookIDs[b]:
# #                 if s>0.5:
# #                     predictions.write(u + ',' + b + ",1\n")
# #                 else:
# #                     predictions.write(u + ',' + b + ",0\n")
# #                 continue
    
    
#     users = set(ratingsPerItem[b])
#     for b2,_ in ratingsPerUser[u]:
#         sim = Jaccard(users, set(ratingsPerItem[b2]))
#         if sim>maxSim:
#             maxSim = sim
#     if (maxSim > 0.013) or b in return1 or len(ratingsPerItem[b])>40:
#         predictions.write(u + ',' + b + ",1\n")
#     else:
#         predictions.write(u + ',' + b + ",0\n")

# predictions.close()


# In[109]:


####################################### final implementation ###########################################

scorePerUser = defaultdict(list)
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    maxSim=0
#     if u in userIDs and b in bookIDs and len(ratingsPerUser[u])>1:
#         ids, score = model.recommend(userIDs[u], Xui_csr[userIDs[u]], N=len(ratingsPerUser[u])/2)
#         for i,s in zip(ids, score):
#             if i==bookIDs[b]:
#                 if s>0.5:
#                     predictions.write(u + ',' + b + ",1\n")
#                 else:
#                     predictions.write(u + ',' + b + ",0\n")
#                 continue
    
    
    users = set(ratingsPerItem[b])
    for b2,_ in ratingsPerUser[u]:
        sim = Jaccard(users, set(ratingsPerItem[b2]))
        if sim>maxSim:
            maxSim = sim
    scorePerUser[u].append((maxSim, b))
    
for u in scorePerUser:
    scorePerUser[u].sort(key=lambda tup: tup[0], reverse=True)  
    count = 0
    for scores,b in scorePerUser[u]:
        if (count < int(len(scorePerUser[u])/2)) and (scores>0.013) or b in return1 or len(ratingsPerItem[b])>40:
            predictions.write(u + ',' + b + ",1\n")
        else: 
            predictions.write(u + ',' + b + ",0\n")
        count += 1
        
# for l in open("pairs_Read.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     maxSim=0
#     median = scorePerUser[u][int(len(scorePerUser[u])/2)]
#     users = set(ratingsPerItem[b])
#     for b2,_ in ratingsPerUser[u]:
#         sim = Jaccard(users, set(ratingsPerItem[b2]))
#         if sim>maxSim:
#             maxSim = sim
    
#     if (maxSim >= median):
#         predictions.write(u + ',' + b + ",1\n")
#     else:
#         predictions.write(u + ',' + b + ",0\n")

predictions.close()


# In[155]:





# In[156]:


##################################################
# Rating prediction                              #
##################################################


# In[157]:


### Question 6


# In[36]:


#use ratingsValid and ratingsTrain
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
data = Dataset.load_from_file("train_Interactions.csv", reader=reader)


# In[37]:


trainset, testset = train_test_split(data, test_size=1/20)


# In[38]:


### Question 7


# In[39]:


minBeta = model.bu[0]
maxBeta = model.bu[0]
userIndex = 0
minUser = trainset.to_raw_uid(0)
maxUser = trainset.to_raw_uid(0)

for u in trainset.all_users():
    b = model.bu[userIndex]
    if b<minBeta:
        minBeta = b
        minUser = trainset.to_raw_uid(u)
    if b>maxBeta:
        maxBeta = b
        maxUser = trainset.to_raw_uid(u)
    userIndex += 1


# In[40]:


### Question 8


# In[57]:


mses = []
lambdas = [0.0075, 0.008, 0.0085]
for lr in lambdas:
    model = SVDpp(reg_all=0.11, n_factors=3, lr_all=0.0075)
    model.fit(trainset)
    predictions = model.test(testset)
    sse = 0
    for p in predictions:
        sse += (p.r_ui - p.est)**2
    mses.append(sse / len(predictions))

validMSE = mses[0]
lamb = lambdas[0]
for (m,l) in zip(mses, lambdas):
    if m<validMSE:
        validMSE = m
        lamb = l


# In[58]:


(lamb, validMSE)


# In[179]:


answers['Q8']


# In[205]:


assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])


# In[59]:


model = SVDpp(reg_all=0.11, n_factors=3, lr_all=0.0075)
model.fit(trainset)
Rui = trainset.global_mean


# In[60]:


########################################## predict rating #########################################


# In[61]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    pred = model.predict(u, b, r_ui=Rui)
    predictions.write(u + ',' + b + ',' + str(pred.est) + '\n')
predictions.close()


# In[ ]:





# In[ ]:





# In[ ]:




