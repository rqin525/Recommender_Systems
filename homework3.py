#!/usr/bin/env python
# coding: utf-8

# In[152]:


get_ipython().system('pip install scikit-surprise')


# In[153]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import warnings
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
warnings.filterwarnings("ignore")


# In[110]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[111]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[112]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[113]:


answers = {}


# In[114]:


# Some data structures that will be useful


# In[115]:


allRatings = []
userIDs = {}
bookIDs = {}
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    if not l[0] in userIDs: userIDs[l[0]] = len(userIDs)
    if not l[1] in bookIDs: bookIDs[l[1]] = len(bookIDs)


# In[116]:


len(allRatings)


# In[117]:


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


# In[118]:


ratingsValid[0]


# In[119]:


##################################################
# Read prediction                                #
##################################################


# In[120]:


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


# In[121]:


### Question 1


# In[122]:


unseenPerUser = defaultdict(list)
#uniqueUser = []
for u in userIDs: #for every user   
    for b in bookIDs: #find a book
            if u not in booksPerUser or b not in booksPerUser[u]: #the user hasn't seen
                unseenPerUser[u].append(b)


# In[123]:


ratingsValidNeg = []
numpy.random.seed(0)
for u,b,r in ratingsValid:
    ratingsValidNeg.append((u,b,1)) #has read
    unseenList = unseenPerUser[u]
    bookNum = numpy.random.randint(len(unseenList))
    ratingsValidNeg.append((u, unseenList[bookNum], 0)) #has not read
random.shuffle(ratingsValidNeg)


# In[124]:


predictions = []
ytest = [v[2] for v in ratingsValidNeg]
for l in ratingsValidNeg:
    if l[1] in return1:
        predictions.append(1)
    else:
        predictions.append(0)


# In[125]:


TP = sum([predi and yi for (predi, yi) in zip(predictions, ytest)])
TN = sum([(predi==False and yi==False) for (predi, yi) in zip(predictions, ytest)])
FP = sum([(predi==True and yi==False) for (predi, yi) in zip(predictions, ytest)])
FN = sum([(predi==False and yi==True) for (predi, yi) in zip(predictions, ytest)])
acc1 = (TP+TN)/(TP+TN+FP+FN)


# In[126]:


answers['Q1'] = acc1


# In[127]:


answers['Q1']


# In[128]:


assertFloat(answers['Q1'])


# In[129]:


### Question 2


# In[130]:


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


# In[131]:


predictions = []
ytest = [v[2] for v in ratingsValidNeg]
for l in ratingsValidNeg:
    if l[1] in return1:
        predictions.append(1)
    else:
        predictions.append(0)


# In[132]:


TP = sum([predi and yi for (predi, yi) in zip(predictions, ytest)])
TN = sum([(predi==False and yi==False) for (predi, yi) in zip(predictions, ytest)])
FP = sum([(predi==True and yi==False) for (predi, yi) in zip(predictions, ytest)])
FN = sum([(predi==False and yi==True) for (predi, yi) in zip(predictions, ytest)])
acc2 = (TP+TN)/(TP+TN+FP+FN)


# In[133]:


answers['Q2'] = [threshold, acc2]


# In[134]:


answers['Q2']


# In[135]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[136]:


### Question 3/4


# In[137]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[140]:


predictions = []
for u,b,_ in ratingsValidNeg:
    similarities = []
    for b2 in booksPerUser[u]:
        sim = Jaccard(usersPerBook[b], usersPerBook[b2])
        similarities.append(sim)
    similarities.sort(reverse=True)
    if (similarities and similarities[0] > 0.0025): #threshold
        predictions.append(1)
    else:
        predictions.append(0)


# In[141]:


TP = sum([predi and yi for (predi, yi) in zip(predictions, ytest)])
TN = sum([(predi==False and yi==False) for (predi, yi) in zip(predictions, ytest)])
FP = sum([(predi==True and yi==False) for (predi, yi) in zip(predictions, ytest)])
FN = sum([(predi==False and yi==True) for (predi, yi) in zip(predictions, ytest)])
acc3 = (TP+TN)/(TP+TN+FP+FN)


# In[143]:


predictions = []
for u,b,_ in ratingsValidNeg:
    similarities = []
    for b2 in booksPerUser[u]:
        sim = Jaccard(usersPerBook[b], usersPerBook[b2])
        similarities.append(sim)
    similarities.sort(reverse=True)
    if b in return1 or (similarities and similarities[0] > 0.0025): #threshold
        predictions.append(1)
    else:
        predictions.append(0)


# In[144]:


TP = sum([predi and yi for (predi, yi) in zip(predictions, ytest)])
TN = sum([(predi==False and yi==False) for (predi, yi) in zip(predictions, ytest)])
FP = sum([(predi==True and yi==False) for (predi, yi) in zip(predictions, ytest)])
FN = sum([(predi==False and yi==True) for (predi, yi) in zip(predictions, ytest)])
acc4 = (TP+TN)/(TP+TN+FP+FN)


# In[145]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[146]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[147]:


[acc3, acc4]


# In[148]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    similarities = []
    for b2 in booksPerUser[u]:
        sim = Jaccard(usersPerBook[b], usersPerBook[b2])
        similarities.append(sim)
    similarities.sort(reverse=True)
    if b in return1 or (similarities and similarities[0] > 0.0025): #threshold
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")

predictions.close()


# In[149]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[155]:


assert type(answers['Q5']) == str


# In[156]:


##################################################
# Rating prediction                              #
##################################################


# In[157]:


### Question 6


# In[164]:


#use ratingsValid and ratingsTrain
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
data = Dataset.load_from_file("train_Interactions.csv", reader=reader)


# In[165]:


trainset, testset = train_test_split(data, test_size=1/20)


# In[166]:


# Rui = sum([r for _,_,r in ratingsTrain]) / len(ratingsTrain)
# ytest = [v[2] for v in ratingsValid]
# init
# alpha = Rui
# betaU = random.uniform(0, 0.001, len(userIDs))
# betaI = random.uniform(0, 0.001, len(bookIDs))
# lamb = 1.0
# def predict(u,i):
#     return alpha + betaU[u] + betaI[i]
# def reg():
#     return lamb * (sum([u**2 for u in betaU]) + sum([i**2 for i in betaI]))


# In[181]:


model = SVD()
model.fit(trainset)
predictions = model.test(testset)


# In[182]:


sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

validMSE = sse / len(predictions)


# In[183]:


#compute alpha and betas to minimize sum of squared errors SSE


# In[184]:


answers['Q6'] = validMSE


# In[185]:


answers['Q6']


# In[186]:


assertFloat(answers['Q6'])


# In[173]:


### Question 7


# In[201]:


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


# In[202]:


answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]


# In[203]:


answers['Q7']


# In[ ]:


assert [type(x) for x in answers['Q7']] == [str, str, float, float]


# In[176]:


### Question 8


# In[177]:


mses = []
lambdas = [0.0001, 0.005, 0.001, 0.05, 0.01]
for reg in lambdas:
    model = SVD(reg_all=reg)
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


# In[178]:


answers['Q8'] = (lamb, validMSE)


# In[179]:


answers['Q8']


# In[205]:


assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])


# In[206]:


model = SVD(reg_all=0.05)
model.fit(trainset)
Rui = trainset.global_mean


# In[208]:


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


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:




