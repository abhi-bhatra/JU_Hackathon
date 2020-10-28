#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams


# In[3]:


## TO CHANGE THE DISPLAY TEXT SIZE
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["NonFraud", "Fraud"]


# In[4]:


data = pd.read_csv('c:/Users/DEREK/Desktop/Hackathon1/creditcard.csv')
data.head()


# In[5]:


data.info()


# In[6]:


##To check if their is any null value in the given dataset

data.isnull().values.any()


# In[7]:


count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[8]:


#Differentiate the fraud and normal dataset
fraud = data[data['Class'] == 1]
nonfraud = data[data['Class'] == 0]


# In[9]:


len(fraud)


# In[10]:


len(nonfraud)


# In[11]:


fraud.Amount.describe()


# In[12]:


nonfraud.Amount.describe()


# In[13]:


#COMPARISON BETWEEN THE FRAUD AND NON-FRAUD AMOUNT TRANSACTION IN DOLLARS
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(nonfraud.Amount, bins = bins)
ax2.set_title('NonFraud')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[14]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(nonfraud.Time, nonfraud.Amount)
ax2.set_title('NonFraud')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[15]:


data1= data.sample(frac = 0.1,random_state=1)
data1.shape


# In[16]:


data.shape


# In[17]:


fraud = data1[data1['Class'] == 1]


# In[18]:


nonfraud = data1[data1['Class'] == 0]


# In[19]:


outlier_fraction = len(fraud)/len(nonfraud)


# In[20]:


print(outlier_fraction)


# In[21]:


print("Fraud Cases: {}".format(len(fraud)))


# In[22]:


print("Non-Fraud Cases: {}".format(len(nonfraud)))


# In[23]:


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:





# In[24]:


#Create independent and Dependent Features
columns = data1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[29]:


'''##Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1, random_state=state)
   
}
'''


# In[30]:


classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}


# In[31]:


type(classifiers)


# In[ ]:


n_outliers = len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))


# In[ ]:




