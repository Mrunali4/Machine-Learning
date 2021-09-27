#!/usr/bin/env python
# coding: utf-8

# In[44]:


# for data Manipulation 
import numpy as np
import pandas as pd 

# for data visualiazations
import matplotlib.pyplot as plt
import seaborn as sns 

# for data interactivity 
from ipywidgets import interact 

# read data set 
data=pd.read_csv('data.csv')


# In[46]:


#check shape of data set 
print(data.shape)
data.head()


# In[47]:


data.isnull().sum()


# In[48]:


data['label'].value_counts()


# In[49]:


print("Averange Ratio of Nitrogen in SOil:{0:.2f}".format(data['N'].mean()))
print("Averange Ratio of Phosphosurs in SOil:{0:.2f}".format(data['P'].mean()))
print("Averange Ratio of Potassium in SOil:{0:.2f}".format(data['K'].mean()))
print("Averange Ratio of temperature in SOil:{0:.2f}".format(data['temperature'].mean()))
print("Averange Ratio of Humidity in SOil in %:{0:.2f}".format(data['humidity'].mean()))
print("Averange Ratio of ph in SOil:{0:.2f}".format(data['ph'].mean()))
print("Averange Ratio of rainfall in SOil in mm:{0:.2f}".format(data['rainfall'].mean()))


# In[50]:


# checks the summary of each of the crops. 
@interact
def summary(crops=list(data['label'].value_counts().index)):
    x=data[data['label']==crops]
    print('=============================================')
    print('Statics for Nitrogen')
    print('Min N required :',x['N'].min())
    print('Avrage N required :',x['N'].mean())
    print('Max N required :',x['N'].max())
    
    print('=============================================')
    print('Statics for Phosphosurs')
    print('Min P required :',x['P'].min())
    print('Avrage P required :',x['P'].mean())
    print('Max P required :',x['P'].max())
    
    print('=============================================')
    print('Statics for Potassium')
    print('Min K required :',x['K'].min())
    print('Avrage K required :',x['K'].mean())
    print('Max K required :',x['K'].max())
    
    print('=============================================')
    print('Statics for Temperature')
    print('Min temperature required :{0:.2f}'.format(x['temperature'].min()))
    print('Avrage temperature required :{0:.2f}'.format(x['temperature'].mean()))
    print('Max temperature required :{0:.2f}'.format(x['temperature'].max()))
    
    print('=============================================')
    print('Statics for Humidity')
    print('Min K required :{0:.2f}'.format(x['humidity'].min()))
    print('Avrage K required :{0:.2f}'.format(x['humidity'].mean()))
    print('Max K required :{0:.2f}'.format(x['humidity'].max()))
    
    print('=============================================')
    print('Statics for ph')
    print('Min ph required :{0:.2f}'.format(x['ph'].min()))
    print('Avrage ph required :{0:.2f}'.format(x['ph'].mean()))
    print('Max ph required :{0:.2f}'.format(x['ph'].max()))
    
    print('=============================================')
    print('Statics for Rainfall')
    print('Min rainfall required :{0:.2f}'.format(x['rainfall'].min()))
    print('Avrage rainfall required :{0:.2f}'.format(x['rainfall'].mean()))
    print('Max rainfall required :{0:.2f}'.format(x['rainfall'].max()))
    
    
    
    
    
    
    
    
    


# In[51]:


# compare averange requirement of each crops with average condyions 
@interact 
def comapre(conditions=['N','P','K','ph','temperature','rainfall']):
    print("Crops which require greaater than average",conditions,'\n')
    print(data[data[conditions]>data[conditions].mean()]['label'].unique())
    print("------------------------------------------------------")
    print("crops which requies less than average",conditions,'\n')
    print(data[data[conditions]<=data[conditions].mean()]['label'].unique())

            
    
    


# In[52]:


plt.subplot(2,4,3)
sns.distplot(data['K'],color='darkblue')
plt.xlabel('Ratio of Potassium',fontsize=12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['N'],color='darkblue')
plt.xlabel('Ratio of Nitrogen',fontsize=12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(data['temperature'],color='darkblue')
plt.xlabel('Ratio of temperature',fontsize=12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['rainfall'],color='darkblue')
plt.xlabel('Ratio of rainfall',fontsize=12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['humidity'],color='darkblue')
plt.xlabel('Ratio of humidity',fontsize=12)
plt.grid()



plt.show()



# In[53]:


from sklearn.cluster import KMeans 
#removing the label column 
x=data.drop(['label'],axis=1)
#selecting all values of data 
x=x.values 

print(x.shape)


# In[54]:


# determine number cluster in data set 

plt.rcParams['figure.figsize']=(10,4)

wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
#lets plot the results 
plt.plot(range(1,11),wcss)
plt.title("the Elbow Method",fontsize=20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()
    


# In[55]:


# lets implement the k means agorithms to perform clustring anlaysis 
km=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=km.fit_predict(x)
   
#lets find out the results 
a=data['label']
y_means=pd.DataFrame(y_means)
z=pd.concat([y_means,a],axis=1)
z=z.rename(columns={0:'cluster'})

# lets check the result of each crops 
print("let's check the results after applying the Kmeans Clustring Analysis \n")
print('crops in First Cluster:',z[z['cluster']==0]['label'].unique())
print('---------------------------------------------------------------')

print('crops in Second Cluster:',z[z['cluster']==1]['label'].unique())
print('---------------------------------------------------------------')

print('crops in Third Cluster:',z[z['cluster']==2]['label'].unique())
print('---------------------------------------------------------------')

print('crops in Fourth Cluster:',z[z['cluster']==3]['label'].unique())
print('---------------------------------------------------------------')


# In[56]:


# lets split data set in 2 column 

y=data['label']
x=data.drop(['label'],axis=1)

print('shape of x:',x.shape)
print('shape of  y:',y.shape)


# In[57]:


# lets split Training and Testing sets for Validation of Results 
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train ,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("shape of X train:",x_train.shape)
print("shape of X test:",x_test.shape)
print("shape of y train:",y_train.shape)
print("shape of y train:",y_test.shape)



# In[58]:


# lets create prdictive model 
from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
# print(y_pred)


# In[59]:


# Evaluate model performs 
from sklearn.metrics import confusion_matrix

# print the confusion matrix first
plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='Wistia')
plt.title('Confusion Matrix for Logistic Regression',fontsize=15)
plt.show()


# In[60]:


# Classification report
cr=classification_report(y_test,y_pred)
print(cr)


# In[61]:


# lets check the head of DS
data.head()


# In[65]:


Prediction=model.predict((np.array([[190,
                                     40,
                                     40,
                                     20,
                                     80,
                                     7,
                                     200
                                    ]])))

print("Prediction for Given Climate Conditions is :" ,Prediction)


# In[ ]:




