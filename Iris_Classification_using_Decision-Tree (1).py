#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np #linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


df = pd.read_csv("C:\\Users\\Anuj Kesharwani\\Downloads\\Iris.csv") #importing dataset 
df


# In[67]:


df.info() #information about our dataset


# In[68]:


df.describe() # description_of_dataset 


# In[69]:


df.shape # shape of dataset


# In[70]:


df.isnull().sum() #Checking is there any null value in our dataset


# In[71]:


df.Species.value_counts() # Total_count_of_our_target_variable


# In[72]:


df.drop('Id',axis=1,inplace=True) #dropping the Id column as it is unecessary


# In[73]:


df.head() #checking the dataset after dropping Id Cloumn


# In[94]:


fig = df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# ### The above graph shows relationship between the sepal length and width. Now we will check relationship between the petal length and width.

# In[95]:


fig = df[df.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
df[df.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
df[df.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# ### As we can see that the Petal Features are giving a better cluster division compared to the Sepal features. This is an indication that the Petals can help in better and accurate Predictions over the Sepal. We will check that later.

# #### Now let us see how are the length and width are distributed

# In[96]:


df.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[77]:


#Declare feature vector and target variable
X = df.drop(['Species'], axis=1)

y = df['Species']


# In[78]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[79]:


X_train.head(3)


# In[80]:


X_test.head(3)


# In[81]:


y_train.head(3)# get top 3 values


# In[82]:


y_test.head(3)


# In[83]:


from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm


# In[84]:


from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics


# In[85]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,y_test))


# In[86]:


# instantiate the DecisionTreeClassifier model with criterion entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)


# In[87]:


y_pred_en = clf_en.predict(X_test)


# In[88]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))


# In[89]:


y_pred_train_en = clf_en.predict(X_train)


# In[97]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


# In[98]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))


# ## Visualize decision-trees

# In[99]:


plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_en.fit(X_train, y_train)) 


# ## Visualize decision-trees with graphviz

# In[93]:


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
 
dot_data = export_graphviz(clf_en, filled=True, rounded=True,
                                    class_names=['Iris-setosa','Iris-virginica','Iris-versicolor'],
                                    feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],
                                    out_file=None)
graph = graphviz.Source(dot_data) 
graph


# In[ ]:




