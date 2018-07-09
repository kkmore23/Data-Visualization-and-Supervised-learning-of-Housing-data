
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
training = pd.read_csv('D:/Masters Academic/Extracurricullar/Bing housing real dataset/BUDatathon2018-master/351Bing_Complete.csv')
print(training.head())
print(training.describe())
print(training.info())
training["Address"] = training["Address"].astype('category')
print(training.info())
price=training['Price']
#Visual representation of space and bedroom
sns.pairplot(training)

# Display the plot
plt.show()
"""plt.subplot(1,2,1)
plt.scatter(training['Space Ft2'],price,color='blue',alpha=0.5)
plt.title('Space vs Price')
plt.xlabel("Space Ft2")
plt.ylabel("Price")
plt.legend(loc='upper right')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1,2,2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.scatter(training['Bedroom'],price,color='red', alpha=0.5)
plt.title('Bedroom vs Price')
plt.xlabel("Bedroom")
plt.ylabel("Price")
plt.legend(loc='upper center')
# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()
"""



# In[2]:


plt.subplot(1,3,1)
sns.lmplot(x='Bedroom',y='Price',data=training,hue='Laundry',palette='Set2')
plt.show()

plt.subplot(1,3,2)
sns.lmplot(x='Space Ft2', y='Price', data=training,row='Laundry')
plt.show()
plt.subplot(1,3,3)
sns.swarmplot(x='Bedroom',y='Price',data=training,hue='Laundry')
plt.show()




# In[3]:


plt.subplot(2,2,1)
plt.hist(price)
plt.show()
plt.subplot(2,2,2)

plt.hist(np.log(price))
plt.show()
plt.subplot(2,2,3)
sns.jointplot(x='Space Ft2',y='Price',data=training,kind='scatter')
plt.show()
plt.subplot(2,2,4)
sns.jointplot(x='Bedroom',y='Price',data=training)
plt.show()







# In[4]:


from bokeh.plotting import figure


# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='Space', y_axis_label='price')

# Add a circle glyph to the figure p
p.circle(training['Space Ft2'],price)

# Call the output_file() function and specify the name of the file
output_file('binghouse.html')

# Display the plot
show(p)


# In[54]:


# Import figure from bokeh.plotting


# In[5]:


data = training.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
y = np.log(data.Price)
X = data.drop(['Price','Date'], axis=1)
# partition the data
X_training, X_testing, y_training, y_testing = train_test_split(
                                    X, y, random_state=42, test_size=.33) # 42 for reproducible results, 33% for hold-out select


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#cross validation
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=10)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))







# In[6]:


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


