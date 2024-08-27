#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as panda
from sklearn.model_selection import train_test_split

#supervised learning using random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from math import sqrt

# Model persistence to avoid training https://scikit-learn.org/stable/model_persistence.html
import sklearn.externals
import joblib

# interactive charts, I prefer this over the static matplotlib chats.
# Really good documentation and very easy to make dashboards or embed the figures elsewhere because it uses js too
import plotly.express as px

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plot


# In[2]:


# preprocess the data

panda.set_option('display.max_columns', None)
panda.set_option('mode.chained_assignment', None)

# read the dataframe
pokemon_dataframe = panda.read_csv('./PokemonArchive/data/cleaned_pokemon_stats.csv', sep = ',')
pokemon_dataframe.head()


# In[3]:


# print colmns names 
list(pokemon_dataframe.columns.values)


# In[4]:


# Delete Nan 
pokemon_dataframe = pokemon_dataframe.dropna(axis=0, how='any')


# In[5]:


# X is features, y is win_percentage
# the features include the following ['id','name','primary_type','secondary_type','total','hp','attack','defense','special_atk','special_def','speed','generation']
# index starts from zero
X = pokemon_dataframe.iloc[:, 4:12].values
y = pokemon_dataframe.iloc[:, -2].values

# printing the values here will help get the accurate column to adjust, uncomment the two lines below for the same
# X
# y


# In[6]:


X


# In[7]:


target = pokemon_dataframe['attack']
features = pokemon_dataframe[['defense', 'speed', 'hp']]

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=8)

model_linear = LinearRegression()

model_linear.fit(x_train, y_train)
prediction = model_linear.predict(x_test)

print("Model score: {}".format(model_linear.score(x_test, y_test)))

mse = np.sqrt(mean_squared_error(y_test, prediction))
print("Mean squared error: {}".format(mse))


# In[8]:


plot.scatter(y_test, prediction)
plot.xlabel("Attack:")
plot.ylabel("Predicted attack:")
plot.title("Attack vs Predicted attack:")


# In[9]:


# Calculate R-squared
r_squared = r2_score(y_test, prediction)
print(r_squared)


# In[10]:


test_data = panda.read_csv('./PokemonArchive/data/tests.csv', sep = ',')
new_test_data = test_data[["First_pokemon","Second_pokemon"]].replace(pokemon_dataframe.name)
new_test_data.head()


# In[11]:


# Filter the dataframe to get rows for Charmander and Bulbasaur
charmander_row = pokemon_dataframe[pokemon_dataframe['name'] == 'Charmander']
bulbasaur_row = pokemon_dataframe[pokemon_dataframe['name'] == 'Bulbasaur']

# Extract the features for Charmander and Bulbasaur
charmander_features = charmander_row[['defense', 'speed', 'hp']]
bulbasaur_features = bulbasaur_row[['defense', 'speed', 'hp']]

# Predict attack stat for Charmander and Bulbasaur
charmander_predicted_attack = model_linear.predict(charmander_features)
bulbasaur_predicted_attack = model_linear.predict(bulbasaur_features)

# Extract the actual attack stat values for Charmander and Bulbasaur
charmander_actual_attack = charmander_row['attack'].values[0]
bulbasaur_actual_attack = bulbasaur_row['attack'].values[0]

print("Predicted attack stat for Charmander:", charmander_predicted_attack[0])
print("Actual attack stat for Charmander:", charmander_actual_attack)

print("\nPredicted attack stat for Bulbasaur:", bulbasaur_predicted_attack[0])
print("Actual attack stat for Bulbasaur:", bulbasaur_actual_attack)


# In[ ]:




