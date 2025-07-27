#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


PERSONALITY = "C:/Users/User/Documents/Personality"


# In[2]:


def personality_data(filename, personality = PERSONALITY):
    csv_path = os.path.join(personality, filename)
    return pd.read_csv(csv_path)

train_data = personality_data("train.csv")
test_data  = personality_data("test.csv")
train_data.head(15)


# In[3]:


train_data.info()


# In[4]:


train_data.describe()


# In[5]:


train_data["Stage_fear"].value_counts()[1]


# In[6]:


plt.figure(figsize=(15, 6))  # Two plots in one row—adjust height for side-by-side

# 1. Histogram: Time Spent Alone by Personality
plt.subplot(1, 2, 1)
sns.histplot(
    data=train_data,
    x='Time_spent_Alone',
    hue='Personality',
    multiple='stack',
    bins=10,
    palette={'Introvert':'blue', 'Extrovert':'orange'}
)
plt.title('Histogram of Time Spent Alone by Personality')
plt.xlabel('Time Spent Alone (hours)')
plt.ylabel('Frequency')

# 2. Count Plot: Personality Class Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='Personality', data=train_data, palette={'Introvert':'blue', 'Extrovert':'orange'})
plt.title('Personality Class Distribution')
plt.xlabel('Personality')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[7]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[8]:


num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Time_spent_Alone","Social_event_attendance","Going_outside","Friends_circle_size",
    "Post_frequency"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])


# In[9]:


num_pipeline.fit_transform(train_data)


# In[10]:


class MostFrequentImputer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[11]:


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Stage_fear", "Drained_after_socializing"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse_output=False)),
    ])


# In[12]:


cat_pipeline.fit_transform(train_data)


# In[13]:


preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[14]:


X_train = preprocess_pipeline.fit_transform(train_data)

y_train = train_data["Personality"]


# In[15]:


svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train, y_train)



# In[16]:


X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


# In[17]:


from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()


# In[18]:


forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()


# In[19]:


y_test_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test_pred)

print(accuracy)


# In[20]:


prediction_df = pd.DataFrame({
    "id": test_data["id"],
    "Predicted_Personality": y_pred
})


# In[21]:


print(prediction_df)


# In[22]:


prediction_df.head()


# In[ ]:




