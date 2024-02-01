#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset/data
df = pd.read_csv("C:\\Users\\amit9\\Downloads\\py.csv\\dementia_patients_health_data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# # Data Preprocessing

# In[7]:


df.isnull().sum()


# In[8]:


categorical_columns = []
for column in df.columns:
    if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
        categorical_columns.append(column)
print("Categorical Columns :", categorical_columns,
     "\nNumber Of Columns :", len(categorical_columns))


# In[9]:


Numerical_columns = []
for column in df.columns:
    if df[column].dtype == 'int64' or pd.api.types.is_any_real_numeric_dtype(df[column]):
        Numerical_columns.append(column)
print("Numerical Columns :", Numerical_columns,
     "\nNumber Of Columns :", len(Numerical_columns))


# In[10]:


df['Prescription'].value_counts()


# In[11]:


df['Prescription'].unique()


# In[12]:


df['Dosage in mg'].value_counts()


# In[13]:


df['Dosage in mg'].unique()


# In[14]:


df['Chronic_Health_Conditions'].value_counts()


# In[15]:


# Since, we have only 3 variables with Nan values, 
#I will be replacing the Nan values with the relevant data points, 
#so, that we don't have to remove/alter the data set. 

df['Chronic_Health_Conditions'] = df['Chronic_Health_Conditions'].fillna('Unknown')
df['Prescription'] = df['Prescription'].fillna('Not Prescribed')
df['Dosage in mg'] = df['Dosage in mg'].fillna(0)


# In[16]:


df.isnull().sum()


# In[17]:


df.shape


# In[18]:


df.head()


# In[19]:


df.columns


# # Data Visualization:

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


for variables in Numerical_columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df[variables], bins=20, kde=True)
    plt.title(f'Histogram of {variables}')
    plt.xlabel(variables)
    plt.ylabel('Frequency')
    plt.show()    


# In[22]:


for variables in categorical_columns:
    plt.figure(figsize=(8,2))
    sns.countplot(data = df, x = variables)
    plt.title(f'Bar Plot of {variables}')
    plt.xlabel(variables)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()   


# In[23]:


#scatter plots for pairwise relationships between numerical_columns
sns.pairplot(df[Numerical_columns])
plt.suptitle('Pairwise Scatter Plots of Numerical Features', y=1.02)
plt.show()


# In[24]:


plt.figure(figsize=(10,5))
sns.heatmap(df[Numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap of Numerical Features')
plt.show()           


# In[25]:


category = df['Prescription']

plt.figure(figsize=(45, 25))
for idx, variable in enumerate(Numerical_columns):
    plt.subplot(5, 6, idx + 1)
    sns.boxplot(x= category, y=variable, data=df)
    plt.title(f'Box Plot of {variable} across Prescription')
    plt.xlabel('Diabetic')
    plt.ylabel(variable)

plt.tight_layout()
plt.show()


# In[26]:


category = df['Prescription']

plt.figure(figsize=(45, 25))
for idx, variable in enumerate(Numerical_columns):
    sns.violinplot(x=category, y=variable, data=df)
    plt.title(f'Violin Plot of {variable} across Prescription', fontweight='bold', fontsize=25)
    plt.xlabel('Prescription', fontsize=30)
    plt.ylabel(variable, fontsize=30)
    plt.xticks(rotation=45,fontsize=30 )
    plt.yticks(fontsize=30 )

plt.tight_layout()
plt.show()


# In[27]:


# camparision b/w Smoking habit & Chronical Diseases

plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Smoking_Status', 'Chronic_Health_Conditions']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Smoking Status and Chronic Health Conditions')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Chronic Health Conditions')
plt.tight_layout()
plt.show()


# In[28]:


df.columns


# In[29]:


plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Depression_Status', 'Dementia']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Depression Status and Dementia')
plt.xlabel('Depression Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Dementia')
plt.tight_layout()
plt.show()

print("\nPerson dignosed with Dementia were surely having Depression")


# In[30]:


plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Sleep_Quality', 'Chronic_Health_Conditions']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Sleep Quality and Chronic Health Conditions')
plt.xlabel('Sleep Quality')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Chronic Health Conditions')
plt.tight_layout()
plt.show()

print("\nPerson having Chronical Condition and their sleeping habits")


# In[31]:


plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Medication_History', 'Dementia']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Medication_History and Dementia')
plt.xlabel('Medication_History')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Dementia')
plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Age', 'Dementia']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Age and Dementia')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Dementia')
plt.tight_layout()
plt.show()


# In[33]:


#Let us find which are the top 5 reasons which are most dependable for diagnosing dementia.

from sklearn.ensemble import RandomForestClassifier

x = df.drop('Dementia', axis=1)
x = pd.get_dummies(x)

y = df['Dementia']

model = RandomForestClassifier()
model.fit(x,y)

depandable_variable = model.feature_importances_

depandable_variable_df = pd.DataFrame({'Dependable_Variable':x.columns, 'Importance' : depandable_variable})
depandable_variable_df = depandable_variable_df.sort_values(by='Importance', ascending=False)

print("Top 5 most important features for diagnosing dementia:"
      "\n",
     depandable_variable_df.head(5))


# In[34]:


plt.figure(figsize=(10, 10))
plt.barh(depandable_variable_df['Dependable_Variable'], depandable_variable_df ['Importance'])
plt.xlabel('Importance')
plt.ylabel('Dependable_Variable')
plt.title('Most Dependable variables for Predicting Dementia')
plt.gca().invert_yaxis()
plt.show()


# In[35]:


plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Dosage in mg', 'Dementia']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Dosage in mg and Dementia')
plt.xlabel('Dosage in mg')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Dementia')
plt.show()


# In[36]:


plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['Prescription', 'Dementia']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of Prescription and Dementia')
plt.xlabel('Prescription')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Dementia')
plt.show()


# In[37]:


#APOE_ε4: Shows the presence of the APOE ε4 allele, a genetic variant associated with Alzheimer's disease.
plt.figure(figsize=(10,6))

smoking_vs_chronic_diseases = df.groupby(['APOE_ε4', 'Dementia']).size().unstack(fill_value=0)
smoking_vs_chronic_diseases.plot(kind='bar', stacked=True)
plt.title('Comparison of APOE_ε4 and Dementia')
plt.xlabel('APOE_ε4')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Dementia')
plt.show()


# # Machine Learning

# In[38]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


# In[40]:


model_r = RandomForestClassifier()
model_r.fit(x_train, y_train)

y_pred = model_r.predict(x_test)


# In[41]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[42]:


kfold  = KFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_val_score(model_r, x,y, cv=kfold, scoring='accuracy')

print("Cross-Validation Results:")
print("Mean Accuracy:", cv.mean())
print("Standard Deviation:", cv.std())


# In[43]:


confusion_met = confusion_matrix(y_test, y_pred)
confusion_met


# In[44]:


plt.figure(figsize=(5, 2))
sns.heatmap(confusion_met, annot=True, cmap='Blues', fmt='g', 
            xticklabels=['False Negative', 'False Positive'], 
            yticklabels=['True Negative', 'True Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# # Logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


model_l = LogisticRegression(max_iter=1000)
model_l.fit(x_train, y_train)

y_pred_l = model_l.predict(x_test)
y_pred_l


# In[47]:


accuracy_ = accuracy_score(y_test, y_pred_l)
print("Accuracy :", accuracy_)


# In[48]:


print("Classification Report :",
      "\n",
      "\n",
      classification_report(y_test, y_pred_l))


# In[49]:


print("Confusion Metrix :", 
      "\n",
      confusion_matrix(y_test, y_pred_l))


# In[ ]:




