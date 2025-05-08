#!/usr/bin/env python
# coding: utf-8

# # Loan Elegibility

# ## Problem Statement
# Dream Housing Finance company deals in all home loans. They have a presence across
# all urban, semi-urban, and rural areas. Customer-first applies for a home loan after
# that company validates the customer eligibility for a loan. The company wants to
# automate the loan eligibility process (real-time) based on customer detail provided
# while filling the online application form. These details are Gender, Marital Status,
# Education, Number of Dependents, Income, Loan Amount, Credit History, and others.
# 
# We will attempt to tackle this problem in an efficient manner while maintaining a good accuracy.

# ## Dataset Description
# The dataset includes the following attributes: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status.
# 
# Loan_Status represents the status of the loan, where y depicts yes, and n depicts no.

# ## Importing The Needed Libraries

# In[142]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Decision Tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


# ## Exploratory Data Analysis

# In[143]:


df = pd.read_csv("loan_eligibility.csv")
df.head()


# In[144]:


df.drop(columns=['Loan_ID'], axis=1, inplace=True)
df.head()


# In[145]:


df.info()


# In[146]:


df.isnull().sum()


# ## Data Preparation and Visualization

# ### Handling Missing Data
# We could calculate the average and replace the missing data but we have gone with removing the rows with missing data.
# That leaves us with enough data, so it shouldn't be a problem

# In[147]:


df.dropna(inplace=True)
df.isnull().sum()
df.info()


# In[148]:


df.describe()


# In[149]:


sns.set_style("whitegrid")
sns.set_palette("bright")
b = sns.pairplot(data=df, hue="Loan_Status", diag_kind='hist')
plt.show()


# Visually we can tell that the Credit History and Loan Amount have a greater effect on the outcome than other attributes. We will confirm that during feature selection.

# ### Detecting Outliers
# We have visualized the data that has outliers, and we'll remove them by calculating the Z score and saving the data that has a Z score less than 3 (not outlier). Then we visualize the data again to show a more normal distribution. We tried to do this using IQR, but it left a lot of outliers.

# #### Loan Amount

# In[150]:


sns.histplot(data=df, x='LoanAmount', hue='Loan_Status', kde=True)


# In[151]:


from scipy import stats

z_scores = stats.zscore(df['LoanAmount'])
abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3)
new_df = df[filtered_entries]


# In[152]:


sns.histplot(data=new_df, x='LoanAmount', hue='Loan_Status', kde=True)


# #### Applicant Income

# In[153]:


sns.histplot(data=df, x='ApplicantIncome', hue='Loan_Status', kde=True)


# In[154]:


# use the already altered data frame
z_scores = stats.zscore(new_df['ApplicantIncome'])
abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3)
new_df = new_df[filtered_entries]


# In[155]:


sns.histplot(data=new_df, x='ApplicantIncome', hue='Loan_Status', kde=True)


# #### Coapplicant Income

# In[156]:


sns.histplot(data=df, x='CoapplicantIncome', hue='Loan_Status', kde=True)


# In[157]:


z_scores = stats.zscore(new_df['CoapplicantIncome'])
abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3)
new_df = new_df[filtered_entries]


# In[158]:


sns.histplot(data=new_df, x='CoapplicantIncome', hue='Loan_Status', kde=True)


# In[159]:


new_df.info()


# ### Data Reduction

# In[160]:


print(new_df["Loan_Status"].value_counts())


# The data is not sufficiently imbalanced, so we won't need any data reduction techniques.

# ### Feature Selection
# We will use Random Forest to select the best features.

# #### Encoding Categorical Data
# We will encode the categorical data to be able to use it in the Random Forest model.

# In[161]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

X = new_df.drop(columns=['Loan_Status'])
y = new_df['Loan_Status']

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])
X['Married'] = le.fit_transform(X['Married'])
X['Education'] = le.fit_transform(X['Education'])
X['Self_Employed'] = le.fit_transform(X['Self_Employed'])
X['Property_Area'] = le.fit_transform(X['Property_Area'])
X['Dependents'] = le.fit_transform(X['Dependents'])

print(X.head())


# In[162]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)

selector = SelectFromModel(
    estimator=rf,
    threshold='mean',
    prefit=False
)

# We get the selected features and their data
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()].tolist()
print(selected_features)


# ## Model Training and Testing

# ### Model Selection
# The data we have at hand requires a supervised machine learning model, a classification model, since it includes the classes. We have selected a Decision Tree model due to its fast prediction times suiting the requested real time performance.

# ### Training The Model
# We will attempt to train the model using the Decision Tree Classifier. First, we will use the selected features from the previous step. Second, we will use all the features. We will compare the results and choose the best model.

# ### Using Selected Features

# #### Splitting The Data
# We will split the data 70% for training and 30% for testing. We will use the selected features from the previous step.

# In[272]:


x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
# check the shape of the training data
x_train.shape


# In[273]:


# check the shape of the testing data
x_test.shape


# #### Training The Model

# In[307]:


# We have set a random_state to preserve the accuracy across runs
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model = model.fit(x_train, y_train)


# #### Model Evaluation
# We will evaluate the model by calculating the confusion matrix and classification report (precision, recall, f1-score). We will also visualize the decision tree.

# In[275]:


y_pred = model.predict(x_test)
print('Accuracy: {:0.4f}'.format(metrics.accuracy_score(y_test, y_pred) * 100))


# In[276]:


confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[277]:


print(classification_report(y_test,y_pred))


# In[169]:


text_representation = tree.export_text(model)
print(text_representation)


# In[170]:


fig = plt.figure(figsize=(60, 40))
tree.plot_tree(model,
               feature_names=selected_features.sort(),
               class_names=['Y', 'N'],
               filled=True,
               rounded=True)
# fig.savefig("Decision Tree Results.png")


# ### Using All Features

# #### Splitting The Data

# In[171]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# check the shape of the training data
x_train.shape


# In[172]:


# check the shape of the testing data
x_test.shape


# #### Training The Model

# In[173]:


model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model = model.fit(x_train, y_train)


# #### Model Evaluation
# We will evaluate the model by calculating the confusion matrix and classification report (precision, recall, f1-score). We will also visualize the decision tree.

# In[174]:


y_pred = model.predict(x_test)
print('Accuracy: {:0.4f}'.format(metrics.accuracy_score(y_test, y_pred) * 100))


# In[175]:


confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[176]:


print(classification_report(y_test,y_pred))


# In[177]:


fig = plt.figure(figsize=(60, 40))
tree.plot_tree(model,
               class_names=['Y', 'N'],
               filled=True,
               rounded=True)
# fig.savefig("Decision Tree Results.png")


# ## Results
# The model using the selected features performed better than the model using all the features across all 3 metrics measured. We will try KNN to see how the performance of the model could improve and what tradeoffs we will make.

# ## KNN

# ### Training The Model

# ### Using Selected Features

# #### Splitting The Data

# In[263]:


x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
# check the shape of the training data
x_train.shape


# In[264]:


x_test.shape


# #### Training The Model

# In[288]:


# We tried a range of more than 20 but the precision plateaus
neighbors = np.arange(1,20)
print(neighbors)


# In[289]:


# Create a test_accuracy array to store the accuracy of the model for different values of k
test_accuracy = np.empty(len(neighbors))


# In[290]:


i = 0
for k in neighbors:
    print(k)
    # Set up a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors = k)
    # Fit the model
    knn.fit(x_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(x_test, y_test)
    print(test_accuracy[i])
    i+= 1


# ##### Calculating Elbow Point

# In[291]:


sns.lineplot(x=neighbors, y=test_accuracy)
plt.title("Elbow Point")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()


# There is no clear elbow point but model precision improves drastically at k = 16

# In[268]:


# Select the best K value
knn = KNeighborsClassifier(n_neighbors=16)


# In[269]:


# Fit the model
knn.fit(x_train, y_train)


# In[270]:


# check accuracy
knn.score(x_test, y_test)


# In[278]:


y_pred = knn.predict(x_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[279]:


print(classification_report(y_test,y_pred))


# 

# ### Using All Features

# In[296]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# check the shape of the training data
x_train.shape


# In[297]:


x_test.shape


# #### Training The Model

# In[298]:


neighbors = np.arange(1,20)
print(neighbors)


# In[299]:


# Create a test_accuracy array to store the accuracy of the model for different values of k
test_accuracy = np.empty(len(neighbors))


# In[300]:


i = 0
for k in neighbors:
    print(k)
    # Set up a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors = k)
    # Fit the model
    knn.fit(x_train, y_train)
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(x_test, y_test)
    print(test_accuracy[i])
    i+= 1


# ##### Calculating Elbow Point

# In[301]:


sns.lineplot(x=neighbors, y=test_accuracy)
plt.title("Elbow Point")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()


# There is no clear elbow point but model precision peaks at k = 15.

# In[302]:


# Select the best K value
knn = KNeighborsClassifier(n_neighbors=15)


# In[303]:


# Fit the model
knn.fit(x_train, y_train)


# In[304]:


# check accuracy
knn.score(x_test, y_test)


# In[305]:


y_pred = knn.predict(x_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])


# In[306]:


print(classification_report(y_test,y_pred))


# ### Result
# The KNN model using the selected features performed better than the model using all the features across all 3 metrics measured.

# ## Conclusion
# The Decision Tree model performed and the KNN model performed comparably both with feature selection and without. Decision Tree is more suited to this problem than KNN for its prediction speed and efficiency, and since it yielded comparable results, it is the better model to use as it has preferable tradeoffs. The Decision Tree model with feature selection performed the best.
# 
# Future enhancement could include exploring a different approach to data cleaning for example by filling in the missing instead of removing it. We could also enhance the performance by sampling the data to make it more balanced.
