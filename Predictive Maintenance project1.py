#!/usr/bin/env python
# coding: utf-8

# # EDA

# ### a.Describe the dataset

# #### Data Preprocessing

# In[1]:


#load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("Maintenance.csv")


# In[3]:


#dataset overview
df.head()


# In[4]:


df.tail()


# In[5]:


print(df.dtypes)


# In[6]:


print(df["Type"].unique())
print(df["Product ID"].unique())


# In[7]:


df.info()


# In[8]:


df.describe()


# ### b. Cleaning The Data

# In[9]:


# Cleaning column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[', '').str.replace(']', '').str.lower()

# Verify the column names
print(df.columns)


# ### c. Checking Invalid Records

# In[10]:


# Example: Checking for negative values in numerical columns
invalid_records = df[(df['air_temperature_k'] < 0) | 
                     (df['process_temperature_k'] < 0) | 
                     (df['rotational_speed_rpm'] < 0) | 
                     (df['torque_nm'] < 0) | 
                     (df['tool_wear_min'] < 0)]


# ### d. Missing Value Detection and Imputation
# 

# In[11]:


#check for missing values
missing_values=df.isnull().sum()
missing_values


# In[12]:


#inpute missing value
# df.fillna(method='ffill',inplace=True)


# ### e. Duplicated Records
# 

# In[13]:


duplicates=df.duplicated().sum()
df.drop_duplicates(inplace=True)


# ### f.Outliers

# In[14]:


#detect outlier
import seaborn as sns 
import matplotlib.pyplot as plt


# In[15]:


numerical_columns=["air_temperature_k","process_temperature_k","rotational_speed_rpm", "torque_nm", "tool_wear_min"]
for column in numerical_columns:
    sns.boxplot(x=df[column])
    plt.show()


# In[16]:


#handle outliers
#removing outlier using IQR method
Q1=df[numerical_columns].quantile(0.25)
Q3=df[numerical_columns].quantile(0.75)
IQR=Q3-Q1
df=df[~((df[numerical_columns]<(Q1-1.5*IQR))  |  (df[numerical_columns]>(Q3+1.5*IQR))).any(axis=1)]


# # 2. Data Visualization
# 

# ### 1.Histograms

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns 
sns.histplot(df['air_temperature_k'],bins=30,kde=True)
plt.title('Distribution of Air Temperature')
plt.xlabel('Air Temperature[k]')
plt.ylabel('Frequency')
plt.show()


# ### 2. Barcharts

# In[18]:


sns.countplot(x='type', data=df)
plt.title('Number of Machine Failures by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# ### 3.Boxplots

# In[19]:


sns.boxplot(x='type', y='torque_nm', data=df)
plt.title('Distribution of Torque by Type')
plt.xlabel('Type')
plt.ylabel('Torque [Nm]')
plt.show()


# ### 4. Scatter Plots
# 

# In[20]:


sns.scatterplot(x='air_temperature_k', y='process_temperature_k', data=df, hue='machine_failure')
plt.title('Air Temperature vs Process Temperature')
plt.xlabel('Air Temperature [K]')
plt.ylabel('Process Temperature [K]')
plt.show()


# ### 5.Line chart

# In[21]:


df['time']=pd.to_datetime(df['udi'])
df.set_index('time',inplace=True)
df['tool_wear_min'].plot(figsize=(10,6))
plt.title('Tool wear over time')
plt.xlabel('time')
plt.ylabel('Tool wear [min]')
plt.show()


# ### 6.Pair plot

# In[22]:


sns.pairplot(df[['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'machine_failure']], hue='machine_failure',corner=True)
plt.show()


# ### 7.Heatmaps

# In[23]:


correlation_matrix=df.corr()
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm')
plt.title('correlation matrix')
plt.show()


# ### 8 Pie Chart

# In[24]:


type_counts = df['type'].value_counts()
type_counts.plot.pie(autopct='%1.1f%%', figsize=(5,5), legend=True)
plt.title('Proportion of Each Type')
plt.show()


# # Feature Engineering

# In[25]:


df


# In[26]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


# In[27]:


# Verify the columns to ensure 'machine_failure' exists
print("Columns in the dataframe:", df.columns)


# In[28]:


# Create new features
df['temperature_diff'] = df['process_temperature_k'] - df['air_temperature_k']
df['torque_speed_interaction'] = df['torque_nm'] * df['rotational_speed_rpm']


# In[29]:


# Plot new features
plt.figure(figsize=(10, 6))
sns.histplot(df['temperature_diff'], bins=50, kde=False)
plt.title('Distribution of Temperature Difference')
plt.show()


# In[30]:


plt.figure(figsize=(10, 6))
sns.histplot(df['torque_speed_interaction'], bins=50, kde=True)
plt.title('Distribution of Torque-Speed Interaction')
plt.show()


# In[31]:


# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['type'], drop_first=True)
df


# In[32]:


# Label encoding for high cardinality categorical variable
le = LabelEncoder()
df['product_id'] = le.fit_transform(df['product_id'])


# In[33]:


# Standard scaling(standaddize features by removing the mean & scaling to unit variance)
scaler = StandardScaler()
numerical_columns = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'temperature_diff', 'torque_speed_interaction']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[34]:


# Transform features by scaling each feature to a given range, e.g., between 0 and 1:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[35]:


# Plot scaled features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns])
plt.title('Boxplot of Scaled Numerical Features')
plt.xticks(rotation=45)
plt.show()


# In[36]:


# Feature selection: Variance Threshold(removing low variance feature)
selector = VarianceThreshold(threshold=0.01)
df_var_thresh = selector.fit_transform(df.drop('machine_failure', axis=1))
df_var_thresh = pd.DataFrame(df_var_thresh, columns=df.drop('machine_failure', axis=1).columns[selector.get_support()])


# In[37]:


# Feature selection: Recursive Feature Elimination
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(df_var_thresh, df['machine_failure'])
selected_features = df_var_thresh.columns[fit.support_]
df_selected = df[selected_features]

# Plot selected features
plt.figure(figsize=(10, 6))
sns.barplot(x=selected_features, y=fit.estimator_.feature_importances_)
plt.title('Feature Importances from RFE')
plt.xticks(rotation=45)
plt.show()


# #### handle imbalanced data

# In[38]:


# Add the target column back before resampling
df_selected['machine_failure'] = df['machine_failure']


# In[39]:


# Handling imbalanced data manually using upsampling
# Separate majority and minority classes
df_majority = df_selected[df_selected['machine_failure'] == 0]
df_minority = df_selected[df_selected['machine_failure'] == 1]


# In[40]:


# Upsample minority class to match the size of majority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Shuffle the dataset
df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)


# In[41]:


# Separate features and target variable
X_resampled = df_upsampled.drop('machine_failure', axis=1)
y_resampled = df_upsampled['machine_failure']


# In[42]:


# Display the final resampled DataFrame
print(df_upsampled.head())


# # Model Building . Hyperparameter Tuning & Evaluation

# In[43]:


# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[44]:


# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'k-Nearest Neighbors': KNeighborsClassifier()
}


# In[45]:


from sklearn.model_selection import cross_validate, StratifiedKFold
# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


# In[ ]:


# Train and evaluate models with cross-validation
results = {}
for model_name, model in models.items():
    scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=scoring)
    results[model_name] = {metric: scores[f'test_{metric}'].mean() for metric in scoring}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)


# #### Model Evaluation

# In[ ]:


# Plot evaluation results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, metric in enumerate(scoring):
    ax = axes[i // 3, i % 3]
    sns.barplot(x=results_df.index, y=results_df[metric], ax=ax)
    ax.set_title(f'Mean {metric.capitalize()}')
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=45)

# Remove the empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()


# by using random forest model we get approximately 100% accuracy so hyperparameter tuning is not reguired

# # model Deployment

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


# In[ ]:


# Define the preprocessing and model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(pipeline, 'trained_model.joblib')


# In[ ]:


# Load the model from the file
loaded_model = joblib.load('trained_model.joblib')

# Make predictions on the test data
y_pred = loaded_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[ ]:


pip install Flask


# In[ ]:


from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('trained_model.joblib')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        
        # Convert data into a numpy array
        input_data = np.array([data['features']])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Return the result as JSON
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




