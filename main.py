import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Check for missing values
missing_values = data.isnull().sum()


# Fill missing values in 'bmi' with the median
data['bmi'].fillna(data['bmi'].median(), inplace=True)

# Check for missing values again

# Display the distribution of categorical variables and the target variable 'stroke'
categorical_variables = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']

fig, axs = plt.subplots(2, 3, figsize=(20, 10))

for ax, column in zip(axs.flatten(), categorical_variables):
    sns.countplot(x=column, data=data, ax=ax)
    ax.set_title(column)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

# Display summary statistics of numerical variables
numerical_variables = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
print(data[numerical_variables].describe())

# Create a correlation matrix for numerical variables
numerical_data = data[numerical_variables + ['stroke']]
correlation_matrix = numerical_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.1)
plt.title('Correlation Matrix')
plt.show()

# Drop 'id' column as it's not a feature for prediction
data.drop('id', axis=1, inplace=True)

# Perform one-hot encoding
data_encoded = pd.get_dummies(data)

# Split the data into features (X) and the target variable (y)
X = data_encoded.drop('stroke', axis=1)
y = data_encoded['stroke']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the size of the training and testing sets
print(X_test)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize the models
adaboost = AdaBoostClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)

# Train the models
adaboost.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Make predictions
y_pred_adaboost = adaboost.predict(X_test)
y_pred_rf = random_forest.predict(X_test)

# Evaluate the models
print("AdaBoost metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_adaboost))
print("Precision:", precision_score(y_test, y_pred_adaboost))
print("Recall:", recall_score(y_test, y_pred_adaboost))
print("F1 Score:", f1_score(y_test, y_pred_adaboost))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_adaboost))

print("Random Forest metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf))
