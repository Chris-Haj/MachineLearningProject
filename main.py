import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Fill missing values in 'bmi' with the median
data['bmi'].fillna(data['bmi'].median(), inplace=True)

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

# Plot the distribution of 'stroke' before upsampling
plt.figure(figsize=(6, 4))
sns.countplot(x='stroke', data=data_encoded)
plt.title('Distribution of \'stroke\' (before upsampling)')
plt.show()

data_majority = data_encoded[data_encoded.stroke==0]
data_minority = data_encoded[data_encoded.stroke==1]

# Upsample minority class
data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=42)

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled])

# Plot the distribution of 'stroke' after upsampling
plt.figure(figsize=(6, 4))
sns.countplot(x='stroke', data=data_upsampled)
plt.title('Distribution of \'stroke\' (after upsampling)')
plt.show()

# Split the upsampled data into features (X) and the target variable (y)
X_upsampled = data_upsampled.drop('stroke', axis=1)
y_upsampled = data_upsampled['stroke']
X_train, X_test, y_train, y_test = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

# Initialize the models
adaboost = AdaBoostClassifier(random_state=42)
randomForest = RandomForestClassifier(random_state=42)

# Train the models
adaboost.fit(X_train, y_train)
randomForest.fit(X_train, y_train)

# Make predictions
y_pred_adaboost = adaboost.predict(X_test)
y_pred_rf = randomForest.predict(X_test)

# Evaluate the models
print("AdaBoost metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_adaboost))
print("Precision:", precision_score(y_test, y_pred_adaboost))
print("Recall:", recall_score(y_test, y_pred_adaboost))
print("F1 Score:", f1_score(y_test, y_pred_adaboost))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_adaboost))

print("\nRandom Forest metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf))

# Hyperparameter tuning with GridSearchCV
param_grid_ada = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
param_grid_rf = {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

grid_search_ada = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid_ada, cv=5, scoring='roc_auc')
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='roc_auc')

grid_search_ada.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)

# Train models with best parameters
adaboost_best = grid_search_ada.best_estimator_
random_forest_best = grid_search_rf.best_estimator_

adaboost_best.fit(X_train, y_train)
random_forest_best.fit(X_train, y_train)

# Make predictions
y_pred_adaboost_best = adaboost_best.predict(X_test)
y_pred_rf_best = random_forest_best.predict(X_test)

# Evaluate the models
print("\nAdaBoost metrics (after hyperparameter tuning):")
print("Accuracy:", accuracy_score(y_test, y_pred_adaboost_best))
print("Precision:", precision_score(y_test, y_pred_adaboost_best))
print("Recall:", recall_score(y_test, y_pred_adaboost_best))
print("F1 Score:", f1_score(y_test, y_pred_adaboost_best))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_adaboost_best))

print("\nRandom Forest metrics (after hyperparameter tuning):")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_best))
print("Precision:", precision_score(y_test, y_pred_rf_best))
print("Recall:", recall_score(y_test, y_pred_rf_best))
print("F1 Score:", f1_score(y_test, y_pred_rf_best))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf_best))