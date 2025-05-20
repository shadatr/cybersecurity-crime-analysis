import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('train.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n--- Data Preprocessing ---")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values and select relevant features
print("\nHandling missing values and selecting features...")

# Select features for the model
# We'll use a subset of columns that are likely to be predictive
# You can modify this list based on your domain knowledge or feature importance analysis
features = ['Area_ID', 'Victim_Age', 'Victim_Sex', 'Victim_Descent', 
           'Weapon_Used_Code', 'Part 1-2', 'Time_Occurred']

# Target variable
target = 'Crime_Category'

# Print unique values in target variable
print(f"\nTarget variable ({target}) unique values:")
print(df[target].value_counts())

# Create X and y
X = df[features].copy()
y = df[target].copy()

# Handle categorical features and missing values
categorical_features = ['Victim_Sex', 'Victim_Descent']
numeric_features = ['Area_ID', 'Victim_Age', 'Weapon_Used_Code', 'Part 1-2', 'Time_Occurred']

# Define preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train/Test split
print("\n--- Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Model Training and Evaluation
print("\n--- Model Training and Evaluation ---")

# 1. Random Forest Classifier
print("\n1. Random Forest Classifier")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nRandom Forest Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('rf_confusion_matrix.png')
print("Confusion matrix saved as 'rf_confusion_matrix.png'")

# 2. Support Vector Machine
print("\n2. Support Vector Machine")
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))
])

# Train the model
svm_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_pipeline.predict(X_test)

# Evaluate the model
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nSVM Confusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('svm_confusion_matrix.png')
print("Confusion matrix saved as 'svm_confusion_matrix.png'")

# 3. Anomaly Detection with Isolation Forest
print("\n3. Anomaly Detection with Isolation Forest")

# Preprocess data for anomaly detection
X_preprocessed = preprocessor.fit_transform(X)

# Apply Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_scores = isolation_forest.fit_predict(X_preprocessed)

# Convert predictions to binary (1 for normal, -1 for anomaly)
anomalies = pd.Series(anomaly_scores).map({1: 0, -1: 1})

# Add anomaly flag to original dataframe
df_anomalies = df.copy()
df_anomalies['is_anomaly'] = anomalies.values

# Save anomalies to CSV
anomaly_data = df_anomalies[df_anomalies['is_anomaly'] == 1]
anomaly_data.to_csv('anomalies.csv', index=False)
print(f"\nNumber of anomalies detected: {anomaly_data.shape[0]}")
print("Anomalies saved to 'anomalies.csv'")

# Analyze anomalies by crime category
print("\nAnomalies by Crime Category:")
anomaly_by_category = anomaly_data['Crime_Category'].value_counts()
print(anomaly_by_category)

# Hyperparameter Optimization
print("\n--- Hyperparameter Optimization ---")

# Grid search for Random Forest
print("\nOptimizing Random Forest...")
param_grid_rf = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=3, n_jobs=-1, verbose=0)
grid_search_rf.fit(X_train, y_train)

print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best cross-validation score: {grid_search_rf.best_score_:.4f}")

# Make predictions with the optimized model
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("\nOptimized Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best_rf))

# Save the best model
print("\n--- Saving Models ---")
import joblib
joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(isolation_forest, 'isolation_forest_model.pkl')
print("Models saved as 'best_rf_model.pkl' and 'isolation_forest_model.pkl'")

print("\n--- Data Mining Complete ---")
