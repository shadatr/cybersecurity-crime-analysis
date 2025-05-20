import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

print("Loading trained models...")
# Load the trained models
rf_model = joblib.load('best_rf_model.pkl')
isolation_forest = joblib.load('isolation_forest_model.pkl')

print("Models loaded successfully!")

# Load a sample of the data for demonstration
print("\nLoading sample data for demonstration...")
df = pd.read_csv('train.csv')
sample_data = df.sample(n=10, random_state=42)  # Take 10 random samples

# Extract features used during training
features = ['Area_ID', 'Victim_Age', 'Victim_Sex', 'Victim_Descent', 
           'Weapon_Used_Code', 'Part 1-2', 'Time_Occurred']

X_sample = sample_data[features].copy()
y_sample = sample_data['Crime_Category'].copy()

print("\nSample data:")
print(sample_data[['Location', 'Crime_Category']].to_string())

# Make predictions using the Random Forest model
print("\n--- Random Forest Predictions ---")
y_pred = rf_model.predict(X_sample)
print("\nPredictions vs Actual:")
for i, (actual, pred) in enumerate(zip(y_sample, y_pred)):
    print(f"Sample {i+1}: Actual: {actual} | Predicted: {pred}")

# Calculate accuracy on the sample
accuracy = sum(y_sample == y_pred) / len(y_sample)
print(f"\nAccuracy on sample: {accuracy:.2f}")

# Detect anomalies in the sample using Isolation Forest
print("\n--- Anomaly Detection ---")

# Since we're having issues with the dimensions, let's recreate the preprocessing pipeline
# Define the preprocessing steps similar to what was used in training
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

# Apply preprocessing
X_sample_processed = preprocessor.fit_transform(X_sample)

try:
    # Try to predict anomalies using the full dataset approach
    print("\nApproach 1: Using the full dataset for anomaly detection")
    # Get all data and preprocess
    X_full = df[features].copy()
    X_full_processed = preprocessor.transform(X_full)
    
    # Fit a new Isolation Forest on the full dataset
    print("Fitting a new Isolation Forest model on the full dataset...")
    new_isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    new_isolation_forest.fit(X_full_processed)
    
    # Predict anomalies on the sample
    anomaly_scores = new_isolation_forest.predict(X_sample_processed)
    anomalies = pd.Series(anomaly_scores).map({1: 'Normal', -1: 'Anomaly'})
    
    print("\nAnomaly detection results:")
    for i, anomaly in enumerate(anomalies):
        print(f"Sample {i+1}: {anomaly}")
    
    # Count anomalies
    anomaly_count = sum(anomalies == 'Anomaly')
    print(f"\nNumber of anomalies detected in sample: {anomaly_count}")
except Exception as e:
    print(f"\nError with Approach 1: {e}")
    
    # Alternative approach: Fit a new Isolation Forest directly on the sample
    print("\nApproach 2: Fitting a new Isolation Forest directly on the sample")
    try:
        sample_isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        sample_isolation_forest.fit(X_sample_processed)
        
        anomaly_scores = sample_isolation_forest.predict(X_sample_processed)
        anomalies = pd.Series(anomaly_scores).map({1: 'Normal', -1: 'Anomaly'})
        
        print("\nAnomaly detection results:")
        for i, anomaly in enumerate(anomalies):
            print(f"Sample {i+1}: {anomaly}")
        
        # Count anomalies
        anomaly_count = sum(anomalies == 'Anomaly')
        print(f"\nNumber of anomalies detected in sample: {anomaly_count}")
    except Exception as e:
        print(f"\nError with Approach 2: {e}")
        print("\nSkipping anomaly detection due to errors.")

# Feature importance from Random Forest (if available)
if hasattr(rf_model.named_steps['classifier'], 'feature_importances_'):
    print("\n--- Feature Importance ---")
    # Get feature names after preprocessing
    # This is a simplified approach - in reality, we'd need to extract feature names from the preprocessor
    feature_importance = rf_model.named_steps['classifier'].feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(feature_importance)[::-1]
    
    print("\nFeature ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_importance)), feature_importance[indices], align="center")
    plt.xticks(range(len(feature_importance)), indices)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

print("\nModel import and demonstration complete!") 