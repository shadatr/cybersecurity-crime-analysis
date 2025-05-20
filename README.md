# Crime Data Mining Project

This project applies data mining techniques to analyze and predict crime categories based on various features such as location, victim demographics, weapon used, and time of occurrence.

## Project Components

### 1. Data Preprocessing and Model Training (`main.py`)

This script performs:
- Data loading and preprocessing
- Feature selection and engineering
- Train/test splitting (80/20)
- Model training (Random Forest and SVM)
- Anomaly detection using Isolation Forest
- Hyperparameter optimization
- Model evaluation and saving

### 2. Model Import and Demonstration (`import_models.py`)

This script demonstrates:
- Loading trained models
- Making predictions on sample data
- Displaying feature importance
- Anomaly detection on sample data

### 3. Interactive Prediction Tool (`predict.py`)

An interactive command-line tool that:
- Takes user input for crime details
- Makes predictions using the trained model
- Displays prediction confidence

### 4. Batch Prediction Tool (`batch_predict.py`)

A tool for processing multiple records that:
- Reads crime data from a CSV file
- Makes predictions for all records
- Saves results with confidence scores
- Provides a summary of predictions

## Models

The project includes two main models:

1. **Random Forest Classifier**: Used for predicting crime categories
   - Accuracy: ~77%
   - Best for: Main classification tasks

2. **Isolation Forest**: Used for anomaly detection
   - Identifies unusual crime patterns
   - Contamination rate: 5%

## Data Features

The following features are used for prediction:

| Feature | Description |
|---------|-------------|
| Area_ID | Police area identifier (1-21) |
| Victim_Age | Age of the victim |
| Victim_Sex | Sex of the victim (M/F/X) |
| Victim_Descent | Descent/ethnicity of victim |
| Weapon_Used_Code | Code for weapon used |
| Part 1-2 | Crime classification code |
| Time_Occurred | Time of occurrence (0-2359) |

## Crime Categories

The model predicts the following crime categories:
- Violent Crimes
- Property Crimes
- Crimes against Public Order
- Crimes against Persons
- Other Crimes
- Fraud and White-Collar Crimes

## Usage

### Training the Models

```bash
python main.py
```

### Interactive Prediction

```bash
python predict.py
```

### Batch Prediction

```bash
python batch_predict.py input_file.csv -o output_file.csv
```

Parameters:
- `input_file.csv`: CSV file containing crime data
- `-o, --output`: Output file (optional)
- `-m, --model`: Model file to use (default: best_rf_model.pkl)

## Results

The Random Forest model achieved:
- Overall accuracy: 77%
- Strong performance on Violent Crimes (95% recall)
- Moderate performance on other categories

## Files

- `train.csv`: Original training data
- `best_rf_model.pkl`: Optimized Random Forest model
- `isolation_forest_model.pkl`: Isolation Forest model for anomaly detection
- `anomalies.csv`: Detected anomalies in the dataset
- `rf_confusion_matrix.png`: Confusion matrix visualization
- `svm_confusion_matrix.png`: SVM model confusion matrix
- `feature_importance.png`: Feature importance visualization

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install requirements with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
``` 