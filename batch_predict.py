import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime

def load_model(model_path='best_rf_model.pkl'):
    """Load the trained model"""
    print(f"Loading the model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model

def process_input_file(input_file, model):
    """Process input file and make predictions"""
    print(f"Reading input file: {input_file}")
    
    try:
        # Read the input file
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        required_columns = ['Area_ID', 'Victim_Age', 'Victim_Sex', 'Victim_Descent', 
                           'Weapon_Used_Code', 'Part 1-2', 'Time_Occurred']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Input file is missing required columns: {missing_columns}")
            return None
        
        # Extract features
        features = df[required_columns].copy()
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(features)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)
        confidence = np.max(probabilities, axis=1)
        
        # Add predictions and confidence to the original dataframe
        df['Predicted_Crime_Category'] = predictions
        df['Confidence'] = confidence
        
        return df
        
    except Exception as e:
        print(f"Error processing input file: {str(e)}")
        return None

def save_results(df, output_file=None):
    """Save prediction results to a file"""
    if output_file is None:
        # Generate default output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.csv"
    
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Results saved successfully!")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total records processed: {len(df)}")
    print("\nPrediction distribution:")
    print(df['Predicted_Crime_Category'].value_counts())
    
    # Calculate average confidence
    avg_confidence = df['Confidence'].mean()
    print(f"\nAverage prediction confidence: {avg_confidence:.2%}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch prediction for crime categories')
    parser.add_argument('input_file', help='Input CSV file with crime data')
    parser.add_argument('-o', '--output', help='Output file for predictions (default: predictions_<timestamp>.csv)')
    parser.add_argument('-m', '--model', default='best_rf_model.pkl', help='Model file to use (default: best_rf_model.pkl)')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Process input file
    result_df = process_input_file(args.input_file, model)
    
    # Save results if processing was successful
    if result_df is not None:
        save_results(result_df, args.output)
    
if __name__ == "__main__":
    main() 