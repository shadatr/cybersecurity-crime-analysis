import pandas as pd
import numpy as np
import joblib

# Load the trained model
print("Loading the trained model...")
model = joblib.load('best_rf_model.pkl')
print("Model loaded successfully!")

def get_user_input():
    """Get crime details from the user"""
    print("\nPlease enter crime details:")
    
    # Get input for each feature
    area_id = int(input("Area ID (1-21): "))
    victim_age = int(input("Victim Age (0-99): "))
    
    # Get victim sex with validation
    while True:
        victim_sex = input("Victim Sex (M/F/X): ").upper()
        if victim_sex in ['M', 'F', 'X']:
            break
        print("Invalid input. Please enter M, F, or X.")
    
    # Get victim descent with validation
    print("\nVictim Descent Codes:")
    print("A - Asian, B - Black, H - Hispanic, W - White, O - Other")
    while True:
        victim_descent = input("Victim Descent (A/B/H/W/O/X): ").upper()
        if victim_descent in ['A', 'B', 'H', 'W', 'O', 'X']:
            break
        print("Invalid input. Please enter A, B, H, W, O, or X.")
    
    # Get weapon code
    weapon_code = int(input("\nWeapon Used Code (e.g., 100-500): "))
    
    # Get Part 1-2 code
    part_code = int(input("Part 1-2 Code (1 or 2): "))
    
    # Get time occurred
    time_occurred = int(input("Time Occurred (0-2359): "))
    
    # Create a dictionary of inputs
    input_data = {
        'Area_ID': area_id,
        'Victim_Age': victim_age,
        'Victim_Sex': victim_sex,
        'Victim_Descent': victim_descent,
        'Weapon_Used_Code': weapon_code,
        'Part 1-2': part_code,
        'Time_Occurred': time_occurred
    }
    
    return input_data

def predict_crime_category(input_data):
    """Predict crime category based on input data"""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Get prediction probability
    probabilities = model.predict_proba(input_df)
    max_prob = np.max(probabilities)
    
    return prediction[0], max_prob

def main():
    print("Crime Category Prediction Tool")
    print("=" * 30)
    
    while True:
        # Get user input
        input_data = get_user_input()
        
        # Make prediction
        predicted_category, confidence = predict_crime_category(input_data)
        
        # Display results
        print("\n" + "=" * 50)
        print(f"Predicted Crime Category: {predicted_category}")
        print(f"Confidence: {confidence:.2%}")
        print("=" * 50)
        
        # Ask if user wants to make another prediction
        another = input("\nDo you want to make another prediction? (y/n): ").lower()
        if another != 'y':
            break
    
    print("\nThank you for using the Crime Category Prediction Tool!")

if __name__ == "__main__":
    main() 