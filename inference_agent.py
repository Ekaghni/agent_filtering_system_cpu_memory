import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('best_model.joblib')

# Load the scaler used during training
scaler = joblib.load('scaler.joblib')

# Make predictions on new data
new_data = pd.DataFrame({'cpu_usage': [1], 'memory_usage': [80]})  # Replace with your own data
scaled_new_data = scaler.transform(new_data)
prediction = loaded_model.predict(scaled_new_data)

# Print the prediction
print(f"Prediction: {prediction}")
