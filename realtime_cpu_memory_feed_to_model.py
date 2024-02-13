import pandas as pd
import joblib
import psutil
import time

# Load the trained model
loaded_model = joblib.load('/home/telaverge/agent_filtering_system/best_model.joblib')

# Load the scaler used during training
scaler = joblib.load('/home/telaverge/agent_filtering_system/scaler.joblib')

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

def predict_signal(cpu_percent, memory_percent):
    # Make predictions on new data
    new_data = pd.DataFrame({'cpu_usage': [cpu_percent], 'memory_usage': [memory_percent]})
    scaled_new_data = scaler.transform(new_data)
    prediction = loaded_model.predict(scaled_new_data)
    return prediction[0]

def main():
    try:
        while True:
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            # Make prediction based on current usage
            prediction = predict_signal(cpu_usage, memory_usage)

            if prediction == "yes":
                # Print the CPU and memory usage only when prediction is "yes"
                # print(f"CPU Usage: {cpu_usage}%")
                # print(f"Memory Usage: {memory_usage}%")
                # print(f"Prediction: {prediction}\n")
                print(prediction,flush=True)
                break

            # time.sleep(5)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
