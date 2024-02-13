import pandas as pd
import joblib
import psutil
import time
import torch
import torch.nn as nn
import joblib
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)
def get_memory_usage():
    return psutil.virtual_memory().percent
def main():
    try:
        while True:
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()

            input_size = 2 
            model = NeuralNetwork(input_size)
            model.load_state_dict(torch.load('pytorch_model.pth'))
            scaler = joblib.load('scaler.joblib')
            input_data = scaler.transform([[cpu_usage, memory_usage]])
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)
            model = model.to(device)
            with torch.no_grad():
                model.eval()
                output_tensor = model(input_tensor)
                print("Output safetensors----> ",output_tensor.item())
                prediction = 'yes' if output_tensor.item() >= 0.7 else 'no'

            print("Prediction:", prediction)

            if prediction == "yes":
                print(f"CPU Usage: {cpu_usage}%")
                print(f"Memory Usage: {memory_usage}%")
                # print(f"Prediction: {prediction}\n")
                print(prediction,flush=True)
                break

            # time.sleep(5)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
