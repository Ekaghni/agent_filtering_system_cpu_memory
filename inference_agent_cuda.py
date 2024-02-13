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

input_size = 2 
model = NeuralNetwork(input_size)
model.load_state_dict(torch.load('pytorch_model.pth'))
scaler = joblib.load('scaler.joblib')
cpu_usage = 79 
memory_usage = 1
input_data = scaler.transform([[cpu_usage, memory_usage]])
input_tensor = torch.tensor(input_data, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
model = model.to(device)
with torch.no_grad():
    model.eval()
    output_tensor = model(input_tensor)
    print("Output safetensors----> ",output_tensor.item())
    prediction = 'yes' if output_tensor.item() >= 0.8 else 'no'

print("Prediction:", prediction)
