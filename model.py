import torch.nn as nn
import torch.nn.functional as F

class ParkinsonNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParkinsonNet, self).__init__()
        # Input is 22 (the number of voice features in the dataset)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size) # Output 1 probability (Healthy vs PD)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # We use Sigmoid at the end to get a 0.0 to 1.0 probability
        return self.fc3(x)