import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ParkinsonDataset(Dataset):
    def __init__(self, file_path):
        # 1. Load the data
        df = pd.read_csv(file_path)
        
        # 2. Separate features (X) and target (y)
        # 'name' is just a label, 'status' is what we want to predict
        X_raw = df.drop(columns=['name', 'status']).values
        y_raw = df['status'].values

        # 3. Scaling is CRITICAL for medical AI
        # It ensures 'Jitter' (small numbers) isn't ignored compared to 'Frequency' (large numbers)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # 4. Convert to PyTorch Tensors
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]