import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ParkinsonNet # Or ParkinsonNet if you renamed it
from dataset import ParkinsonDataset

# 1. Setup Hyperparameters
INPUT_SIZE = 22  
HIDDEN_SIZE = 64 
OUTPUT_SIZE = 1  
EPOCHS = 50       # Increased epochs; medical data needs more time to converge
BATCH_SIZE = 16   # Smaller batches often help with small datasets
LEARNING_RATE = 0.001

# 2. Prepare Data
# Make sure the file path matches where you put your unzipped file!
dataset = ParkinsonDataset("data/parkinsons.data")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialize Model, Loss, and Optimizer
model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
print(f"Starting Training on {len(dataset)} samples...")
model.train() # Set model to training mode

for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad() 
        loss.backward()      
        optimizer.step()     
        
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    if (epoch + 1) % 10 == 0: # Print every 10 epochs to keep the console clean
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}")

# 5. Save the trained brain
# Ensure the 'models' folder exists before running!
import os
if not os.path.exists('models'):
    os.makedirs('models')

torch.save(model.state_dict(), "models/parkinsons_model.pth")
print("Model saved to models/parkinsons_model.pth")