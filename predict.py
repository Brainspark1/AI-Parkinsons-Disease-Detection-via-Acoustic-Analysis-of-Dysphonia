import torch
from model import ParkinsonNet # Ensure this matches your class name

def predict():
    # 1. These MUST match what you used in train.py
    INPUT_SIZE = 22  
    HIDDEN_SIZE = 64 
    OUTPUT_SIZE = 1  

    # 2. Initialize the model with the correct sizes
    model = ParkinsonNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    # 3. Load the weights
    # Note: added weights_only=True for 2026 best practices/security
    model.load_state_dict(torch.load("models/parkinsons_model.pth", weights_only=True))
    model.eval() 

    # 4. Prepare input data (ensure it has 22 features, not 10!)
    # In a real scenario, this would be a scaled row from your dataset
    dummy_input = torch.randn(1, 22) 

    with torch.no_grad():
        prediction = model(dummy_input)
        # Since we used BCEWithLogitsLoss in training, 
        # we apply sigmoid here to get a 0-1 probability
        probability = torch.sigmoid(prediction)
    
    result = "Parkinson's Detected" if probability.item() > 0.5 else "Healthy"
    print(f"Prediction: {result} ({probability.item()*100:.2f}%)")

if __name__ == "__main__":
    predict()