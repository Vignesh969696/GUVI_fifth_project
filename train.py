import os
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from model import DepressionMLP  # import the MLP model

# Paths
save_dir = "saved_models"
data_split_path = os.path.join(save_dir, "data_split.joblib")
model_save_path = os.path.join(save_dir, "model.pth")

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load preprocessed train/test split
X_train, X_test, y_train, y_test = joblib.load(data_split_path)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model
input_dim = X_train.shape[1]
model = DepressionMLP(input_dim).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        
        # Ensure outputs shape matches target shape [batch_size, 1]
        outputs = outputs.view(-1, 1)
        
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).view(-1,1)  # ensure shape [batch_size,1]
        preds = torch.sigmoid(test_outputs).round()
        acc = accuracy_score(y_test_tensor.cpu(), preds.cpu())
        f1 = f1_score(y_test_tensor.cpu(), preds.cpu())

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {acc:.4f} | F1: {f1:.4f}")

# Save trained model
torch.save(model.state_dict(), model_save_path)
print(f"Training complete. Model saved as {model_save_path}")


