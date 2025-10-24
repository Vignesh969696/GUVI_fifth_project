import os
import joblib
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import DepressionMLP

# Paths
save_dir = "saved_models"
data_split_path = os.path.join(save_dir, "data_split.joblib")
model_path = os.path.join(save_dir, "model.pth")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load preprocessed data (only test set)
_, X_test, _, y_test = joblib.load(data_split_path)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)

# Initialize model
input_dim = X_test.shape[1]
model = DepressionMLP(input_dim).to(device)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Make predictions
with torch.no_grad():
    logits = model(X_test_tensor).view(-1,1)  # ensure shape [batch_size,1]
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int()

# Compute metrics
accuracy = accuracy_score(y_test_tensor.cpu(), preds.cpu())
precision = precision_score(y_test_tensor.cpu(), preds.cpu())
recall = recall_score(y_test_tensor.cpu(), preds.cpu())
f1 = f1_score(y_test_tensor.cpu(), preds.cpu())

print("Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


