import joblib
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model import DepressionMLP

# Load preprocessed data
_, X_test, _, y_test = joblib.load(r"D:\depression_prediction_project\saved_models\data_split.joblib")

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Initialize model
input_dim = X_test.shape[1]
model = DepressionMLP(input_dim)

# Load trained weights
model.load_state_dict(torch.load(r"D:\depression_prediction_project\saved_models\model.pth"))
model.eval()

# Make predictions
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int().numpy()

# Compute metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("Evaluation on Test Set:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
