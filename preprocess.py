import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

# Paths
data_path = os.path.join("data", "train.csv")  # relative path works on Streamlit Cloud
save_dir = os.path.join("saved_models")
os.makedirs(save_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Drop columns not useful or mostly missing
drop_cols = ['id', 'Name', 'Academic Pressure', 'CGPA', 'Study Satisfaction']
df = df.drop(columns=drop_cols)

# Target column
target_col = 'Depression'

# Numeric columns
num_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']

# Fill missing numeric values with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Binary categorical columns
binary_cols = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# Map binary columns to 0/1
binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Other': 2}  # Other for gender mapped to 2
for col in binary_cols:
    df[col] = df[col].map(binary_mapping)

# Multi-class categorical columns
multi_cat_cols = ['City', 'Working Professional or Student', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree']

# Fill missing categorical values with mode
for col in multi_cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encode multi-class categorical columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(df[multi_cat_cols])
cat_feature_names = ohe.get_feature_names_out(multi_cat_cols)

# Combine numeric and categorical features
X = np.hstack([df[num_cols].values, X_cat])
y = df[target_col].values

# Scale numeric features only
scaler = StandardScaler()
X[:, :len(num_cols)] = scaler.fit_transform(X[:, :len(num_cols)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save preprocessing objects
joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
joblib.dump(ohe, os.path.join(save_dir, "onehot_encoder.joblib"))
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(save_dir, "data_split.joblib"))

print("Preprocessing completed. Train-test split and preprocessing objects saved.")


