import pandas as pd

# Path to dataset
file_path = r"D:\depression_prediction_project\data\train.csv"  # raw string to avoid path errors

# Load dataset
df = pd.read_csv(file_path)

# Display first 5 rows
print("\n🔹 First 5 Rows:")
print(df.head())

# Shape (rows, columns)
print("\n🔹 Dataset Shape:", df.shape)

# Data types and non-null counts
print("\n🔹 Dataset Info:")
print(df.info())

# Count missing values
print("\n🔹 Missing Values per Column:")
print(df.isnull().sum())

# Statistical summary of numeric columns
print("\n🔹 Numeric Summary:")
print(df.describe())

# Unique values insight for categorical columns
print("\n🔹 Unique values in categorical columns:")
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f"{col}: {df[col].nunique()} unique values")
