# Task 2: Dataset Preparation
# InternNova - Week 3 Introduction to Machine Learning

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("Dataset/student_performance.csv")

# 2. Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# 3. Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# 4. Display column data types
print("\nData Types:")
print(df.dtypes)

# 5. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Clean the dataset
df = df.dropna()

print("\nDataset After Cleaning:")
print(df.head())

# 7. Select features and target
X = df[["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]]
y = df["Final_Result"]

# 8. Display features and target
print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nDataset Preparation Completed Successfully!")