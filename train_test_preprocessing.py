# Task 3: Train-Test Split & Preprocessing
# InternNova - Week 3 Introduction to Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load dataset
df = pd.read_csv("Dataset/student_performance.csv")

# 2. Remove missing values
df = df.dropna()

# 3. Select features and target
X = df[["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]]
y = df["Final_Result"]

# 4. Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 5. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Preprocessing using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Display results
print("Original Dataset Shape:", X.shape)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Data (First 5 Rows):")
print(X_train.head())

print("\nTesting Data (First 5 Rows):")
print(X_test.head())

print("\nScaled Training Data (First 5 Rows):")
print(X_train_scaled[:5])

print("\nTraining Labels:")
print(y_train)

print("\nTesting Labels:")
print(y_test)

print("\nTrain-Test Split and Preprocessing Completed Successfully!")