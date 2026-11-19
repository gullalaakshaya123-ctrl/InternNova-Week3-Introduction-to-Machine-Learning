# Task 5: Mini Project - Student Performance Prediction
# InternNova - Week 3 Introduction to Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("Dataset/student_performance.csv")

# 2. Clean dataset
df = df.dropna()

# 3. Select features and target
features = ["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]
X = df[features]
y = df["Final_Result"]

# 4. Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTION MINI PROJECT")
print("=" * 50)

print("\nModel Accuracy:", accuracy * 100, "%")

# 8. Predict for 3 new students
new_students = pd.DataFrame([
    [2, 60, 40, 45],
    [5, 85, 70, 75],
    [8, 95, 90, 95]
], columns=features)

predictions = model.predict(new_students)
results = encoder.inverse_transform(predictions)

print("\nPredictions for New Students:")
for i in range(len(new_students)):
    print(f"\nStudent {i + 1}:")
    print("Study Hours:", new_students.iloc[i]["Study_Hours"])
    print("Attendance:", new_students.iloc[i]["Attendance"])
    print("Previous Marks:", new_students.iloc[i]["Previous_Marks"])
    print("Assignment Score:", new_students.iloc[i]["Assignment_Score"])
    print("Predicted Result:", results[i])

print("\n" + "=" * 50)
print("MINI PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 50)