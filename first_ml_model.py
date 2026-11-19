# Task 4: First Machine Learning Model
# InternNova - Week 3 Introduction to Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("Dataset/student_performance.csv")

# 2. Clean dataset
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

# 6. Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# 7. Train model
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("First Machine Learning Model: Decision Tree")
print("------------------------------------------")
print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)
print("Accuracy:", accuracy * 100, "%")

# 10. Classification report (fixed)
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=encoder.classes_,
    zero_division=0
))

# 11. Predict result for a new student
new_student = pd.DataFrame(
    [[5, 85, 70, 75]],
    columns=["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]
)

prediction = model.predict(new_student)
result = encoder.inverse_transform(prediction)

print("\nNew Student Prediction:")
print("Study Hours: 5")
print("Attendance: 85")
print("Previous Marks: 70")
print("Assignment Score: 75")
print("Predicted Result:", result[0])

print("\nModel Training and Prediction Completed Successfully!")