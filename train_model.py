
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Student dataset for practice
data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 2, 3, 5, 6,
        1, 4, 3, 5, 2, 6, 4, 1, 5, 3
    ],
    "attendance": [
        40, 50, 60, 75, 80, 90, 55, 65, 85, 95,
        35, 70, 60, 90, 45, 95, 80, 30, 85, 65
    ],
    "previous_marks": [
        25, 35, 45, 60, 75, 90, 40, 50, 80, 95,
        20, 65, 48, 85, 30, 92, 70, 15, 78, 55
    ],
    "performance": [
        "Needs Improvement", "Needs Improvement", "Average",
        "Good", "Good", "Good", "Needs Improvement", "Average",
        "Good", "Good", "Needs Improvement", "Average", "Average",
        "Good", "Needs Improvement", "Good", "Good",
        "Needs Improvement", "Good", "Average"
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Test accuracy: {accuracy:.2%}")

with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as student_model.pkl")
