import pickle

print("=" * 40)
print("   STUDENT PERFORMANCE PREDICTION")
print("=" * 40)

# Load trained machine learning model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Take student details
name = input("Enter Student Name: ")
study_hours = float(input("Enter Daily Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
previous_marks = float(input("Enter Previous Exam Marks: "))

# Make prediction using trained model
data = [[study_hours, attendance, previous_marks]]
prediction = model.predict(data)

# Display result
print("\n" + "=" * 40)
print("             RESULT")
print("=" * 40)
print("Student Name:", name)
print("Predicted Performance:", prediction[0])
print("=" * 40)