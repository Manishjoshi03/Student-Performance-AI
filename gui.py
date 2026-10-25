import tkinter as tk
from tkinter import messagebox
import pickle

# Load trained model
try:
    with open("student_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None

# Prediction function
def predict_performance():
    if model is None:
        messagebox.showerror(
            "Error",
            "Model not found! Run train_model.py first."
        )
        return

    try:
        name = name_entry.get().strip()
        hours = float(hours_entry.get())
        attendance = float(attendance_entry.get())
        marks = float(marks_entry.get())

        if not name:
            messagebox.showerror("Error", "Enter student name!")
            return

        if not 0 <= hours <= 24:
            messagebox.showerror(
                "Error", "Study hours must be between 0 and 24."
            )
            return

        if not 0 <= attendance <= 100 or not 0 <= marks <= 100:
            messagebox.showerror(
                "Error", "Attendance and marks must be between 0 and 100."
            )
            return

        data = [[hours, attendance, marks]]
        result = model.predict(data)[0]

        result_label.config(
            text=f"Student: {name}\nPerformance: {result}"
        )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter numbers for hours, attendance and marks."
        )

# Main window
root = tk.Tk()
root.title("Student Performance Prediction")
root.geometry("520x600")
root.configure(bg="#eef2f7")
root.resizable(False, False)

# Heading
heading = tk.Label(
    root,
    text="STUDENT PERFORMANCE",
    font=("Arial", 20, "bold"),
    bg="#eef2f7",
    fg="#172554"
)
heading.pack(pady=(25, 5))

subtitle = tk.Label(
    root,
    text="AI & Machine Learning Project",
    font=("Arial", 11),
    bg="#eef2f7",
    fg="#64748b"
)
subtitle.pack(pady=(0, 20))

# Form frame
form = tk.Frame(root, bg="white", padx=25, pady=20)
form.pack(padx=30, fill="x")

def add_field(label_text):
    label = tk.Label(
        form,
        text=label_text,
        font=("Arial", 11, "bold"),
        bg="white",
        fg="#334155"
    )
    label.pack(anchor="w", pady=(8, 4))

    entry = tk.Entry(
        form,
        font=("Arial", 12),
        relief="solid",
        bd=1
    )
    entry.pack(fill="x", ipady=6)
    return entry

name_entry = add_field("Student Name")
hours_entry = add_field("Daily Study Hours")
attendance_entry = add_field("Attendance Percentage")
marks_entry = add_field("Previous Exam Marks")

# Predict button
predict_button = tk.Button(
    root,
    text="PREDICT PERFORMANCE",
    font=("Arial", 12, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    command=predict_performance
)
predict_button.pack(pady=20, ipadx=20, ipady=10)

# Result
result_label = tk.Label(
    root,
    text="Your result will appear here",
    font=("Arial", 12, "bold"),
    bg="#eef2f7",
    fg="#166534",
    justify="center"
)
result_label.pack(pady=5)

root.mainloop()