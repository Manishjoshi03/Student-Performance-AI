
# 🎓 Student Performance AI

An interactive Machine Learning application that predicts student performance based on study hours, attendance, and previous marks.

## 🚀 Features

- Student performance prediction
- Interactive analytics dashboard
- Student records management using SQLite
- Performance charts and summary metrics
- CSV report download
- Bulk student data import
- Personalized study suggestions
- Dark-themed user interface

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- SQLite

## 📁 Project Structure

```text
Student-Performance-Prediction/
├── app.py
├── main.py
├── gui.py
├── train_model.py
├── student_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

The `students.db` database is created locally when the application runs.

## ⚙️ Installation

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Train the model if needed:

```bash
python train_model.py
```

5. Start the application:

```bash
python -m streamlit run app.py
```

## 📊 How It Works

Enter a student's study hours, attendance percentage, and previous exam marks. The application generates a performance estimate and displays analytics and study suggestions.

## ⚠️ Disclaimer

This is an educational demonstration project. Prediction quality depends on the training data and model evaluation. Predictions are estimates, not guaranteed academic outcomes.

## 👨‍💻 Author

**Manish Joshi**

- GitHub: https://github.com/Manishjoshi03
- LinkedIn: https://www.linkedin.com/in/manish-joshi-12367a3ba
