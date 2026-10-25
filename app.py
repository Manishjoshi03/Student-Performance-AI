
import os
import io
import pickle
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "students.db"
MODEL_PATH = "student_model.pkl"


# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d1b, #101a32);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #0b1223;
}
h1, h2, h3 {
    color: #93c5fd;
}
div[data-testid="stMetric"] {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(147, 197, 253, 0.25);
    padding: 18px;
    border-radius: 14px;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 9px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect(DB_PATH)



def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                study_hours REAL NOT NULL,
                attendance INTEGER NOT NULL,
                previous_marks INTEGER NOT NULL,
                performance TEXT NOT NULL,
                prediction_source TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        columns = [
            row[1] for row in conn.execute(
                "PRAGMA table_info(students)"
            ).fetchall()
        ]

        if "created_at" not in columns:
            conn.execute(
                "ALTER TABLE students ADD COLUMN created_at TEXT"
            )

        if "prediction_source" not in columns:
            conn.execute(
                "ALTER TABLE students ADD COLUMN prediction_source TEXT DEFAULT 'Unknown'"
            )

def save_student(name, hours, attendance, marks, performance, source):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO students
            (name, study_hours, attendance, previous_marks,
             performance, prediction_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name, hours, attendance, marks, performance, source
        ))


def load_students():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT id AS ID,
                   name AS Name,
                   study_hours AS "Study Hours",
                   attendance AS Attendance,
                   previous_marks AS "Previous Marks",
                   performance AS Performance,
                   prediction_source AS "Prediction Source",
                   created_at AS "Created At"
            FROM students
            ORDER BY id DESC
        """, conn)


init_db()


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except Exception:
        return None


model = load_model()


def predict_performance(hours, attendance, marks):
    if model is not None:
        features = pd.DataFrame(
            [[hours, attendance, marks]],
            columns=["study_hours", "attendance", "previous_marks"],
        )
        prediction = str(model.predict(features)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                confidence = float(
                    max(model.predict_proba(features)[0]) * 100
                )
            except Exception:
                pass

        return prediction, "Machine Learning Model", confidence

    # Transparent fallback: this is a rule, not an ML prediction.
    if hours >= 4 and attendance >= 75 and marks >= 60:
        result = "Good"
    elif hours >= 2 and attendance >= 60 and marks >= 40:
        result = "Average"
    else:
        result = "Needs Improvement"

    return result, "Rule-based fallback (model unavailable)", None


def study_advice(hours, attendance, marks, performance):
    tips = []

    if hours < 2:
        tips.append("Try setting aside a consistent daily study period.")
    if attendance < 75:
        tips.append("Work toward attending classes more regularly.")
    if marks < 50:
        tips.append("Review difficult topics and practise sample questions.")
    if not tips and performance == "Good":
        tips.append("Maintain your routine and keep revising regularly.")
    if not tips:
        tips.append("Continue regular revision and track your progress.")

    return tips


# ---------------- HEADER ----------------
st.title("🎓 Student Performance AI")
st.caption(
    "Academic Analytics  •  Machine Learning  •  Student Progress"
)

with st.sidebar:
    st.title("🚀 Student AI")
    st.caption("PERFORMANCE ANALYTICS")
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Performance",
            "Student Records",
            "Bulk Import",
            "About Project",
        ],
    )
    st.divider()
    st.write(
        "Model status: "
        + ("Loaded" if model is not None else "Unavailable — fallback enabled")
    )
    st.caption("Student records are stored in a local SQLite database.")


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.header("📊 Analytics Dashboard")

    df = load_students()
    total = len(df)

    good = int((df["Performance"] == "Good").sum()) if total else 0
    average = int((df["Performance"] == "Average").sum()) if total else 0
    needs = (
        int((df["Performance"] == "Needs Improvement").sum())
        if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students Analysed", total)
    c2.metric("🟢 Good", good)
    c3.metric("🟡 Average", average)
    c4.metric("🔴 Needs Improvement", needs)

    if total:
        c5, c6, c7 = st.columns(3)
        c5.metric("Avg. Study Hours", f"{df['Study Hours'].mean():.1f}")
        c6.metric("Avg. Attendance", f"{df['Attendance'].mean():.1f}%")
        c7.metric("Avg. Previous Marks", f"{df['Previous Marks'].mean():.1f}/100")

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                df,
                names="Performance",
                title="Performance Distribution",
                hole=0.48,
                color="Performance",
                color_discrete_map={
                    "Good": "#22c55e",
                    "Average": "#facc15",
                    "Needs Improvement": "#f87171",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.bar(
                df,
                x="Name",
                y="Study Hours",
                color="Performance",
                title="Study Hours by Student",
                hover_data=["Attendance", "Previous Marks"],
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df,
            x="Attendance",
            y="Previous Marks",
            color="Performance",
            size="Study Hours",
            hover_name="Name",
            title="Attendance vs Previous Marks",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Student Records")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No records yet. Open Predict Performance to add your first student.")

    st.caption(
        "Predictions are educational estimates. The model's reliability "
        "depends on the quality and size of its training data."
    )


# ---------------- PREDICTION ----------------
elif page == "Predict Performance":
    st.header("🤖 Predict Student Performance")
    st.write("Enter student details to generate an estimate.")

    with st.form("student_prediction_form"):
        name = st.text_input("Student Name")
        c1, c2, c3 = st.columns(3)

        with c1:
            hours = st.number_input(
                "Daily Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
            )
        with c2:
            attendance = st.slider(
                "Attendance (%)", 0, 100, 75
            )
        with c3:
            marks = st.slider(
                "Previous Exam Marks", 0, 100, 60
            )

        submitted = st.form_submit_button(
            "🚀 ANALYSE STUDENT", use_container_width=True
        )

    if submitted:
        if not name.strip():
            st.warning("Please enter the student's name.")
        else:
            try:
                prediction, source, confidence = predict_performance(
                    hours, attendance, marks
                )

                save_student(
                    name.strip(),
                    hours,
                    attendance,
                    marks,
                    prediction,
                    source,
                )

                st.success("Prediction saved to the database.")
                st.subheader(f"Result for {name.strip()}")
                st.markdown(f"## 🎯 {prediction}")

                if confidence is not None:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                    st.caption(
                        "Confidence is not the same as prediction accuracy."
                    )

                st.caption(f"Prediction method: {source}")

                for tip in study_advice(
                    hours, attendance, marks, prediction
                ):
                    st.info("💡 " + tip)

            except Exception as error:
                st.error("Prediction could not be completed.")
                st.code(str(error))


# ---------------- STUDENT RECORDS ----------------
elif page == "Student Records":
    st.header("👨‍🎓 Student Records")

    df = load_students()

    if not df.empty:
        search = st.text_input("🔍 Search by student name")
        if search.strip():
            df = df[
                df["Name"].str.contains(
                    search.strip(), case=False, na=False
                )
            ]

        choices = sorted(df["Performance"].dropna().unique().tolist())
        selected = st.multiselect(
            "Filter by performance",
            choices,
            default=choices,
        )
        df = df[df["Performance"].isin(selected)]

        st.write(f"Records found: **{len(df)}**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download CSV Report",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="student_performance_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Delete a Record")

        id_options = df["ID"].tolist()
        if id_options:
            chosen_id = st.selectbox("Select record ID", id_options)
            confirm = st.checkbox("I confirm I want to delete this record")

            if st.button("🗑️ Delete Selected Record", disabled=not confirm):
                with get_connection() as conn:
                    conn.execute(
                        "DELETE FROM students WHERE id = ?", (int(chosen_id),)
                    )
                st.success("Record deleted.")
                st.rerun()

        if st.button("🧹 Clear ALL database records"):
            st.session_state["confirm_clear_all"] = True

        if st.session_state.get("confirm_clear_all", False):
            st.warning("This permanently deletes all student records.")
            a, b = st.columns(2)
            if a.button("Confirm delete all", type="primary"):
                with get_connection() as conn:
                    conn.execute("DELETE FROM students")
                st.session_state["confirm_clear_all"] = False
                st.rerun()
            if b.button("Cancel deletion"):
                st.session_state["confirm_clear_all"] = False
                st.rerun()
    else:
        st.info("No student records yet. Add a prediction first.")


# ---------------- BULK IMPORT ----------------
elif page == "Bulk Import":
    st.header("📂 Import Students from CSV")
    st.write(
        "Upload a CSV with these columns: "
        "`Name`, `Study Hours`, `Attendance`, `Previous Marks`."
    )

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            required = [
                "Name", "Study Hours", "Attendance", "Previous Marks"
            ]

            if not all(col in incoming.columns for col in required):
                st.error("CSV columns do not match the required names.")
                st.write("Required columns:", required)
            else:
                st.dataframe(incoming.head(10), use_container_width=True)

                if st.button("Import and predict all rows"):
                    imported = 0
                    for _, row in incoming.iterrows():
                        student_name = str(row["Name"]).strip()
                        if not student_name or student_name.lower() == "nan":
                            continue

                        h = float(row["Study Hours"])
                        a = int(row["Attendance"])
                        m = int(row["Previous Marks"])

                        if not (0 <= h <= 24 and 0 <= a <= 100 and 0 <= m <= 100):
                            continue

                        result, source, _ = predict_performance(h, a, m)
                        save_student(student_name, h, a, m, result, source)
                        imported += 1

                    st.success(f"Imported {imported} valid student rows.")
        except Exception as error:
            st.error("Could not read this CSV file.")
            st.code(str(error))


# ---------------- ABOUT ----------------
elif page == "About Project":
    st.header("💡 About This Project")

    st.markdown("""
    ### Student Performance AI Analytics System

    This educational project combines machine learning with student
    record management and interactive visual analytics.

    **Core features**
    - Student performance estimates
    - Persistent SQLite storage
    - Searchable and filterable student records
    - Interactive Plotly charts
    - CSV report download and bulk import
    - Study habit suggestions

    **Technologies**
    - Python
    - Streamlit
    - Pandas
    - Plotly
    - SQLite
    - Scikit-learn model, when available
    """)

    st.warning(
        "This project is for learning and demonstration. Predictions are "
        "not guaranteed to reflect real academic outcomes. Validate the "
        "model on suitable real-world data before drawing conclusions."
    )


# ---------------- FOOTER ----------------
st.divider()
st.caption("Student Performance AI • Educational Project")
