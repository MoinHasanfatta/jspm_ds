import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

DATA_FILE = Path("StudentPerformanceFactors.csv")
MODEL_FILE = Path("student_performance_model.pkl")

if not DATA_FILE.exists():
    st.error(
        "CSV file not found. Place StudentPerformanceFactors.csv "
        "in the same folder as app.py."
    )
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


df_original = load_data()


# --------------------------------------------------
# 3. CLEAN DATA
# --------------------------------------------------

def clean_data(data):
    data = data.copy()

    # Remove spaces around text values
    text_columns = data.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in text_columns:
        data[col] = data[col].str.strip()

    # Remove exact duplicate rows
    data = data.drop_duplicates()

    # Convert exam scores to numeric values
    if "Exam_Score" in data.columns:
        data["Exam_Score"] = pd.to_numeric(
            data["Exam_Score"],
            errors="coerce"
        )

        # Keep valid scores between 0 and 100
        data = data[
            data["Exam_Score"].between(0, 100)
        ].copy()

    return data


df_clean = clean_data(df_original)


# --------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(data):

    features = [
        "Hours_Studied",
        "Attendance",
        "Previous_Scores",
        "Sleep_Hours",
        "Tutoring_Sessions",
        "Motivation_Level",
        "Internet_Access",
        "Access_to_Resources",
        "Parental_Involvement"
    ]

    required = features + ["Exam_Score"]

    missing_columns = [
        col for col in required
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: " + ", ".join(missing_columns)
        )

    # Select inputs and target
    X = data[features].copy()
    y = data["Exam_Score"].copy()

    # Identify numerical and categorical columns
    numerical = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])

    # Create complete model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict test records
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    return (
        model,
        features,
        X_test,
        y_test,
        predictions,
        mae,
        rmse,
        r2
    )


# Prepare data for modeling
df_model = df_clean.dropna(
    subset=["Exam_Score"]
).copy()


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎓 Student Performance")

st.sidebar.write(
    "Analyze student data and predict exam scores."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Data Overview",
        "Data Cleaning",
        "EDA Visualizations",
        "Make Prediction",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "A data science project using Streamlit, Pandas "
    "and Scikit-learn."
)


# --------------------------------------------------
# 6. HOME DASHBOARD
# --------------------------------------------------

if page == "Home":

    st.title("🎓 Student Performance Prediction")

    st.write(
        "Analyze student data, explore insights, "
        "and predict exam scores using machine learning."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df_original))
    col2.metric("Total Columns", len(df_original.columns))
    col3.metric(
        "Missing Values",
        int(df_original.isnull().sum().sum())
    )
    col4.metric(
        "Duplicate Rows",
        int(df_original.duplicated().sum())
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Dataset Preview")

        st.dataframe(
            df_original.head(),
            use_container_width=True
        )

    with right:
        st.subheader("Exam Score Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            data=df_clean,
            x="Exam_Score",
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_title("Distribution of Exam Scores")

        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 7. DATA OVERVIEW
# --------------------------------------------------

elif page == "Data Overview":

    st.title("📊 Data Overview")

    st.subheader("First Five Records")

    st.dataframe(
        df_original.head(),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    col1.metric("Rows", df_original.shape[0])
    col2.metric("Columns", df_original.shape[1])

    st.subheader("Column Names")

    st.write(df_original.columns.tolist())

    st.subheader("Data Types")

    type_data = pd.DataFrame({
        "Column": df_original.columns,
        "Data Type": df_original.dtypes.astype(str).values,
        "Non-Null Count": df_original.notnull().sum().values
    })

    st.dataframe(type_data, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        df_original.describe(include="all").T,
        use_container_width=True
    )


# --------------------------------------------------
# 8. DATA CLEANING
# --------------------------------------------------

elif page == "Data Cleaning":

    st.title("🧹 Data Cleaning")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before Cleaning")

        st.metric("Rows", len(df_original))

        st.metric(
            "Missing Values",
            int(df_original.isnull().sum().sum())
        )

        st.metric(
            "Duplicate Rows",
            int(df_original.duplicated().sum())
        )

    with col2:
        st.subheader("After Cleaning")

        st.metric("Rows", len(df_clean))

        st.metric(
            "Missing Values",
            int(df_clean.isnull().sum().sum())
        )

        st.metric(
            "Duplicate Rows",
            int(df_clean.duplicated().sum())
        )

    st.subheader("Missing Values by Column")

    missing_data = pd.DataFrame({
        "Column": df_original.columns,
        "Missing Values": df_original.isnull().sum().values,
        "Missing Percentage": (
            df_original.isnull().mean().values * 100
        ).round(2)
    })

    st.dataframe(
        missing_data.sort_values(
            "Missing Values",
            ascending=False
        ),
        use_container_width=True
    )

    st.subheader("Cleaned Dataset")

    st.dataframe(
        df_clean.head(20),
        use_container_width=True
    )

    csv_data = df_clean.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_student_data.csv",
        mime="text/csv"
    )

    st.info(
        "This cleaning step removes exact duplicates and invalid "
        "exam scores. Missing values are handled by the model's "
        "preprocessing pipeline."
    )


# --------------------------------------------------
# 9. EDA VISUALIZATIONS
# --------------------------------------------------

elif page == "EDA Visualizations":

    st.title("📈 Exploratory Data Analysis")

    chart = st.selectbox(
        "Select a chart",
        [
            "Exam Score Distribution",
            "Study Hours vs Exam Score",
            "Attendance vs Exam Score",
            "Scores by Motivation Level",
            "Correlation Heatmap"
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    if chart == "Exam Score Distribution":

        sns.histplot(
            data=df_clean,
            x="Exam_Score",
            bins=20,
            kde=True,
            ax=ax
        )

    elif chart == "Study Hours vs Exam Score":

        sns.scatterplot(
            data=df_clean,
            x="Hours_Studied",
            y="Exam_Score",
            ax=ax
        )

    elif chart == "Attendance vs Exam Score":

        sns.scatterplot(
            data=df_clean,
            x="Attendance",
            y="Exam_Score",
            ax=ax
        )

    elif chart == "Scores by Motivation Level":

        sns.boxplot(
            data=df_clean,
            x="Motivation_Level",
            y="Exam_Score",
            ax=ax
        )

    elif chart == "Correlation Heatmap":

        numeric_data = df_clean.select_dtypes(
            include=np.number
        )

        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

    ax.set_title(chart)

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 10. MAKE PREDICTION
# --------------------------------------------------

elif page == "Make Prediction":

    st.title("🎯 Predict Student Exam Score")

    st.write("Enter student details below.")

    try:

        (
            model,
            features,
            X_test,
            y_test,
            predictions,
            mae,
            rmse,
            r2
        ) = train_model(df_model)

        with st.form("student_form"):

            col1, col2, col3 = st.columns(3)

            with col1:

                hours = st.number_input(
                    "Hours Studied",
                    min_value=0,
                    max_value=24,
                    value=10
                )

                attendance = st.number_input(
                    "Attendance (%)",
                    min_value=0,
                    max_value=100,
                    value=85
                )

                previous = st.number_input(
                    "Previous Scores",
                    min_value=0,
                    max_value=100,
                    value=80
                )

            with col2:

                sleep = st.number_input(
                    "Sleep Hours",
                    min_value=0,
                    max_value=24,
                    value=7
                )

                tutoring = st.number_input(
                    "Tutoring Sessions",
                    min_value=0,
                    max_value=100,
                    value=2
                )

                motivation = st.selectbox(
                    "Motivation Level",
                    ["Low", "Medium", "High"]
                )

            with col3:

                internet = st.selectbox(
                    "Internet Access",
                    ["Yes", "No"]
                )

                resources = st.selectbox(
                    "Access to Resources",
                    ["Low", "Medium", "High"]
                )

                parental = st.selectbox(
                    "Parental Involvement",
                    ["Low", "Medium", "High"]
                )

            submitted = st.form_submit_button("Predict Score")

        if submitted:

            new_student = pd.DataFrame([{
                "Hours_Studied": hours,
                "Attendance": attendance,
                "Previous_Scores": previous,
                "Sleep_Hours": sleep,
                "Tutoring_Sessions": tutoring,
                "Motivation_Level": motivation,
                "Internet_Access": internet,
                "Access_to_Resources": resources,
                "Parental_Involvement": parental
            }])

            predicted_score = model.predict(
                new_student[features]
            )[0]

            st.success("Prediction completed!")

            st.metric(
                "Predicted Exam Score",
                f"{np.clip(predicted_score, 0, 100):.2f} / 100"
            )

            st.caption(
                "This is a model-based estimate, not a guaranteed result."
            )

    except (ValueError, KeyError) as error:

        st.error(str(error))


# --------------------------------------------------
# 11. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    try:

        (
            model,
            features,
            X_test,
            y_test,
            predictions,
            mae,
            rmse,
            r2
        ) = train_model(df_model)

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.3f}")

        st.subheader("Actual vs Predicted Scores")

        results = pd.DataFrame({
            "Actual Score": y_test.to_numpy(),
            "Predicted Score": predictions
        })

        st.dataframe(
            results.head(20),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(y_test, predictions, alpha=0.5)

        minimum = min(
            float(y_test.min()),
            float(np.min(predictions))
        )

        maximum = max(
            float(y_test.max()),
            float(np.max(predictions))
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--"
        )

        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title("Actual vs Predicted Scores")

        st.pyplot(fig)
        plt.close(fig)

        if st.button("Save Trained Model"):

            joblib.dump(model, MODEL_FILE)

            st.success(
                f"Model saved as {MODEL_FILE.name}"
            )

    except (ValueError, KeyError) as error:

        st.error(str(error))


# --------------------------------------------------
# 12. SIDEBAR FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python and Streamlit")