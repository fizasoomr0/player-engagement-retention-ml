
# =========================================================
# PLAYER ENGAGEMENT & RETENTION INTELLIGENCE SYSTEM
# Streamlit Application
# =========================================================

# -----------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -----------------------------

# json is used to read saved model evaluation metrics.
import json

# pathlib makes file paths work correctly even when Streamlit
# is launched from a different working directory.
from pathlib import Path

# joblib loads the already-trained machine learning pipeline.
import joblib

# matplotlib is used for charts.
import matplotlib.pyplot as plt

# pandas is used for data handling and CSV files.
import pandas as pd

# streamlit creates the web application.
import streamlit as st


# --------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------

# Configure the browser tab and use a wide professional layout.
st.set_page_config(
    page_title="Player Engagement Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------
# 3. PROJECT FILE PATHS
# --------------------------------

# Get the folder where this APP.py file is located.
# This prevents "file not found" errors caused by running
# Streamlit from another folder.
BASE_DIR = Path(__file__).resolve().parent

# Define all project files relative to APP.py.
MODEL_PATH = BASE_DIR / "models" / "player_engagement_xgboost.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
DATA_PATH = BASE_DIR / "online_gaming_behavior_dataset.csv"


# --------------------------------
# 4. PROFESSIONAL CUSTOM THEME
# --------------------------------

# Use navy + wine + blue + white as the main visual system.
st.markdown(
    """
    <style>

    /* Main application background and text */
    .stApp {
        background: #F8FAFC;
        color: #0F172A;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700;
    }

    /* Small subtitle / normal text */
    p, label, .stMarkdown {
        color: #334155;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0F172A 0%, #172554 65%, #7F1D1D 100%);
        border-radius: 18px;
        padding: 30px 34px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15);
    }

    .hero h1 {
        color: white !important;
        margin: 0 0 8px 0;
        font-size: 2.25rem;
    }

    .hero p {
        color: #E2E8F0 !important;
        margin: 0;
        font-size: 1rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        transition: all 0.25s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #2563EB;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
    }

    [data-testid="stMetricValue"] {
        color: #7F1D1D !important;
        font-weight: 800;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: #2563EB;
        color: white !important;
        border: 1px solid #2563EB;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #1D4ED8;
        border-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.25);
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: scale(0.98);
    }

    /* Input fields */
    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Prediction result */
    .prediction-box {
        background: white;
        border-left: 6px solid #7F1D1D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
    }

    .prediction-label {
        color: #64748B;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }

    .prediction-value {
        color: #7F1D1D;
        font-size: 1.7rem;
        font-weight: 800;
    }

    /* Section cards */
    .section-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0 20px 0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    /* Footer */
    .footer {
        margin-top: 45px;
        padding: 20px 0 5px 0;
        border-top: 1px solid #CBD5E1;
        text-align: center;
        color: #64748B;
        font-size: 0.88rem;
    }

    /* Active sidebar radio option gets a little more visual weight. */
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #93C5FD !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------
# 5. SAFE RESOURCE LOADERS
# --------------------------------

@st.cache_resource
def load_model():
    """Load the trained ML pipeline once and reuse it."""

    # Check that the model actually exists before trying to load it.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    # Load the complete trained pipeline.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Load saved evaluation metrics."""

    # Check that the JSON file exists.
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file was not found:\n{METRICS_PATH}"
        )

    # Read the JSON file.
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_data():
    """Load the original gaming dataset."""

    # Check that the dataset exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file was not found:\n{DATA_PATH}"
        )

    # Read the CSV dataset.
    return pd.read_csv(DATA_PATH)


# Load all project resources.
# A clear error is displayed instead of showing a long traceback
# if one of the required files is missing.
try:
    model = load_model()
    metrics = load_metrics()
    df_app = load_data()
except Exception as error:
    st.error("The application could not load the required project files.")
    st.code(str(error))
    st.info(
        "Make sure APP.py, model_metrics.json, "
        "online_gaming_behavior_dataset.csv, and the models folder "
        "are in the correct project folder."
    )
    st.stop()


# --------------------------------
# 6. MODEL INPUT FEATURES
# --------------------------------

# These must match the features used during model training.
required_features = [
    "Age",
    "Gender",
    "Location",
    "GameGenre",
    "PlayTimeHours",
    "InGamePurchases",
    "GameDifficulty",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked"
]


# --------------------------------
# 7. ENGAGEMENT LABEL MAPPING
# --------------------------------

# The trained model was created with:
# Low = 0, Medium = 1, High = 2.
label_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}


def readable_prediction(prediction):
    """
    Convert a model prediction into a readable engagement label.

    This function also handles a model that already returns a string.
    """

    # If the model already returns Low/Medium/High, use it directly.
    if isinstance(prediction, str):
        return prediction

    # Otherwise convert the numeric class to its readable label.
    return label_mapping.get(
        int(prediction),
        str(prediction)
    )


# --------------------------------
# 8. SIDEBAR
# --------------------------------

# Sidebar branding.
st.sidebar.markdown(
    """
    <div style="
        padding: 10px 0 18px 0;
        border-bottom: 1px solid #334155;
        margin-bottom: 20px;
    ">
        <div style="
            font-size: 1.45rem;
            font-weight: 800;
            color: white;
        ">
            FIZA FATIMA
        </div>
        <div style="
            color: #93C5FD;
            font-size: 0.82rem;
            margin-top: 4px;
        ">
            ML Intelligence System
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Short application description.
st.sidebar.write(
    "Player Engagement & Retention Prediction System"
)

# Navigation menu.
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Dashboard",
        "Predict",
        "Batch Prediction",
        "Model Insights"
    ]
)


# --------------------------------
# 9. HOME PAGE
# --------------------------------

if page == "Home":

    # Professional hero section.
    st.markdown(
        """
        <div class="hero">
            <h1>🎮 Player Engagement & Retention Intelligence</h1>
            <p>
                An end-to-end machine learning system for predicting
                player engagement and supporting data-driven gaming decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Performance section.
    st.subheader("Model Performance")

    # Four metric cards.
    col1, col2, col3, col4 = st.columns(4)

    # Dataset size.
    col1.metric(
        "Dataset Records",
        f"{len(df_app):,}"
    )

    # Number of model input features.
    col2.metric(
        "Input Features",
        len(required_features)
    )

    # Test accuracy loaded from model_metrics.json.
    col3.metric(
        "Test Accuracy",
        f"{float(metrics['accuracy']) * 100:.2f}%"
    )

    # F1-Macro loaded from model_metrics.json.
    col4.metric(
        "F1-Macro",
        f"{float(metrics['f1_macro']) * 100:.2f}%"
    )

    # Business problem.
    st.markdown(
        """
        <div class="section-card">
            <h3>Business Problem</h3>
            <p>
                Gaming platforms need to understand player engagement
                to identify different engagement levels and support
                data-driven retention strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ML solution.
    st.markdown(
        """
        <div class="section-card">
            <h3>Machine Learning Solution</h3>
            <p>
                The system uses player behavioral and demographic
                information to predict Low, Medium, or High engagement.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Project workflow.
    st.markdown(
        """
        <div class="section-card">
            <h3>Project Workflow</h3>
            <p>
                Data Collection → EDA → Preprocessing → Model Training →
                Hyperparameter Tuning → Evaluation → Prediction →
                Business Insights
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------
# 10. DASHBOARD PAGE
# --------------------------------

elif page == "Dashboard":

    st.title("📊 Player Behavior Dashboard")

    st.write(
        "Explore player behavior and engagement patterns "
        "using the available gaming dataset."
    )

    # Dataset overview metrics.
    col1, col2 = st.columns(2)

    col1.metric(
        "Total Players",
        f"{len(df_app):,}"
    )

    col2.metric(
        "Total Columns",
        len(df_app.columns)
    )

    st.subheader("Dataset Preview")

    # Show a small preview rather than the complete dataset.
    st.dataframe(
        df_app.head(10),
        use_container_width=True
    )

    st.subheader("Engagement Level Distribution")

    # Count Low, Medium, and High players.
    engagement_counts = df_app["EngagementLevel"].value_counts()

    # Create a chart.
    fig, ax = plt.subplots(figsize=(8, 4))

    engagement_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Engagement Level")
    ax.set_ylabel("Number of Players")
    ax.set_title("Player Engagement Distribution")

    # Keep the chart compact and readable.
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Display the chart.
    st.pyplot(fig)

    # Close the matplotlib figure after rendering.
    plt.close(fig)


# --------------------------------
# 11. SINGLE PREDICTION PAGE
# --------------------------------

elif page == "Predict":

    st.title("🔮 Predict Player Engagement")

    st.write(
        "Enter player information below and generate an engagement prediction."
    )

    # Three input columns.
    col1, col2, col3 = st.columns(3)

    # -----------------------------
    # Basic player information
    # -----------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=25,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            sorted(df_app["Gender"].dropna().unique().tolist())
        )

        location = st.selectbox(
            "Location",
            sorted(df_app["Location"].dropna().unique().tolist())
        )

        game_genre = st.selectbox(
            "Game Genre",
            sorted(df_app["GameGenre"].dropna().unique().tolist())
        )

    # -----------------------------
    # Game behavior information
    # -----------------------------

    with col2:

        play_time = st.number_input(
            "Play Time Hours",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

        in_game_purchases = st.number_input(
            "In-Game Purchases",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        game_difficulty = st.selectbox(
            "Game Difficulty",
            sorted(df_app["GameDifficulty"].dropna().unique().tolist())
        )

        sessions_per_week = st.number_input(
            "Sessions Per Week",
            min_value=0,
            value=5,
            step=1
        )

    # -----------------------------
    # Progress information
    # -----------------------------

    with col3:

        avg_session_duration = st.number_input(
            "Average Session Duration (Minutes)",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

        player_level = st.number_input(
            "Player Level",
            min_value=0,
            value=10,
            step=1
        )

        achievements = st.number_input(
            "Achievements Unlocked",
            min_value=0,
            value=5,
            step=1
        )

    # Prediction button.
    if st.button(
        "Predict Engagement",
        use_container_width=True
    ):

        # Build a one-row DataFrame using exactly the same
        # feature names expected by the trained model.
        input_data = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Location": location,
                    "GameGenre": game_genre,
                    "PlayTimeHours": play_time,
                    "InGamePurchases": in_game_purchases,
                    "GameDifficulty": game_difficulty,
                    "SessionsPerWeek": sessions_per_week,
                    "AvgSessionDurationMinutes": avg_session_duration,
                    "PlayerLevel": player_level,
                    "AchievementsUnlocked": achievements
                }
            ],
            columns=required_features
        )

        try:

            # Generate the class prediction.
            prediction = model.predict(input_data)[0]

            # Convert numeric output into Low/Medium/High.
            predicted_label = readable_prediction(prediction)

            # Display the prediction.
            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-label">
                        Predicted Engagement Level
                    </div>
                    <div class="prediction-value">
                        {predicted_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # If the model supports probability prediction,
            # also display the confidence for the selected class.
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                # Get the class order from the pipeline/model.
                if hasattr(model, "classes_"):
                    classes = model.classes_
                else:
                    classes = list(range(len(probabilities)))

                # Find the probability belonging to the prediction.
                confidence = None

                for class_value, probability in zip(
                    classes,
                    probabilities
                ):
                    if str(class_value) == str(prediction):
                        confidence = float(probability)
                        break

                if confidence is not None:
                    st.metric(
                        "Prediction Confidence",
                        f"{confidence * 100:.2f}%"
                    )

        except Exception as error:

            # Show a clear message without crashing the whole app.
            st.error("Prediction could not be completed.")

            # Technical details help with debugging if needed.
            st.code(str(error))


# --------------------------------
# 12. BATCH PREDICTION PAGE
# --------------------------------

elif page == "Batch Prediction":

    st.title("📁 Batch Player Prediction")

    st.write(
        "Upload a CSV file containing player information "
        "to generate predictions for multiple players."
    )

    st.subheader("Required CSV Columns")

    # Display the exact required feature names.
    st.code(", ".join(required_features))

    # Create an empty CSV template.
    template_df = pd.DataFrame(
        columns=required_features
    )

    # Convert template to CSV format.
    template_csv = template_df.to_csv(
        index=False
    )

    # Allow the user to download the template.
    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_csv,
        file_name="player_prediction_template.csv",
        mime="text/csv"
    )

    # Upload CSV file.
    uploaded_file = st.file_uploader(
        "Upload Player CSV",
        type=["csv"]
    )

    # Continue only when a file has been uploaded.
    if uploaded_file is not None:

        try:

            # Read the uploaded CSV.
            batch_df = pd.read_csv(uploaded_file)

        except Exception as error:

            st.error("The uploaded CSV could not be read.")
            st.code(str(error))
            st.stop()

        st.subheader("Uploaded Data")

        # Display uploaded records.
        st.dataframe(
            batch_df,
            use_container_width=True
        )

        # Find missing required columns.
        missing_columns = [
            column
            for column in required_features
            if column not in batch_df.columns
        ]

        # Stop if required columns are missing.
        if missing_columns:

            st.error(
                "The uploaded CSV is missing required columns."
            )

            st.write("Missing columns:")
            st.write(missing_columns)

            st.stop()

        # Select only the columns required by the model.
        # This also enforces the exact training column order.
        prediction_df = batch_df[
            required_features
        ].copy()

        # Count all missing values in prediction columns.
        missing_values = (
            prediction_df
            .isnull()
            .sum()
            .sum()
        )

        # Stop if missing values exist.
        if missing_values > 0:

            st.error(
                "The uploaded CSV contains missing values "
                "in required prediction columns."
            )

            st.dataframe(
                prediction_df.isnull().sum().to_frame("Missing Values")
            )

            st.stop()

        try:

            # Generate predictions for every uploaded row.
            batch_predictions = model.predict(
                prediction_df
            )

            # Convert model outputs to readable labels.
            readable_predictions = [
                readable_prediction(prediction)
                for prediction in batch_predictions
            ]

            # Add predictions to the uploaded dataset.
            result_df = batch_df.copy()

            result_df["PredictedEngagement"] = (
                readable_predictions
            )

            st.subheader("Prediction Results")

            # Display results.
            st.dataframe(
                result_df,
                use_container_width=True
            )

            # Convert results into downloadable CSV.
            csv_output = result_df.to_csv(
                index=False
            )

            # Provide results download.
            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_output,
                file_name="player_engagement_predictions.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(
                "Prediction could not be completed for "
                "the uploaded file."
            )

            st.code(str(error))


# --------------------------------
# 13. MODEL INSIGHTS PAGE
# --------------------------------

elif page == "Model Insights":

    st.title("🧠 Model Insights")

    st.write(
        "Performance and feature importance of the trained "
        "XGBoost engagement prediction model."
    )

    st.subheader("Model Performance")

    # Three performance metrics.
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{float(metrics['accuracy']) * 100:.2f}%"
    )

    col2.metric(
        "F1-Macro",
        f"{float(metrics['f1_macro']) * 100:.2f}%"
    )

    col3.metric(
        "ROC-AUC",
        f"{float(metrics['roc_auc']) * 100:.2f}%"
    )

    st.subheader("Top Features Driving Predictions")

    try:

        # Access the XGBoost model from the trained pipeline.
        xgb_model = model.named_steps["model"]

        # Access the preprocessing stage.
        preprocessor = model.named_steps["preprocessor"]

        # Get transformed feature names after encoding.
        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        # Get XGBoost feature importance values.
        importances = (
            xgb_model.feature_importances_
        )

        # Make sure the number of names matches the
        # number of importance values.
        if len(feature_names) != len(importances):
            raise ValueError(
                "The number of transformed feature names does not "
                "match the number of model feature importances."
            )

        # Create a feature importance table.
        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )

        # Remove preprocessing prefixes for readability.
        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace(
                "num__",
                "",
                regex=False
            )
            .str.replace(
                "cat__",
                "",
                regex=False
            )
        )

        # Sort from most important to least important.
        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(10)
        )

        # Create horizontal feature importance chart.
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        importance_df.sort_values(
            "Importance"
        ).plot(
            x="Feature",
            y="Importance",
            kind="barh",
            ax=ax,
            legend=False
        )

        ax.set_title(
            "Top 10 Feature Importances"
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_ylabel(
            "Feature"
        )

        plt.tight_layout()

        # Display chart.
        st.pyplot(fig)

        # Close the figure to avoid duplicate/unused
        # matplotlib objects during Streamlit reruns.
        plt.close(fig)

        # Display the importance table.
        st.dataframe(
            importance_df,
            use_container_width=True
        )

        st.subheader("Business Interpretation")

        st.write(
            "The trained model relies heavily on player activity "
            "features when determining engagement level. "
            "Sessions per week and average session duration are "
            "particularly important indicators in the trained model."
        )

        st.info(
            "Feature importance shows how strongly the trained model "
            "relies on a feature for prediction. It does not by itself "
            "prove that the feature causes player engagement."
        )

    except Exception as error:

        st.error(
            "Feature importance could not be displayed."
        )

        st.code(str(error))


# --------------------------------
# 14. FOOTER
# --------------------------------

# Display project ownership and copyright information.
st.markdown(
    """
    <div class="footer">
        © 2026 Fiza Fatima — All Rights Reserved.<br>
        Player Engagement & Retention Intelligence System
    </div>
    """,
    unsafe_allow_html=True
)
