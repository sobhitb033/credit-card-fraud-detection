import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "fraud_scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_RESULTS = {
    "Logistic Regression": {
        "Accuracy": 0.9736721531033025,
        "Precision": 0.053035143769968054,
        "Recall": 0.8736842105263158,
        "F1-Score": 0.10000000000000001
    },
    "Decision Tree": {
        "Accuracy": 0.9974271314277658,
        "Precision": 0.35260115606936415,
        "Recall": 0.6421052631578947,
        "F1-Score": 0.4552238805970149
    },
    "Random Forest": {
        "Accuracy": 0.9994889507630493,
        "Precision": 0.9125,
        "Recall": 0.7684210526315789,
        "F1-Score": 0.8342857142857143
    }
}

CONFUSION_MATRICES = {
    "Logistic Regression": np.array([
        [55169, 1482],
        [12, 83]
    ]),
    "Decision Tree": np.array([
        [56539, 112],
        [34, 61]
    ]),
    "Random Forest": np.array([
        [56644, 7],
        [22, 73]
    ])
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    return df

model = load_model()
scaler = load_scaler()
df = load_data()

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "fraud_probability" not in st.session_state:
    st.session_state.fraud_probability = 0.0

if "legitimate_probability" not in st.session_state:
    st.session_state.legitimate_probability = 100.0

for feature in FEATURES:
    key = f"input_{feature}"

    if key not in st.session_state:
        st.session_state[key] = 0.0

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #aaaaaa;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 25px;
        font-weight: 700;
    }

    .fraud-box {
        background-color: #3b1111;
        border: 1px solid #ff4b4b;
        color: #ff6b6b;
    }

    .legitimate-box {
        background-color: #0d3b27;
        border: 1px solid #21c77a;
        color: #36e68b;
    }

    .risk-low {
        background-color: #103d29;
        color: #43e68d;
        padding: 18px;
        border-radius: 10px;
        font-size: 17px;
    }

    .risk-medium {
        background-color: #493c0d;
        color: #ffd43b;
        padding: 18px;
        border-radius: 10px;
        font-size: 17px;
    }

    .risk-high {
        background-color: #451515;
        color: #ff6b6b;
        padding: 18px;
        border-radius: 10px;
        font-size: 17px;
    }

    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #151821;
        border: 1px solid #30343d;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #888888;
        padding: 30px 0;
        font-size: 14px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        min-height: 45px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based detection of potentially fraudulent credit card transactions</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown(
    '<div class="section-title">⚡ Quick Demo</div>',
    unsafe_allow_html=True
)

st.write(
    "Load a real transaction from the dataset to demonstrate the trained Random Forest model."
)

def load_transaction(row):
    for feature in FEATURES:
        st.session_state[f"input_{feature}"] = float(row[feature])

    st.session_state.prediction = None
    st.session_state.fraud_probability = 0.0
    st.session_state.legitimate_probability = 100.0

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(
        "🟢 Load Legitimate Transaction",
        use_container_width=True
    ):
        legitimate_rows = df[df["Class"] == 0]

        if len(legitimate_rows) > 0:
            load_transaction(legitimate_rows.iloc[0])
            st.rerun()

with col2:
    if st.button(
        "🔴 Load Fraudulent Transaction",
        use_container_width=True
    ):
        fraud_rows = df[df["Class"] == 1]

        if len(fraud_rows) > 0:
            load_transaction(fraud_rows.iloc[0])
            st.rerun()

with col3:
    if st.button(
        "🔄 Clear Sample",
        use_container_width=True
    ):
        for feature in FEATURES:
            st.session_state[f"input_{feature}"] = 0.0

        st.session_state.prediction = None
        st.session_state.fraud_probability = 0.0
        st.session_state.legitimate_probability = 100.0

        st.rerun()

st.divider()

st.markdown(
    '<div class="section-title">💳 Transaction Details</div>',
    unsafe_allow_html=True
)

input_values = {}

col1, col2, col3 = st.columns(3)

with col1:
    input_values["Time"] = st.number_input(
        "Transaction Time",
        key="input_Time",
        format="%.2f"
    )

    for i in range(1, 10):
        feature = f"V{i}"

        input_values[feature] = st.number_input(
            feature,
            key=f"input_{feature}",
            format="%.6f"
        )

with col2:
    for i in range(10, 20):
        feature = f"V{i}"

        input_values[feature] = st.number_input(
            feature,
            key=f"input_{feature}",
            format="%.6f"
        )

with col3:
    for i in range(20, 29):
        feature = f"V{i}"

        input_values[feature] = st.number_input(
            feature,
            key=f"input_{feature}",
            format="%.6f"
        )

    input_values["Amount"] = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        key="input_Amount",
        format="%.2f"
    )

if st.button(
    "🔍 Check Transaction",
    use_container_width=True
):
    transaction = np.array(
        [[input_values[feature] for feature in FEATURES]]
    )

    transaction_scaled = scaler.transform(transaction)

    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0]

    st.session_state.prediction = int(prediction)
    st.session_state.legitimate_probability = float(probability[0] * 100)
    st.session_state.fraud_probability = float(probability[1] * 100)

st.divider()

st.markdown(
    '<div class="section-title">📊 Prediction Result</div>',
    unsafe_allow_html=True
)

if st.session_state.prediction is None:

    st.info(
        "Enter transaction details or load a sample transaction to get a prediction."
    )

else:

    prediction = st.session_state.prediction
    fraud_probability = st.session_state.fraud_probability
    legitimate_probability = st.session_state.legitimate_probability

    if prediction == 1:
        st.markdown(
            '<div class="result-box fraud-box">🚨 FRAUDULENT TRANSACTION</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box legitimate-box">✅ LEGITIMATE TRANSACTION</div>',
            unsafe_allow_html=True
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

    st.markdown(
        '<div class="section-title">Risk Analysis</div>',
        unsafe_allow_html=True
    )

    st.progress(
        min(max(fraud_probability / 100, 0.0), 1.0)
    )

    if fraud_probability < 20:

        st.markdown(
            '<div class="risk-low">🟢 Low fraud risk — the model considers this transaction likely legitimate.</div>',
            unsafe_allow_html=True
        )

    elif fraud_probability < 60:

        st.markdown(
            '<div class="risk-medium">🟡 Medium fraud risk — the transaction should be reviewed.</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="risk-high">🔴 High fraud risk — the model considers this transaction potentially fraudulent.</div>',
            unsafe_allow_html=True
        )

st.divider()

st.markdown(
    '<div class="section-title">📈 Dataset Overview</div>',
    unsafe_allow_html=True
)

total_transactions = len(df)
legitimate_count = int((df["Class"] == 0).sum())
fraud_count = int((df["Class"] == 1).sum())
fraud_percentage = fraud_count / total_transactions * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Legitimate",
        f"{legitimate_count:,}"
    )

with col3:
    st.metric(
        "Fraudulent",
        f"{fraud_count:,}"
    )

with col4:
    st.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.3f}%"
    )

st.write("")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        ["Legitimate", "Fraudulent"],
        [legitimate_count, fraud_count]
    )

    ax.set_title("Transaction Class Distribution")
    ax.set_ylabel("Number of Transactions")

    st.pyplot(fig)

with chart_col2:

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pie(
        [legitimate_count, fraud_count],
        labels=["Legitimate", "Fraudulent"],
        autopct="%1.2f%%"
    )

    ax.set_title("Fraud vs Legitimate Transactions")

    st.pyplot(fig)

st.divider()

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

performance_df = pd.DataFrame(MODEL_RESULTS).T

display_df = performance_df.copy()

for column in display_df.columns:
    display_df[column] = (
        display_df[column] * 100
    ).round(2)

display_df.columns = [
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1-Score (%)"
]

st.dataframe(
    display_df,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 5))

models = list(MODEL_RESULTS.keys())

x = np.arange(len(models))
width = 0.2

accuracy = [
    MODEL_RESULTS[m]["Accuracy"]
    for m in models
]

precision = [
    MODEL_RESULTS[m]["Precision"]
    for m in models
]

recall = [
    MODEL_RESULTS[m]["Recall"]
    for m in models
]

f1 = [
    MODEL_RESULTS[m]["F1-Score"]
    for m in models
]

ax.bar(x - width * 1.5, accuracy, width, label="Accuracy")
ax.bar(x - width / 2, precision, width, label="Precision")
ax.bar(x + width / 2, recall, width, label="Recall")
ax.bar(x + width * 1.5, f1, width, label="F1-Score")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Comparison of Fraud Detection Models")
ax.legend()

st.pyplot(fig)

st.divider()

st.markdown(
    '<div class="section-title">🔎 Confusion Matrix Analysis</div>',
    unsafe_allow_html=True
)

selected_model = st.selectbox(
    "Select Model",
    list(CONFUSION_MATRICES.keys())
)

cm = CONFUSION_MATRICES[selected_model]

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm)

ax.set_title(
    f"{selected_model} Confusion Matrix"
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_yticklabels(["Legitimate", "Fraud"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center"
        )

fig.colorbar(im, ax=ax)

st.pyplot(fig)

tn, fp, fn, tp = cm.ravel()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("True Negatives", f"{tn:,}")

with col2:
    st.metric("False Positives", f"{fp:,}")

with col3:
    st.metric("False Negatives", f"{fn:,}")

with col4:
    st.metric("True Positives", f"{tp:,}")

st.divider()

st.markdown(
    '<div class="section-title">⚙️ How the System Works</div>',
    unsafe_allow_html=True
)

steps = [
    (
        "1️⃣ Data Collection",
        "Credit card transaction data containing 30 input features and a fraud class label is used."
    ),
    (
        "2️⃣ Data Preprocessing",
        "Duplicate transactions are removed and numerical features are standardized using a trained scaler."
    ),
    (
        "3️⃣ Class Imbalance Handling",
        "SMOTE is applied to the training data because fraudulent transactions represent a very small portion of the dataset."
    ),
    (
        "4️⃣ Model Training",
        "Logistic Regression, Decision Tree and Random Forest models are trained and evaluated."
    ),
    (
        "5️⃣ Model Selection",
        "Random Forest is used by the application for transaction prediction."
    ),
    (
        "6️⃣ Fraud Prediction",
        "The entered transaction is transformed using the same scaler and passed to the trained Random Forest model."
    ),
    (
        "7️⃣ Risk Analysis",
        "The model's fraud probability is displayed along with a corresponding risk level."
    )
]

for title, description in steps:
    st.markdown(
        f"""
        <div class="info-box">
        <strong>{title}</strong><br>
        {description}
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

st.markdown(
    '<div class="section-title">📌 Project Summary</div>',
    unsafe_allow_html=True
)

st.write(
    """
    This project implements a machine learning based credit card fraud
    detection system. The dataset was analyzed using exploratory data
    analysis, duplicate detection, transaction amount analysis,
    transaction time analysis, correlation analysis and class distribution
    analysis.

    Because fraudulent transactions represent a very small proportion of
    the dataset, SMOTE was applied to the training data to address class
    imbalance.

    Three machine learning algorithms were evaluated: Logistic Regression,
    Decision Tree and Random Forest.

    The trained Random Forest model achieved approximately 99.95% accuracy,
    91.25% precision, 76.84% recall and 83.43% F1-score on the test dataset.
    """
)

st.markdown(
    """
    <div class="footer">
    Credit Card Fraud Detection System<br>
    Machine Learning Project | Python | Scikit-learn | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)