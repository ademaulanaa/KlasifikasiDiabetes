# app.py
# Streamlit – Diabetes Prediction (berbasis notebook Colab)

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Diabetes Prediction - Haris",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        margin-top: 0px;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🩺 Diabetes Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Portofolio Final Project – Haris Nur K</p>', unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# FUNGSI LOAD DATA (AUTOMATIS)
# =========================================================
DATA_PATH = "Dataset9_Diabetes_Prediction.csv"  # Pastikan file ini ada di folder yang sama

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

# =========================================================
# FUNGSI PREPROCESSING
# =========================================================
def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Kolom yang 0 dianggap missing (sesuai notebook)
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Ganti 0 jadi NaN hanya di kolom tertentu
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

    # Impute median
    for col in zero_as_missing:
        df[col].fillna(df[col].median(), inplace=True)

    # Pisahkan fitur dan target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, df


# =========================================================
# FUNGSI TRAINING MODEL
# =========================================================
def train_models(X_train, y_train, use_smote=True):
    X_tr = X_train
    y_tr = y_train

    if use_smote:
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_train, y_train)

    models = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=200, n_jobs=-1)
    log_reg.fit(X_tr, y_tr)
    models["Logistic Regression"] = log_reg

    # KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_tr, y_tr)
    models["KNN (k=7)"] = knn

    # Random Forest (hyperparameter bisa disesuaikan)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    models["Random Forest"] = rf

    return models


# =========================================================
# FUNGSI EVALUASI MODEL
# =========================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    roc = None
    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "cm": cm,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, labels=("Negatif", "Positif")):
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Prediksi", y="Aktual"),
    )
    fig.update_layout(title="Confusion Matrix", margin=dict(l=40, r=40, t=40, b=40))
    return fig


# =========================================================
# LOAD & PREPROCESS
# =========================================================
try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"File `{DATA_PATH}` tidak ditemukan. Letakkan file CSV di folder yang sama dengan app.py atau ubah variabel DATA_PATH."
    )
    st.stop()

X_train, X_test, y_train, y_test, scaler, df_clean = preprocess_data(raw_df)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙ Pengaturan")
page = st.sidebar.radio(
    "Navigasi",
    ["Data & EDA", "Training & Evaluasi", "Prediksi Pasien Baru"],
    index=0,
)

use_smote = st.sidebar.checkbox("Gunakan SMOTE (imbalance handling)", value=True)
chosen_model_name = st.sidebar.selectbox(
    "Model utama untuk evaluasi & prediksi",
    ["Random Forest", "Logistic Regression", "KNN (k=7)"],
    index=0,
)

# Latih model sekali, bisa dipakai di semua menu
models_global = train_models(X_train, y_train, use_smote=use_smote)
model_global = models_global[chosen_model_name]

# =========================================================
# 1. DATA & EDA
# =========================================================
if page == "Data & EDA":
    st.subheader("📊 Data & Exploratory Data Analysis (EDA)")

    with st.expander("Lihat contoh data (5 baris pertama)", expanded=True):
        st.dataframe(raw_df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Baris", len(raw_df))
    with col2:
        st.metric("Jumlah Fitur", raw_df.shape[1] - 1)
    with col3:
        st.metric("Proporsi Positif Diabetes", f"{raw_df['Outcome'].mean():.2%}")

    st.markdown("### Distribusi Outcome (Positif vs Negatif)")
    outcome_counts = raw_df["Outcome"].value_counts().rename({0: "Negatif", 1: "Positif"})
    fig_pie = px.pie(
        names=outcome_counts.index,
        values=outcome_counts.values,
        title="Distribusi Pasien Diabetes",
        hole=0.3,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Hubungan Fitur Penting dengan Outcome")
    features_important = ["Glucose", "BMI", "Age", "Pregnancies"]

    tab1, tab2 = st.tabs(["Distribusi Fitur", "Korelasi"])

    with tab1:
        feat = st.selectbox("Pilih fitur untuk dilihat distribusinya:", features_important)
        fig_hist = px.histogram(
            raw_df,
            x=feat,
            color=raw_df["Outcome"].map({0: "Negatif", 1: "Positif"}),
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            labels={"color": "Outcome"},
        )
        fig_hist.update_layout(
            title=f"Distribusi {feat} berdasarkan Status Diabetes",
            legend_title_text="Status",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        corr = raw_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# 2. TRAINING & EVALUASI
# =========================================================
elif page == "Training & Evaluasi":
    st.subheader("🤖 Training & Evaluasi Model")

    model = model_global
    eval_result = evaluate_model(model, X_test, y_test)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{eval_result['accuracy']:.3f}")
    col2.metric("Precision", f"{eval_result['precision']:.3f}")
    col3.metric("Recall", f"{eval_result['recall']:.3f}")
    col4.metric("F1-Score", f"{eval_result['f1']:.3f}")
    if eval_result["roc_auc"] is not None:
        col5.metric("ROC-AUC", f"{eval_result['roc_auc']:.3f}")
    else:
        col5.metric("ROC-AUC", "N/A")

    st.markdown("### Confusion Matrix")
    fig_cm = plot_confusion_matrix(eval_result["cm"])
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### Classification Report")
    report = classification_report(y_test, eval_result["y_pred"], target_names=["Negatif", "Positif"])
    st.text(report)

    # Feature importance khusus RandomForest
    if chosen_model_name == "Random Forest":
        st.markdown("### Feature Importance (Random Forest)")
        rf_model: RandomForestClassifier = model
        importances = rf_model.feature_importances_
        feat_names = X_train.columns

        fi_df = pd.DataFrame(
            {"Feature": feat_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.dataframe(fi_df, use_container_width=True)

# =========================================================
# 3. PREDIKSI PASIEN BARU
# =========================================================
else:
    st.subheader("🧬 Prediksi Pasien Baru")

    st.markdown(
        "Isi form di bawah ini dengan data pasien. "
        "Model akan menampilkan **persentase berpotensi** dan **tidak berpotensi** diabetes."
    )

    # Form input
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Pregnancies (Jumlah Kehamilan)", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)

        with col2:
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
            insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)

        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35)
            age = st.number_input("Age", min_value=10, max_value=120, value=35)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Siapkan dataframe 1 baris
        input_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }
        input_df = pd.DataFrame([input_dict])

        # Konsisten dengan preprocessing – ganti 0 jadi median untuk kolom tertentu
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_as_missing:
            if input_df.loc[0, col] == 0:
                input_df.loc[0, col] = df_clean[col].median()

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Pakai model global
        model_pred = model_global

        # Prediksi probabilitas
        if hasattr(model_pred, "predict_proba"):
            probs = model_pred.predict_proba(input_scaled)[0]
            prob_tidak = probs[0]        # kelas 0
            prob_berpotensi = probs[1]   # kelas 1
        else:
            pred_label_tmp = model_pred.predict(input_scaled)[0]
            prob_berpotensi = 1.0 if pred_label_tmp == 1 else 0.0
            prob_tidak = 1.0 - prob_berpotensi

        # Tentukan label utama
        pred_label = 1 if prob_berpotensi >= 0.5 else 0

        # Menampilkan hasil
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Berpotensi Diabetes", f"{prob_berpotensi*100:.2f} %")
        with col_b:
            st.metric("Tidak Berpotensi Diabetes", f"{prob_tidak*100:.2f} %")

        # Teks interpretasi
        if pred_label == 1:
            st.error("📌 Interpretasi: Pasien **BERPOTENSI** mengidap diabetes.")
        else:
            st.success("📌 Interpretasi: Pasien **TIDAK BERpotensi** mengidap diabetes.")

        st.markdown("#### Detail Input")
        st.dataframe(input_df.T, use_container_width=True)

        # 🎈 Tambahan animasi balon saat hasil prediksi muncul
        st.balloons()

