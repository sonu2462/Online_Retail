import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv(r"C:\Users\sonal\Downloads\Sonali ML Fnl\Sonali ML\online_shoppers_intention.csv")
    
    target_column = 'Revenue'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    if y.dtype == 'object':
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)
        label_encoders[target_column] = target_le
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, y, label_encoders, scaler, X.columns.tolist()

# 2. Train model
@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred

# 3. Main app logic with navigation
def main():
    st.set_page_config(page_title="Shopper Intention ML App", layout="wide")
    st.title("🛒 Online Shoppers Intention Prediction")

    df, X_scaled, y, label_encoders, scaler, feature_names = load_and_process_data()
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X_scaled, y)

    # --- Sidebar navigation ---
    menu = ["Home", "Data Preview", "Visualizations", "Model Evaluation", "Predict"]
    choice = st.sidebar.selectbox("Navigate", menu)

    if choice == "Home":
        st.subheader("Welcome to the Shopper Intention Prediction App")
        st.markdown("""
        This app uses machine learning to predict whether an online shopper will make a purchase.
        Navigate using the sidebar to explore the dataset, visualize insights, evaluate the model, and make predictions.
        """)

    elif choice == "Data Preview":
        st.subheader("📊 Data Preview")
        st.write(df.head(20))

        st.markdown("### Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    elif choice == "Visualizations":
        st.subheader("📈 Visual Explorations")

        # Revenue Distribution
        st.markdown("#### Revenue Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Revenue', ax=ax1)
        st.pyplot(fig1)

        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        st.pyplot(fig2)

        # Feature Importance
        st.markdown("#### Feature Importance")
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=feat_imp_df, x='Importance', y='Feature', ax=ax3)
        st.pyplot(fig3)

    elif choice == "Model Evaluation":
        st.subheader("🧪 Model Evaluation")

        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Classification Report
        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            'precision': "{:.2f}",
            'recall': "{:.2f}",
            'f1-score': "{:.2f}",
            'support': "{:.0f}"
        }))

    elif choice == "Predict":
        st.subheader("🎯 Make a Prediction")

        user_input = {}
        for feature in feature_names:
            if feature in label_encoders:
                classes = label_encoders[feature].classes_
                user_input[feature] = st.selectbox(f"{feature}", classes)
            else:
                user_input[feature] = st.number_input(
                    f"{feature}",
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].mean())
                )

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            for feature in label_encoders:
                if feature in input_df.columns:
                    le = label_encoders[feature]
                    input_df[feature] = le.transform(input_df[feature])

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            if 'Revenue' in label_encoders:
                prediction_label = label_encoders['Revenue'].inverse_transform([prediction])[0]
            else:
                prediction_label = prediction

            st.success(f"🧾 Prediction: {prediction_label}")

if __name__ == "__main__":
    main()
