from pandas import DataFrame
from src.Classical.create_docx import create_docx_random_forest
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io

def random_forest(df:DataFrame, output_path:Path, name:str) -> None:
    # Load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le = LabelEncoder()

    # Load
    df = pd.read_csv('/content/drive/MyDrive/shuffled_data2_ToN-IoT.csv', low_memory=False)

    # Convert categorical and object columns to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # --- Get the Target Names (CORRECT) ---
    # le.classes_ contains the array of original string labels in order of their integer encoding
    target_names = le.classes_

    X = df.drop(["src_ip","dst_ip","proto","service","conn_state","missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes","dns_query","dns_qclass","dns_qtype","dns_rcode","dns_AA","dns_RD","dns_RA","dns_rejected","ssl_version","ssl_cipher","ssl_resumed","ssl_established","ssl_subject","ssl_issuer","http_trans_depth","http_method","http_uri","http_version","http_request_body_len","http_response_body_len","http_status_code","http_user_agent","http_orig_mime_types","http_resp_mime_types","weird_name","weird_addl","weird_notice","label","type"], axis=1)
    y = df["type"]

    # --- FIX: Comprehensive handling of non-finite, extreme values, and data types ---
    # Convert all columns that can be numeric to numeric, coercing errors
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace any infinity values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute NaN values with the mean of their respective columns
    # Use numeric_only=True for mean to avoid errors with non-numeric cols if any slipped through
    X = X.fillna(X.mean(numeric_only=True))

    # Cap extreme outliers to prevent 'value too large' issues.
    # This prevents values from becoming np.inf if scikit-learn internally downcasts to float32.
    # Cap to the 99.9th percentile and 0.1st percentile
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            upper_bound = X[col].quantile(0.999)
            lower_bound = X[col].quantile(0.001)
            X[col] = np.clip(X[col], a_min=lower_bound, a_max=upper_bound)

    # Finally, ensure all feature columns are of float64 type for consistency and precision
    X = X.astype(np.float64)

    # Split the data into train and test sets first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- NEW: Apply Random Under-Sampling (RUS) ---
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    # --- END NEW: Apply RUS ---

    # Define and Train Models
    rf_model = RandomForestClassifier(
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=8,
        n_estimators=100,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    )

    # --- FIX: Fit on the resampled training data ---
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred = rf_model.predict(X_test)

    print("\n--- Model Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    # --- FIX: Passing the correct target_names from LabelEncoder ---
    classification_report_str = classification_report(y_test, y_pred, target_names=target_names)
    print(classification_report_str)

    print("\n--- Feature Importance ---")
    feature_importances = rf_model.feature_importances_
    # Use the resampled features for column names consistency
    feature_names = X_train_resampled.columns.tolist()
    # Sort feature importances in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Create a list of feature importance strings for the DOCX file
    feature_importance_list = []
    for i in sorted_indices:
        importance_str = f"{feature_names[i]}: {feature_importances[i]:.4f}"
        print(importance_str)
        feature_importance_list.append(importance_str)


    # --- NEW FUNCTION: Create Feature Importance Plot ---
    def create_importance_plot_bytes(feature_importances, feature_names, n_features=10):
        """Generates a plot of the top feature importance and returns it as bytes."""
        feature_series = pd.Series(feature_importances, index=feature_names)
        top_features = feature_series.nlargest(n_features)

        plt.figure(figsize=(10, 6))
        top_features.sort_values().plot(kind='barh', color='darkgreen')
        plt.title(f'Top {n_features} Random Forest Feature Importance (RUS Trained)')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return buf

    # Generate the plot buffer
    importance_plot_buffer = create_importance_plot_bytes(feature_importances, feature_names, n_features=10)

    create_docx_random_forest(accuracy, classification_report_str, feature_importance_list, importance_plot_buffer)