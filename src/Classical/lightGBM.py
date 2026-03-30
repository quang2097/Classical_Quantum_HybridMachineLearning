#LIGHTGBM
import pandas as pd
from src.Classical import create_docx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from docx import Document
from docx.shared import Inches
import numpy as np
import io
import matplotlib.pyplot as plt
# --- NEW: Import for ROC Curve calculation ---
from sklearn.metrics import roc_curve, auc
# ---------------------------------------------

# --- FIX: Change the import order ---
import torch
import lightgbm as lgb

from pandas import DataFrame
from pathlib import Path

def lightGBM(df:DataFrame, output_path:Path, name:str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  le = LabelEncoder()

  # --- 1. Load Data and Initial Preprocessing ---
  # Load
  df = pd.read_csv('/content/drive/MyDrive/shuffled_data2_ToN-IoT.csv', low_memory=False)

  # Convert categorical/object columns to numeric using LabelEncoder
  for col in df.select_dtypes(include=['object']).columns:
    # Use LabelEncoder on the target 'type' column and other object columns
    df[col] = le.fit_transform(df[col])

  # --- 2. Feature and Target Definition ---
  # le.classes_ contains the array of original string labels
  target_names = le.classes_

  # Define the columns to drop (features that are usually low-variance, IDs, or duplicates)
  columns_to_drop = [
    "src_ip","dst_ip","proto","service","conn_state",
    "missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes",
    "dns_query","dns_qclass","dns_qtype","dns_rcode","dns_AA",
    "dns_RD","dns_RA","dns_rejected","ssl_version","ssl_cipher",
    "ssl_resumed","ssl_established","ssl_subject","ssl_issuer","http_trans_depth",
    "http_method","http_uri","http_version","http_request_body_len","http_response_body_len",
    "http_status_code","http_user_agent","http_orig_mime_types","http_resp_mime_types","weird_name",
    "weird_addl","weird_notice","label","type"
  ]
  target_column = "type"

  # Filter the list to only include columns that are actually in the DataFrame
  existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

  # Separate features (X) and target (y)
  X = df.drop(existing_cols_to_drop + [target_column], axis=1, errors='ignore')
  y = df[target_column]

  # --- 3. Split Data ---
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55, random_state=42)

  # --- NEW: Apply Random Under-Sampling (RUS) ---
  rus = RandomUnderSampler(random_state=42)
  X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
  # --- END NEW: Apply RUS ---

  # --- 4. LightGBM Setup and Training ---
  num_classes = len(y_train_resampled.unique()) # Use resampled data to define classes
  lgb_params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
  }

  # --- FIX: Use resampled data for training ---
  lgb_train = lgb.Dataset(X_train_resampled, y_train_resampled)
  lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

  # Dictionary to store evaluation results at each boosting round
  evals_result = {}

  gbm = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_eval],
    # Capture history using the evals_result dict and early stopping
    callbacks=[lgb.record_evaluation(evals_result), lgb.early_stopping(10, verbose=False)]
  )

  # --- NEW: Extract Epoch/Round Report ---
  training_log_str = "Round | Train Log Loss | Eval Log Loss\n"
  training_log_str += "------|----------------|---------------\n"

  # Get the list of log loss values for training and evaluation sets
  train_losses = evals_result['training']['multi_logloss']
  eval_losses = evals_result['valid_1']['multi_logloss']
  best_round = gbm.best_iteration

  # Combine and format the data for the report, up to the best round
  for i in range(best_round):
      round_number = f" {i + 1:<4}"
      train_loss = f" {train_losses[i]:<14.6f}"
      eval_loss = f" {eval_losses[i]:<13.6f}"

      # Highlight the final round before early stopping
      if i == best_round - 1:
        round_number = f"*{i + 1:<3}" # Add asterisk for emphasis

      training_log_str += f"{round_number}|{train_loss}|{eval_loss}\n"

  training_log_str += f"\nModel stopped after {best_round} rounds (best score was at round {best_round})."
  # --- End NEW: Extract Epoch/Round Report ---


  # --- 5. Prediction ---
  y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
  y_pred = y_pred_proba.argmax(axis=1)


  # Clean up y_test for consistent evaluation
  y_test_clean = y_test.values.flatten().astype(int)
  y_pred_clean = y_pred.astype(int)


  # --- 6. Evaluate the Model ---
  print("\n--- Model Evaluation ---")

  # Calculate AUC (Area Under the ROC Curve)
  auc_score = roc_auc_score(y_test_clean, y_pred_proba, multi_class='ovr', average='macro')
  print(f"Test AUC Score: {auc_score:.4f}")

  # Calculate Accuracy
  accuracy = accuracy_score(y_test_clean, y_pred_clean)
  print(f"Test Accuracy: {accuracy:.4f}")

  # --- 7. Feature Importance and Classification Report ---
  print("\n--- Feature Importance ---")
  # Use the resampled features for column names consistency in feature importance plot/report
  feature_importance = pd.Series(gbm.feature_importance(), index=X_train_resampled.columns).sort_values(ascending=False)
  print(feature_importance.head())
  # Convert top 5 feature importance to a list of strings for DOCX
  top_feature_importance_list = [f"{idx}: {val}" for idx, val in feature_importance.head().items()]

  print("--- Classification Report ---")
  # Use target_names argument to print original string labels in the report
  report = classification_report(y_test_clean, y_pred_clean, target_names=target_names)
  print(report)
  classification_report_str = report

  # --- 8. Create and Save Feature Importance Plot ---
  def create_importance_plot_bytes(feature_importance, n_features=10):
    """Generates a plot of the top feature importance and returns it as bytes."""
    top_features = feature_importance.head(n_features)

    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind='barh', color='skyblue')
    plt.title(f'Top {n_features} LightGBM Feature Importance (RUS Trained)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

  # Generate the plot buffer
  importance_plot_buffer = create_importance_plot_bytes(feature_importance, n_features=10)


  # --- NEW: ROC Curve Plotting Function ---
  def create_roc_plot_bytes(y_test_clean, y_pred_proba, target_names):
    """
    Generates a multi-class One-vs-Rest ROC curve plot and returns it as bytes.
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = len(target_names)

    # Iterate through each class (0 to n_classes-1)
    for i in range(n_classes):
      # Binarize the true labels (1 if the label is i, 0 otherwise)
      y_true_binary = (y_test_clean == i).astype(int)

      fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_proba[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting setup
    plt.figure(figsize=(10, 8))
    lw = 2

    # Plot each class ROC curve
    for i in range(n_classes):
      plt.plot(
        fpr[i],
        tpr[i],
        lw=lw,
        label=f"ROC curve of class {target_names[i]} (AUC = {roc_auc[i]:0.2f})"
      )

    # Plot the random guess line
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf
  # --- END NEW: ROC Curve Plotting Function ---


  # --- 9. Generate DOCX Report (UPDATED with ROC Plot) ---

  # MODIFIED FUNCTION SIGNATURE to accept roc_plot_buffer

  # --- Final Execution Block (UPDATED) ---

  # Generate the ROC plot buffer
  roc_plot_buffer = create_roc_plot_bytes(y_test_clean, y_pred_proba, target_names)

  # Run the function to create and save the report (passing the new buffer)
  create_docx.create_docx_lightBGM(
    auc_score,
    accuracy,
    classification_report_str,
    top_feature_importance_list,
    importance_plot_buffer,
    training_log_str,
    roc_plot_buffer,
    output_path
  )