import torch
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np

# --- New Imports for Plotting the Tree ---
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io

# NEW IMPORT: Seaborn for heatmap visualization
import seaborn as sns
#from google.colab import drive
#drive.mount('/content/drive')
from src.Classical import create_docx

from pathlib import Path


def decision_tree(df:DataFrame, output_path:Path, name:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le = LabelEncoder()

    # Convert categorical and object columns to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # --- Get the Target Names (CORRECT) ---
    # le.classes_ contains the array of original string labels in order of their integer encoding
    target_names = le.classes_

    if name == "dataset":
        X = df.drop(["frame.time_delta","frame.time_relative","frame.len","ip.src","ip.dst",
                     "tcp.srcport","tcp.dstport","tcp.flags","tcp.time_delta","tcp.len",
                     "tcp.ack","tcp.connection.fin","tcp.connection.rst","tcp.connection.sack","tcp.connection.syn",
                     "tcp.flags.ack","tcp.flags.fin","tcp.flags.push","tcp.flags.reset","tcp.flags.syn",
                     "tcp.flags.urg","tcp.hdr_len","tcp.payload","tcp.pdu.size","tcp.window_size_value",
                     "tcp.checksum","mqtt.clientid","mqtt.clientid_len","mqtt.conack.flags","mqtt.conack.val",
                     "mqtt.conflag.passwd","mqtt.conflag.qos","mqtt.conflag.reserved","mqtt.conflag.retain","mqtt.conflag.willflag",
                     "mqtt.conflags","mqtt.dupflag","mqtt.hdrflags","mqtt.kalive","mqtt.len",
                     "mqtt.msg","mqtt.msgtype","mqtt.qos","mqtt.retain","mqtt.topic",
                     "mqtt.topic_len","mqtt.ver","mqtt.willmsg_len","ip.proto","ip.ttl",
                     "class","label"], axis=1)
        y = df["class"]
    elif name == "CTU":
        X = df.drop(["StartTime","Dur","Proto","SrcAddr","Sport",
                     "Dir","DstAddr","Dport","State","sTos",
                     "dTos","TotPkts","TotBytes","SrcBytes","Label"], axis=1)
        y = df["Label"]
    elif name == "IDS2017":
        X = df.drop(["Flow ID","Source IP","Source Port","Destination IP","Destination Port",
                     "Protocol","Timestamp","Flow Duration","Total Fwd Packets","Total Backward Packets",
                     "Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min",
                     "Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean",
                     "Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std",
                     "Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std",
                     "Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std",
                     "Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags",
                     "Bwd URG Flags","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
                     "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
                     "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
                     "URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size",
                     "Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length.1","Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk",
                     "Fwd Avg Bulk Rate","Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets",
                     "Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward",
                     "act_data_pkt_fwd","min_seg_size_forward","Active Mean","Active Std","Active Max",
                     "Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","label"], axis=1)
        y = df["label"]
    elif name == "ToN-IoT":
        X = df.drop(["src_ip","dst_ip","proto","service","conn_state",
                     "missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes",
                     "dns_query","dns_qclass","dns_qtype","dns_rcode","dns_AA",
                     "dns_RD","dns_RA","dns_rejected","ssl_version","ssl_cipher",
                     "ssl_resumed","ssl_established","ssl_subject","ssl_issuer","http_trans_depth",
                     "http_method","http_uri","http_version","http_request_body_len","http_response_body_len",
                     "http_status_code","http_user_agent","http_orig_mime_types","http_resp_mime_types","weird_name",
                     "weird_addl","weird_notice","label","type"], axis=1)
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

    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)

    # Apply undersampling only to the training set
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    ## A. Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train_resampled, y_train_resampled)
    y_pred_decision = dt_clf.predict(X_test)

    # 5. Evaluate
    print("--- Decision Tree Report (y_pred_decision) ---")
    dt_report_str = classification_report(y_test, y_pred_decision, target_names=target_names, zero_division=0)
    print(dt_report_str)

    dt_accuracy = accuracy_score(y_test, y_pred_decision)
    print("\nDecision Tree Accuracy: {:.2f}".format(dt_accuracy))


    # --- FUNCTION: Create Decision Tree Plot (Optimized for DOCX) ---
    def generate_dt_plot_bytes(dt_clf, X, target_names):
        """Generates a limited-depth Decision Tree plot and returns it as bytes."""
        plt.figure(figsize=(25, 12))
        plot_tree(dt_clf,
            feature_names=X.columns.tolist(),
            class_names=target_names.tolist(),
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=3)
        plt.title("Decision Tree Visualization (Max Depth 3)", fontsize=16)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return buf

    # --- NEW FUNCTION: Create Confusion Matrix Plot ---
    def plot_confusion_matrix_bytes(cm, class_names):
        """Generates a heatmap plot of the confusion matrix and returns it as bytes."""
        plt.figure(figsize=(10, 8))
        # Use Seaborn for a clean, colored heatmap
        sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True
        )

        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Decision Tree Confusion Matrix')

        buf = io.BytesIO()
        # Save the figure to the buffer
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        return buf

    # Generate the plots before creating the report
    print("\n__Generating Decision Tree Visualization (Max Depth 3) for DOCX__")
    dt_plot_buffer = generate_dt_plot_bytes(dt_clf, X_train_resampled, target_names) # Use the correct target names

    cm = confusion_matrix(y_test, y_pred_decision)
    print("Decision Tree Confusion Matrix:\n", cm)

    print("__Generating Confusion Matrix Image for DOCX__")
    cm_plot_buffer = plot_confusion_matrix_bytes(cm, target_names) # Generate the image buffer

    # Run the function to create and save the report with the new confusion matrix image buffer
    create_docx.create_docx_decision_tree(dt_report_str, dt_plot_buffer, dt_accuracy, cm_plot_buffer, output_path)