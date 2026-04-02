from pandas import DataFrame
from pathlib import Path
from src.Classical.create_docx import create_docx_bagging_classifier

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import io

import seaborn as sns

def bagging_classifier(df:DataFrame, output_path:Path, name:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_col = ""
    if name == "dataset": target_col = "class"
    elif name == "CTU": target_col = "Label"
    elif name == "IDS2017": target_col = "label"
    elif name == "ToN-IoT": target_col = "type"
    else:
        print("data name not found.")
        return

    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))
    target_names = le_target.classes_

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    
    if name == "dataset":
        cols_to_drop = [
            "frame.time_delta","frame.time_relative","frame.len","ip.src","ip.dst",
            "tcp.srcport","tcp.dstport","tcp.flags","tcp.time_delta","tcp.len",
            "tcp.ack","tcp.connection.fin","tcp.connection.rst","tcp.connection.sack","tcp.connection.syn",
            "tcp.flags.ack","tcp.flags.fin","tcp.flags.push","tcp.flags.reset","tcp.flags.syn",
            "tcp.flags.urg","tcp.hdr_len","tcp.payload","tcp.pdu.size","tcp.window_size_value",
            "tcp.checksum","mqtt.clientid","mqtt.clientid_len","mqtt.conack.flags","mqtt.conack.val",
            "mqtt.conflag.passwd","mqtt.conflag.qos","mqtt.conflag.reserved","mqtt.conflag.retain","mqtt.conflag.willflag",
            "mqtt.conflags","mqtt.dupflag","mqtt.hdrflags","mqtt.kalive","mqtt.len",
            "mqtt.msg","mqtt.msgtype","mqtt.qos","mqtt.retain","mqtt.topic",
            "mqtt.topic_len","mqtt.ver","mqtt.willmsg_len","ip.proto","ip.ttl",
            "class","label"
        ]
    elif name == "CTU":
        cols_to_drop = [
            "StartTime","Dur","Proto","SrcAddr","Sport",
            "Dir","DstAddr","Dport","State","sTos",
            "dTos","TotPkts","TotBytes","SrcBytes","Label"
        ]
    elif name == "IDS2017":
        cols_to_drop = [
            "Flow ID","Source IP","Source Port","Destination IP","Destination Port",
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
            "Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","label"
        ]
    elif name == "ToN-IoT":
        cols_to_drop = [
            "src_ip","dst_ip","proto","service","conn_state",
            "missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes",
            "dns_query","dns_qclass","dns_qtype","dns_rcode","dns_AA",
            "dns_RD","dns_RA","dns_rejected","ssl_version","ssl_cipher",
            "ssl_resumed","ssl_established","ssl_subject","ssl_issuer","http_trans_depth",
            "http_method","http_uri","http_version","http_request_body_len","http_response_body_len",
            "http_status_code","http_user_agent","http_orig_mime_types","http_resp_mime_types","weird_name",
            "weird_addl","weird_notice","label","type"
        ]

    X = df.drop(columns=cols_to_drop, errors='ignore')

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            upper_bound = X[col].quantile(0.999)
            lower_bound = X[col].quantile(0.001)
            X[col] = np.clip(X[col], a_min=lower_bound, a_max=upper_bound)

    X = X.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ## Bagging Classifier
    base_clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    bg_clf = BaggingClassifier(estimator=base_clf, n_estimators=100, max_features=0.8, random_state=42)
    bg_clf.fit(X_train, y_train)
    y_pred_random = bg_clf.predict(X_test)

    # 5. Evaluate
    print("\n__Bagging Classifier Report (y_pred_random)__")
    bg_report_str = classification_report(y_test, y_pred_random, target_names=target_names, zero_division=0)
    print(bg_report_str)

    bg_accuracy = accuracy_score(y_test, y_pred_random)
    print("Bagging Classifier Accuracy: {:.2f}".format(bg_accuracy))

    def plot_confusion_matrix_bytes(cm, class_names):
        """Generates a heatmap plot of the confusion matrix and returns it as bytes."""
        plt.figure(figsize=(10, 8))
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
        plt.title('Bagging classifier confusion matrix')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        return buf

    cm = confusion_matrix(y_test, y_pred_random)

    print("__Generating Confusion Matrix Image for DOCX__")
    cm_plot_buffer = plot_confusion_matrix_bytes(cm, target_names) # Generate the image buffer

    create_docx_bagging_classifier(bg_report_str, bg_accuracy, cm_plot_buffer, output_path)