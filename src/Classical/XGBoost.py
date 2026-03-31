from pandas import DataFrame
from pathlib import Path
from src.Classical.create_docx import create_docx_XGBoost

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import time
import os

def XGBoost(df:DataFrame, output_path:Path, name:str) -> None:
    # ==========================================
    # 1. CẤU HÌNH & HÀM HỖ TRỢ (dataset CTU)
    # ==========================================
    files = {
        'menti.csv': 'Menti',
        'murlo.csv': 'Murlo',
        'neris.csv': 'Neris',
        'nsisay.csv': 'Nsisay',
        'rbot.csv': 'Rbot',
        'sogou.csv': 'Sogou',
        'virut.csv': 'Virut'
    }

    # Hàm xử lý lỗi cổng (Port) chứa ký tự lạ (VD: 0x6604 -> 26116)
    def clean_port_data(x):
        try:
            return float(x)
        except:
            try:
                # Thử chuyển từ Hex sang Int
                return int(str(x), 16)
            except:
                return 0


    # ==========================================
    # 2. TẢI VÀ LÀM SẠCH DỮ LIỆU (ĐÃ SỬA LỖI NAN/INF)
    # ==========================================
    data_frames = []
    print("Đang tải và xử lý dữ liệu thô...")

    for file_name, label_name in files.items():
        if os.path.exists(file_name):
            try:
                # Đọc toàn bộ dưới dạng string để kiểm soát lỗi
                df = pd.read_csv(file_name, dtype=str)
                df['Target_Label'] = label_name
                data_frames.append(df)
                print(f"-> Đã tải: {file_name} ({len(df)} dòng)")
            except Exception as e:
                print(f"Lỗi đọc file {file_name}: {e}")
        else:
            print(f"Cảnh báo: Không tìm thấy {file_name}")

    if not data_frames:
        raise ValueError("Không có dữ liệu nào được tải!")

    full_df = pd.concat(data_frames, ignore_index=True)

    print("\nĐang làm sạch dữ liệu (Fix lỗi NaN/Infinity)...")

    # 1. Xử lý cột Port
    full_df['Sport'] = full_df['Sport'].apply(clean_port_data)
    full_df['Dport'] = full_df['Dport'].apply(clean_port_data)

    # 2. Ép kiểu số và XỬ LÝ INFINITY
    numeric_cols = ['Dur', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes']
    for col in numeric_cols:
        if col in full_df.columns:
            # Chuyển sang số, lỗi thành NaN
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

            # QUAN TRỌNG: Thay thế vô cực (inf) bằng NaN tạm thời
            full_df[col] = full_df[col].replace([np.inf, -np.inf], np.nan)

            # Điền 0 vào tất cả các ô NaN (bao gồm cả ô vừa là inf)
            full_df[col] = full_df[col].fillna(0)

    # 3. Xóa cột thừa
    cols_to_drop = ['StartTime', 'SrcAddr', 'DstAddr', 'Label']
    full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns], inplace=True)

    # 4. Xóa trùng lặp
    full_df.drop_duplicates(inplace=True)

    # 5. Mã hóa Categorical
    cat_cols = ['Proto', 'Dir', 'State']
    for col in cat_cols:
        if col in full_df.columns:
            # Chuyển về string trước khi mã hóa để tránh lỗi mixed types
            full_df[col] = LabelEncoder().fit_transform(full_df[col].astype(str))

    # 6. Rà soát cuối cùng: Đảm bảo không còn bất kỳ giá trị NaN nào trong toàn bộ bảng
    full_df.fillna(0, inplace=True)
    print("- Đã xử lý triệt để các giá trị lỗi (NaN/Inf).")

    # ==========================================
    # 3. CHUẨN BỊ TRAINING
    # ==========================================
    X = full_df.drop(columns=['Target_Label'])
    y = LabelEncoder().fit_transform(full_df['Target_Label'])

    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Chuẩn hóa (Scale)
    scaler = StandardScaler()

    # Kiểm tra kỹ lại lần cuối trước khi đưa vào Scaler
    X_train = np.nan_to_num(X_train) # Chuyển đổi an toàn: NaN -> 0, Inf -> số cực lớn
    X_test = np.nan_to_num(X_test)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nDữ liệu sẵn sàng cho SVM: {X_train.shape}")

    # ... (Tiếp tục phần 4. CẤU HÌNH MODEL & TRAINING LOOP như cũ) ...

    # ==========================================
    # 4. CẤU HÌNH MODEL & TRAINING LOOP
    # ==========================================
    EPOCHS = 20

    # XGBoost Config
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': len(class_names),
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'max_depth': 6,
        'verbosity': 0
    }
    xgb_model = None

    # SVM Config (SGD)
    svm_model = SGDClassifier(loss='hinge', max_iter=1, tol=None, warm_start=True, random_state=42)

    history = {'xgb': [], 'svm': []}

    print("\n" + "="*40)
    print("   BẮT ĐẦU HUẤN LUYỆN (EPOCH MODE)")
    print("="*40)

    for epoch in range(1, EPOCHS + 1):
        start_ep = time.time()

        # --- Train XGBoost (1 round) ---
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1, xgb_model=xgb_model)
        # Eval XGB
        preds = xgb_model.predict(dtest)
        y_pred_xgb = np.argmax(preds, axis=1)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)

        # --- Train SVM (1 pass) ---
        svm_model.fit(X_train_scaled, y_train)
        # Eval SVM
        y_pred_svm = svm_model.predict(X_test_scaled)
        acc_svm = accuracy_score(y_test, y_pred_svm)

        history['xgb'].append(acc_xgb)
        history['svm'].append(acc_svm)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Time: {time.time()-start_ep:.2f}s")
        print(f"   [XGBoost] Acc: {acc_xgb:.4f}")
        print(f"   [SVM-SGD] Acc: {acc_svm:.4f}")
        print("-" * 40)

    # ==========================================
    # 5. ĐÁNH GIÁ & VẼ BIỂU ĐỒ
    # ==========================================

    # A. Classification Report
    print("\n=== XGBoost Detailed Report ===")
    print(classification_report(y_test, y_pred_xgb, target_names=class_names))
    print("\n=== SVM Detailed Report ===")
    print(classification_report(y_test, y_pred_svm, target_names=class_names))

    # B. Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('XGBoost Confusion Matrix')

    sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('SVM Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # C. ROC Curve Plot
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    y_score_xgb = xgb_model.predict(dtest)
    y_score_svm = svm_model.decision_function(X_test_scaled)

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])

    for i, color in zip(range(len(class_names)), colors):
        if np.sum(y_test_bin[:, i]) == 0: continue

        # XGB ROC
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_xgb[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'XGB {class_names[i]} (AUC={auc_score:.2f})')

        # SVM ROC (Dashed)
        fpr_s, tpr_s, _ = roc_curve(y_test_bin[:, i], y_score_svm[:, i])
        plt.plot(fpr_s, tpr_s, color=color, lw=2, linestyle='--', alpha=0.5)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve Comparison: XGBoost (Solid) vs SVM (Dashed)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', ncol=2, fontsize='small')
    plt.show()

    create_docx_XGBoost()