#Load & Merge Attack and Sensor Data
import pandas as pd
import sys
from pathlib import Path

def shuffle():
    print("select data")
    print("1:dataset    2:CTU  3:IDS2017   4:ToN-IoT")

    i = int(input())
    csv_files = []
    script_location = Path(__file__).resolve().parent
    name = script_location / "invalidData.csv"

    # Load the dataset files from personal drive
    if i <= 1:
        csv_files = [
            script_location / "dataset/Attack.csv",
            script_location / "dataset/environmentMonitoring.csv",
            script_location / "dataset/patientMonitoring.csv"
        ]
        name = script_location / "shuffled_datas" / "dataset.csv"
    elif i == 2:
        csv_files = [
            script_location / "./CTU/menti.csv",
            script_location / "./CTU/murlo.csv",
            script_location / "./CTU/neris.csv",
            script_location / "./CTU/nsisay.csv",
            script_location / "./CTU/rbot.csv",
            script_location / "./CTU/sogou.csv",
            script_location / "./CTU/virut.csv"
        ]
        name = script_location / "shuffled_datas" / "CTU.csv"
    elif i == 3:
        csv_files = [
            script_location / "./IDS2017/bot.csv",
            script_location / "./IDS2017/ddos.csv",
            script_location / "./IDS2017/dos_goldeneye.csv",
            script_location / "./IDS2017/dos_hulk.csv",
            script_location / "./IDS2017/dos_slowhttptest.csv",
            script_location / "./IDS2017/dos_slowloris.csv",
            script_location / "./IDS2017/ftp_patator.csv",
            script_location / "./IDS2017/heartbleed.csv",
            script_location / "./IDS2017/infiltration.csv",
            script_location / "./IDS2017/port_scan.csv",
            script_location / "./IDS2017/ssh_patator.csv",
            script_location / "./IDS2017/web_attack_bruteforce.csv",
            script_location / "./IDS2017/web_attack_sql_injection.csv",
            script_location / "./IDS2017/web_attack_xss.csv"
        ]
        name = script_location / "shuffled_datas" / "IDS2017.csv"
    elif i >= 4:
        csv_files = [
            script_location / "./ToN-IoT/backdoor.csv",
            script_location / "./ToN-IoT/benign_tot.csv",
            script_location / "./ToN-IoT/ddos.csv",
            script_location / "./ToN-IoT/dos.csv",
            script_location / "./ToN-IoT/injection.csv",
            script_location / "./ToN-IoT/mitm.csv",
            script_location / "./ToN-IoT/password.csv",
            script_location / "./ToN-IoT/ransomware.csv",
            script_location / "./ToN-IoT/scanning.csv",
            script_location / "./ToN-IoT/xss.csv"
        ]
        name = script_location / "shuffled_datas" / "ToN-IoT.csv"
    else:
        print("There does not exist a dataset at index " + i)
        sys.exit(0)

    # Load and concatenate all datasets
    dfs = [pd.read_csv(file, low_memory=False) for file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)


    # Shuffle dataset to prevent ordering bias
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_df.to_csv(name, index=False)

    # Load shuffled dataset
    df = pd.read_csv(name)

    # Extract labels and remove categorical columns
    if "label" in df.columns:
        y = df["label"]
    columns_to_drop = ["label"]
    if "Label" in df.columns:
        y = df["Label"]
        columns_to_drop = ["Label"]
    if "class" in df.columns:
        columns_to_drop.append("class")
    if "type" in df.columns:
        columns_to_drop.append("type")
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Convert categorical and object columns to numeric (handling mixed types)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df.fillna(0, inplace=True)