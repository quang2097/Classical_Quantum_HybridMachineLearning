import pandas as pd

df = pd.read_csv("./dataset/Attack.csv")

df["Attack"] = 1  # same value for all rows

df.to_csv("your_file_updated.csv", index=False)