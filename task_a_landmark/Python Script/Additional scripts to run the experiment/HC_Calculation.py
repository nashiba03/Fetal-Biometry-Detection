#HC Calculation.py
import pandas as pd
import numpy as np


input_csv = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\taskA_predictions.csv"
output_csv = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\taskB_HC_results.csv"


df = pd.read_csv(input_csv)


df.columns = df.columns.str.lower()


df["hc"] = np.pi * (df["bpd"] + df["ofd"]) / 2


df.to_csv(output_csv, index=False)

print("Task-B completed")
print(f"Saved as {output_csv}")
