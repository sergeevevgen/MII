import numpy as np
import pandas as pd

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
column_names = df.columns.array
column_type = df.dtypes.array

a = []
for i in range(0, len(column_names)):
    k = df[column_names[i]].mode()[0]
    a.append(k)
    # elif column_type[i] == 'int64':
    #     k = df[column_names[i]].mean()
    #     a.append(k)
    # elif column_type[i] == 'float64':
    #     k = df[column_names[i]].mean()
    #     a.append(k)

for i in range(0, int(len(df.index) * 0.1)):
    df.loc[-1] = a
    df.index = df.index + 1
    df.sort_index()

df.to_csv("Летающие тарелки (Зона 51)/nuforc_reports_new.csv", index=False)
