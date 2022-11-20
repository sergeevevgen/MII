import pandas as pd
import numpy as np

# df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
# data = np.random.randint(1, 20, size=len(df.index))
# df['duration'] = data
# df.to_csv("Летающие тарелки (Зона 51)/nuforc_reports_new.csv", index=False)
df_linear_reg = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
data = []
df_linear_reg['date_time'] = pd.to_datetime(df_linear_reg['date_time'], format='%Y/%m/%dT%H:%M:%S').copy()
df_linear_reg['date_time'] = df_linear_reg['date_time'].dt.strftime('%m')
for (i, j) in df_linear_reg['date_time'].iteritems():
    if j == '01' or j == '02' or j == '12':
        data.append(1)
    if j == '03' or j == '04' or j == '05':
        data.append(2)
    if j == '06' or j == '07' or j == '08':
        data.append(3)
    if j == '09' or j == '10' or j == '11':
        data.append(4)

k = round(len(df_linear_reg.axes[0]) * 0.99)

x = np.array(data[:k])
k = x.size

df_linear_reg = df_linear_reg.head(k)
df_linear_reg['season'] = data

df_linear_reg.to_csv("Летающие тарелки (Зона 51)/nuforc_reports_regression.csv", index=False)
