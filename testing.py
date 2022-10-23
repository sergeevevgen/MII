import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
new_df = df.iloc[0: 100, 0: 12]
# sns.set(style="whitegrid")
# fig, scatter = plt.subplots(figsize=(18, 7))
# sns.barplot(data=new_df, x="city", y="duration")
# plt.show()
sns.set(rc={"figure.figsize": (7.5, 4)}, font_scale=0.7)
bplot = sns.boxplot(data=new_df, x="shape", y="duration", width=0.5)
bplot.axes.set_title("Shapes", fontsize=16)
plt.show()
bplot.figure.savefig("1.jpg", format="jpeg", dpi=100)

# sns.boxplot(data=new_df, x="state", y="duration")
# plt.show()
