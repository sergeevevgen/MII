import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
from random import randint
import numpy as np

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
data = np.random.randint(1, 181, size=len(df.index))
df['duration'] = data
df = df.drop(df.columns[0], axis=1)
new_df = df.iloc[:len(df.index), :len(df.columns)]
new_df.to_csv("Летающие тарелки (Зона 51)/nuforc_reports_new.csv")
