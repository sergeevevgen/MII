import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
from random import randint
import numpy as np

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
data = np.random.randint(1, 20, size=len(df.index))
df['duration'] = data
df.to_csv("Летающие тарелки (Зона 51)/nuforc_reports_new.csv", index=False)
