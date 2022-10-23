import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
from random import randint
import numpy as np

df = pd.read_csv("Летающие тарелки (Зона 51)/nuforc_reports.csv", delimiter=',')
app = Flask(__name__)
about = "Полный текст и геокодированные отчеты о наблюдениях НЛО от Национального центра исследований НЛО (NUFORC). " \
        "Национальный центр исследований НЛО (NUFORC) собирает и обслуживает более 100 000 сообщений о наблюдениях " \
        "НЛО. Этот набор данных содержит само содержимое отчета, включая время, длительность местоположения и другие " \
        "атрибуты, как в необработанном виде, как оно записано на сайте NUFORC, так и в уточненной " \
        "стандартизированной форме, которая также содержит координаты широты. "
link = "http://127.0.0.1:5000"
about_filtration = "Фильтрация по "


@app.route("/")
def index():
    return render_template("index.html", link=link)


@app.route("/get_data", methods=['GET'])
def get_data():
    data = request.args
    a = int(data['columns'].split(',')[0]) - 1
    b = int(data['columns'].split(',')[1]) + 1
    c = int(data['rows'].split(',')[0]) - 1
    d = int(data['rows'].split(',')[1]) + 1
    new_df = df.iloc[c: d, a: b]
    return render_template("datatable.html", link=link, column_names=new_df.columns.values,
                           row_data=list(new_df.values.tolist()), column_types=new_df.dtypes,
                           null_values=new_df.isnull().sum(axis=0).array, about=about, zip=zip)


@app.route("/filters", methods=['GET'])
def filters():
    data = request.args
    str_start = int(data['rows'].split(',')[0]) - 1
    str_end = int(data['rows'].split(',')[1]) + 1
    new_df = df.loc[str_start: str_end]
    new_df = filtration(data['filter'], new_df)

    return render_template("filter_table.html", df_data=list(new_df.values.tolist()), df_names=new_df.columns.values,
                           title_info=about_filtration + data['filter'], zip=zip)


# Function for grouping data
def filtration(filter_arg, new_df):
    if filter_arg == 'duration':
        min_duration = new_df[filter_arg].min()
        max_duration = new_df[filter_arg].max()
        mean_duration = new_df[filter_arg].mean()
        data = {'': ['min', 'max', 'mean'], filter_arg: [min_duration, max_duration, mean_duration]}
        return pd.DataFrame(data)

    if filter_arg == 'difference':
        new_df['posted'] = pd.to_datetime(new_df['posted'], format='%Y/%m/%dT%H:%M:%S').copy()
        new_df['date_time'] = pd.to_datetime(new_df['date_time'], format='%Y/%m/%dT%H:%M:%S').copy()
        new_df[filter_arg] = new_df['posted'] - new_df['date_time']

        min_duration = new_df[filter_arg].min()
        max_duration = new_df[filter_arg].max()
        mean_duration = new_df[filter_arg].mean()
        data = {'': ['min', 'max', 'mean'], filter_arg: [min_duration, max_duration, mean_duration]}
        return pd.DataFrame(data)

    duration = new_df.groupby(filter_arg)[['duration']].min()
    duration.rename(columns={'duration': 'min'}, inplace=True)
    max_duration = new_df.groupby(filter_arg)[['duration']].max()
    mean_duration = new_df.groupby(filter_arg)[['duration']].mean()
    duration['max'] = max_duration['duration']
    duration['average'] = mean_duration['duration']
    # new_df[data['filter']] = new_df.axes[0].values
    duration[filter_arg] = duration.axes[0].values
    return duration


if __name__ == "__main__":
    app.run(debug=True)
